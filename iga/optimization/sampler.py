import torch
from iga.utils.common_utils import *
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm


def get_sampling_method(config, global_step):
    if config['langevin_every_n_steps'] > 0 and \
            global_step % config['langevin_every_n_steps'] == 0 and \
            global_step > config['random_warm_up']:
        sampling_method = 'langevin'
    else:
        sampling_method = 'der_free'
    return sampling_method


def get_sampling_scale(config, global_step):
    if get_sampling_method(config, global_step) == 'langevin':
        return config['pos_ub_global'] * config['pcd_scaling'], config['rot_ub_global']
    else:
        if config['local_perturb_every_n_steps'] > 0 and \
                global_step % config['local_perturb_every_n_steps'] == 0 and \
                global_step > config['random_warm_up']:
            return config['pos_ub_local'] * config['pcd_scaling'], config['rot_ub_local']
        else:
            return config['pos_ub_global'] * config['pcd_scaling'], config['rot_ub_global']


class Sampler:

    def __init__(self, forward_model, perturb_func, graph_mean_func, config, device='cuda'):
        self.forward_model = forward_model
        self.graph_mean_func = graph_mean_func
        self.perturb_func = perturb_func
        self.config = config
        self.device = torch.device(device)
        ############################################################################

    def sample_random_poses(self, num_poses, dof, identity_idx=None):
        poses = torch.zeros(num_poses, 6, device=self.device)
        poses[:, :3] = -2 * self.config['pos_ub'] * torch.rand(num_poses, 3, device=self.device) + self.config['pos_ub']
        poses[:, 3:] = -2 * self.config['rot_ub'] * torch.rand(num_poses, 3, device=self.device) + self.config['rot_ub']
        if identity_idx is not None:
            poses[identity_idx] = torch.zeros(6, device=self.device)

        return poses[:, dof]

    def sample_der_free(self, batch, num_negatives, mode='train', batch_size=1):
        num_samples = (num_negatives + 1) * batch_size
        positive_idx = torch.arange(0, batch_size, device=self.device)
        pose = self.sample_random_poses(num_samples, dof=self.config[f'dof_{mode}'], identity_idx=positive_idx)

        T_noise = self.pose_to_transform(pose, self.config[f'dof_{mode}'])

        batch.pose = pose
        return self.perturb_func(batch, T_noise, num_negatives)

    def pose_to_transform(self, pose, dofs):
        # Create a set of 4x4 matrices
        pose_6d = torch.zeros((pose.shape[0], 6), device=self.device)
        pose_6d[:, dofs] = pose
        T = angle_axis_to_rotation_matrix(pose_6d[:, 3:])
        T[:, :3, 3] = pose_6d[:, :3]
        return T

    def transform_to_pose(self, transform, pose_dof):
        poses = torch.cat([transform[:, :3, 3], matrix_to_axis_angle(transform[:, :3, :3])], dim=-1)
        return poses[:, pose_dof]

    def langevin_step(self, batch, num_negatives, dof=2, noise_scale=0.5, step_size=0.01, batch_size=1):
        # Keeping track of the gradient of the pose vector. Each iteration we start with a zero vector,
        # because we iteratively perturb the batch. In this way, it's the same doing gradient descent on SE(3) manifold.
        num_samples = (num_negatives + 1) * batch_size
        positive_idx = torch.arange(0, batch_size, device=self.device)
        pose = 0 * self.sample_random_poses(num_samples, dof=dof, identity_idx=positive_idx) - 1e-12
        pose.requires_grad = True

        # Convert the vector to a 4x4 transformation matrix
        T_step = self.pose_to_transform(pose, dof)

        # Perturb the batch input with the noise
        batch.pose = pose
        batch = self.perturb_func(batch, T_step, num_negatives)
        # Compute the energy of the perturbed batch
        energy, labels, labels_sorted_idx, batch_sorted = self.forward_model(batch, return_idx=True)

        # We need this reduction because output must be scalar for autograd to backpropagate.
        total_energy = energy[labels == 0].sum()

        grad, = torch.autograd.grad(outputs=total_energy, inputs=pose, create_graph=False, retain_graph=False,
                                    allow_unused=False)

        # Important to detach the tensors from the graph, otherwise the graph will be too large and will cause memory issues.
        total_energy = total_energy.detach()
        pose = pose.detach()
        batch = batch.detach()
        # For some reason, some gradients can be NaN. This is a hack to avoid this.
        # Generally, this should not happen. If it does, it's a sign that something is wrong.
        if torch.isnan(grad).any():
            print('NaN in gradient')
            grad[torch.isnan(grad)] = 0

        # Clip the gradient to avoid exploding gradients
        grad = torch.clamp(grad, -1, 1)

        # Noise in the Langevin Dynamics
        noise = torch.randn_like(pose) * noise_scale

        # Update the pose vector.
        # Because we initialise pose to be "near" zero. This update is the same as doing grad descent on SE(3) Manifold.
        pose = pose - step_size * (0.5 * grad) + noise
        pose[positive_idx] = 0
        return self.pose_to_transform(pose.detach(), dof), labels_sorted_idx, batch_sorted

    def sample_langevin(self, batch, batch_size=1, mode='train', return_trajectory=False,
                        num_negatives=None, perturb_init=True, progress_bar=False):
        ##################################################################################################
        if num_negatives is None:
            num_negatives = self.config[f'num_negatives_{mode}']
        ##################################################################################################
        # Perturbing the batch with a random transformation to create random initial noise.
        num_samples = (num_negatives + 1) * batch_size
        positive_idx = torch.arange(0, batch_size, device=self.device)

        T_noise = torch.eye(4, device=self.device).repeat(num_samples, 1, 1)
        traj = [torch.eye(4, device=self.device).repeat(T_noise.shape[0], 1, 1)]
        graph_means = [self.graph_mean_func(batch)]
        if perturb_init:
            pose = self.sample_random_poses(num_samples, dof=self.config[f'dof_{mode}'], identity_idx=positive_idx)
            T_noise = self.pose_to_transform(pose, self.config[f'dof_{mode}'])

            if return_trajectory:
                traj = [T_noise.clone()]
                graph_means = [self.graph_mean_func(batch)]
            batch = self.perturb_func(batch, T_noise, num_negatives)

        T_noise_init = T_noise.clone()
        ##################################################################################################
        step_size_init = self.config[f'step_size_init_{mode}']
        noise_scale_init = self.config[f'noise_scale_init_{mode}']
        ##################################################################################################
        if progress_bar:
            pbar = tqdm(total=self.config[f'num_langevin_steps_{mode}'],
                        desc=f'Langevin sampling {self.config["mode"]}', leave=False)
        # Performing Langevin dynamics to sample from the distribution
        for i in range(self.config[f'num_langevin_steps_{mode}']):
            step_size = step_size_init * np.exp(-self.config[f'step_size_decay_{mode}'] * i)
            noise_scale = noise_scale_init * np.exp(-self.config[f'noise_decay_{mode}'] * i)
            T_step, labels_sorted_idx, batch_sorted = self.langevin_step(batch.clone(),
                                                                         num_negatives,
                                                                         self.config[f'dof_{mode}'],
                                                                         noise_scale,
                                                                         step_size,
                                                                         batch_size=batch_size)

            if return_trajectory:
                T_traj_step, _ = to_dense_batch(T_step[labels_sorted_idx], batch_sorted)
                traj.append(T_traj_step.detach())
                means, _ = to_dense_batch(self.graph_mean_func(batch)[labels_sorted_idx].detach(), batch_sorted)
                graph_means.append(means.detach())

            T_noise[:, :3, :3] = torch.bmm(T_step[:, :3, :3], T_noise[:, :3, :3])
            T_noise[:, :3, 3] = T_step[:, :3, 3] + T_noise[:, :3, 3]

            if i == self.config[f'num_langevin_steps_{mode}'] - 1:
                batch.pose = self.transform_to_pose(T_noise, self.config[f'dof_{mode}'])

            batch = self.perturb_func(batch, T_step.detach(), num_negatives)

            if progress_bar:
                pbar.update(1)

        if progress_bar:
            pbar.close()
        if not return_trajectory:
            return batch
        energy, labels, labels_sorted_idx, batch_sorted = self.forward_model(batch, return_idx=True)
        lowest_energy_idx = torch.argmin(energy + 10e20 * labels, dim=1)
        T_noise_new, _ = to_dense_batch(T_noise[labels_sorted_idx], batch_sorted)

        self.mean_gt_energy = torch.mean(energy.detach().flatten()[labels.flatten() == 0])
        self.mean_min_energy = torch.mean(torch.min(energy.detach(), dim=1)[0])

        # We need to adjust the first transform in traj by reshaping it according to batch
        traj[0], _ = to_dense_batch(traj[0][labels_sorted_idx], batch_sorted)
        graph_means[0], _ = to_dense_batch(graph_means[0][labels_sorted_idx], batch_sorted)
        return batch, T_noise_new, lowest_energy_idx, torch.stack(traj), torch.stack(graph_means), T_noise_init

    def sample(self, batch, batch_size, config, sampling_mehtod='der_free', mode='train', return_trajectory=False):

        if sampling_mehtod.lower() == 'der_free':
            return self.sample_der_free(batch, config[f'num_negatives_{mode}'], mode=mode, batch_size=batch_size)
        elif sampling_mehtod.lower() == 'langevin':
            return self.sample_langevin(batch, batch_size, mode=mode, return_trajectory=return_trajectory)
        else:
            raise NotImplementedError('Sampling method not implemented.')
