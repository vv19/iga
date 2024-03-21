import time

from iga.models.ebm import EBM
import json
from iga.utils.alignment_dataset import AlignDataset
from torch_geometric.data import DataLoader
import torch
from iga.models.ebm import add_spec_norm, dfs_freeze
from scipy.spatial.transform import Rotation as Rot
import numpy as np
from torch_geometric.data import Data
from iga.utils.alignment_dataset import create_clustered_data_sample, LocalFeatureEncoder
from torch_geometric.nn import knn
import os
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class IGA:

    def __init__(self,
                 trans_model_path,
                 rot_model_path,
                 num_negatives_trans=None,  # If None, use what is in the config.
                 num_steps_trans=None,  # If None, use what is in the config.
                 step_size_trans=None,  # If None, use what is in the config.
                 step_size_decay_trans=None,  # If None, use what is in the config.
                 noise_scale_init_trans=None,  # If None, use what is in the config.
                 noise_decay_trans=None,  # If None, use what is in the config.

                 num_negatives_rot=None,  # If None, use what is in the config.
                 num_steps_rot=None,  # If None, use what is in the config.
                 step_size_rot=None,  # If None, use what is in the config.
                 step_size_decay_rot=None,  # If None, use what is in the config.
                 noise_scale_init_rot=None,  # If None, use what is in the config.
                 noise_decay_rot=None,  # If None, use what is in the config.

                 dof_rot=(False, False, False, True, True, True),
                 dof_trans=(True, True, True, False, False, False),

                 device='cuda'
                 ):
        self.device = device

        self.local_feature_encoder = LocalFeatureEncoder(aggr='mean')
        self.number_local_centers = 8
        self.local_neighbours = 20
        self.scaling_factor_local = 100
        self.scaling_factor_global = 100

        self.ebm_trans = self.load_model(trans_model_path,
                                         config_path='/'.join(trans_model_path.split('/')[:-1]) + '/config_trans.json'
                                         )
        self.ebm_rot = self.load_model(rot_model_path,
                                       config_path='/'.join(rot_model_path.split('/')[:-1]) + '/config_rot.json')

        self.ebm_rot.sampler.config['dof_val'] = dof_rot
        self.ebm_trans.sampler.config['dof_val'] = dof_trans

        self.visualiser = self.ebm_trans.visualiser

        # Overwrite the config values if the user has provided them.
        if num_negatives_trans is not None:
            self.ebm_trans.sampler.config['num_negatives_val'] = num_negatives_trans
        if num_steps_trans is not None:
            self.ebm_trans.sampler.config['num_langevin_steps_val'] = num_steps_trans
        if step_size_trans is not None:
            self.ebm_trans.sampler.config['step_size_init_val'] = (step_size_trans
                                                                   * self.ebm_trans.sampler.config['pcd_scaling'])
        if step_size_decay_trans is not None:
            self.ebm_trans.sampler.config['step_size_decay_val'] = step_size_decay_trans
        if noise_scale_init_trans is not None:
            self.ebm_trans.sampler.config['noise_scale_init_val'] = (
                    noise_scale_init_trans * self.ebm_trans.sampler.config['pcd_scaling'])
        if noise_decay_trans is not None:
            self.ebm_trans.sampler.config['noise_decay_val'] = noise_decay_trans

        if num_negatives_rot is not None:
            self.ebm_rot.sampler.config['num_negatives_val'] = num_negatives_rot
        if num_steps_rot is not None:
            self.ebm_rot.sampler.config['num_langevin_steps_val'] = num_steps_rot
        if step_size_rot is not None:
            self.ebm_rot.sampler.config['step_size_init_val'] = step_size_rot
        if step_size_decay_rot is not None:
            self.ebm_rot.sampler.config['step_size_decay_val'] = step_size_decay_rot
        if noise_scale_init_rot is not None:
            self.ebm_rot.sampler.config['noise_scale_init_val'] = noise_scale_init_rot
        if noise_decay_rot is not None:
            self.ebm_rot.sampler.config['noise_decay_val'] = noise_decay_rot

    def load_model(self, model_path, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        config['pre_trained_local_encoder'] = False
        config['record'] = False
        config['spectral_norm'] = False
        config['perturb_initial_pos'] = False
        config['local_nn_dims'][-1] = config['hidden_dim']
        config['log_images'] = False

        model = EBM(config)

        checkpoint = torch.load(model_path)
        if model.use_ema:
            model.graph_encoder.load_state_dict(checkpoint['state_dict_ema_graph_encoder'], strict=False)
            model.energy_predictor.load_state_dict(checkpoint['state_dict_ema_energy_predictor'], strict=False)
        else:
            model.graph_encoder.load_state_dict(checkpoint['state_dict_graph_encoder'], strict=False)
            model.energy_predictor.load_state_dict(checkpoint['state_dict_energy_predictor'], strict=False)

        add_spec_norm(model.energy_predictor)
        add_spec_norm(model.graph_encoder)

        model.use_ema = False
        model.local_encoder.load_state_dict(checkpoint['state_dict_local_encoder'], strict=False)

        # Freeze the weights, otherwise memory usage increases over time.
        dfs_freeze(model.local_encoder)
        dfs_freeze(model.graph_encoder)
        dfs_freeze(model.energy_predictor)

        model.train()  # Eval breaks the gradient propagation for Langevin sampling.
        model.to(self.device)
        ################################################################################################################
        # Initialise the model with random data.
        model.initialise()
        ################################################################################################################
        return model

    def get_transform(self, demo_pcds, live_pcds, visualise=True, overlay=False, vis_graph=False):
        context_length = len(demo_pcds['pcds_grasped']) + 1
        self.ebm_trans.config['context_length'] = context_length
        self.ebm_trans.local_encoder.initialise(context_length)
        self.ebm_rot.config['context_length'] = context_length
        self.ebm_rot.local_encoder.initialise(context_length)

        data = self.sample_to_data(demo_pcds, live_pcds).to(self.device)
        ################################################################################################################
        # Initialise the translation as the mean of the first demo point cloud.
        mean_b_1 = data.pos_b_1.mean(dim=0)
        mean_b_0 = data.pos_b_0.mean(dim=0)
        centering = -mean_b_1 + mean_b_0
        data.pos_b_0 -= centering
        data.centres_b_0 -= centering
        T_centering_init = torch.eye(4, device=mean_b_0.device).unsqueeze(0)
        T_centering_init[:, :3, 3] = -centering
        ################################################################################################################
        # Optimise the rotation.
        _, T_noise_new, lowest_energy_idx, traj_rot, graph_means_rot, T_noise = \
            self.ebm_rot.inference(data, visualise=False, progress_bar=True, ret_optim_results=True)
        best_T_rot = T_noise_new[0][lowest_energy_idx]  # @ T_centering_init[0]
        traj_rot, graph_means_rot = (traj_rot[:, 0, lowest_energy_idx.squeeze(), ...],
                                     graph_means_rot[:, 0, lowest_energy_idx.squeeze(), ...])
        ################################################################################################################
        # Transform the point cloud according to the optimised rotation.
        T_centering = torch.eye(4, device=best_T_rot.device)
        node_means = data.centres_b_0.mean(dim=0)
        T_centering[:3, 3] = -node_means.squeeze()
        data.pos_b_0 = (best_T_rot[0, :3, :3] @ (data.pos_b_0 - node_means).T).T + node_means
        data.centres_b_0 = (best_T_rot[0, :3, :3] @ (data.centres_b_0 - node_means).T).T + node_means
        data.features_b_0 = (best_T_rot[0, :3, :3] @ (data.features_b_0.view(-1, 3).T)).T.view(-1, 3, 3).contiguous()
        ################################################################################################################
        # Optimise the translation.
        _, T_noise_new, lowest_energy_idx, traj_trans, graph_means_trans, T_noise = \
            self.ebm_trans.inference(data, visualise=False, progress_bar=True, ret_optim_results=True)
        best_T_trans = T_noise_new[0][lowest_energy_idx]
        traj_trans, graph_means_trans = (traj_trans[:, 0, lowest_energy_idx.squeeze(), ...],
                                         graph_means_trans[:, 0, lowest_energy_idx.squeeze(), ...])
        ################################################################################################################
        if visualise:
            node_means = data.centres_b_0.mean(dim=0)
            data.pos_b_0 = (best_T_rot[0, :3, :3].T @ (data.pos_b_0 - node_means).T).T + node_means
            data.centres_b_0 = (best_T_rot[0, :3, :3].T @ (data.centres_b_0 - node_means).T).T + node_means
            traj_trans[:, :3, :3] = traj_rot[:, :3, :3]
            self.visualiser.visualise_trajectory(data, traj_trans, graph_means_trans,
                                                 context_length=self.ebm_trans.config['context_length'],
                                                 overlayed=overlay, graph=vis_graph)
        ################################################################################################################
        T_final = best_T_trans[0] @ torch.linalg.inv(T_centering) @ best_T_rot[0] @ T_centering @ T_centering_init[0]
        T_final[:3, 3] /= self.scaling_factor_global
        return np.linalg.inv(T_final.cpu().numpy())

    def sample_to_data(self, demo_pcds, live_pcds):
        data = Data()
        data = self.process_pcds(data, live_pcds['pcd_grasped'], live_pcds['pcd_target'], idx=0)

        for i in range(len(demo_pcds['pcds_grasped'])):
            data = self.process_pcds(data, demo_pcds['pcds_grasped'][i], demo_pcds['pcds_target'][i], idx=i + 1)

        return data

    def process_pcds(self, data, pcd_a, pcd_b, idx=0):
        pcd_a = torch.tensor(pcd_a, dtype=torch.float)
        pcd_b = torch.tensor(pcd_b, dtype=torch.float)
        centre_idx_a, point_idx_a, centres_a = create_clustered_data_sample(pcd_a,
                                                                            num_clusters=self.number_local_centers)
        id_k_neighbor = knn(pcd_a, pcd_a, k=self.local_neighbours)
        pcd_a_local = (pcd_a - centres_a[centre_idx_a]) * self.scaling_factor_local
        features_a = self.local_feature_encoder(pcd_a_local, id_k_neighbor).view(-1, 3, 3)

        centre_idx_b, point_idx_b, centres_b = create_clustered_data_sample(pcd_b,
                                                                            num_clusters=self.number_local_centers)

        pcd_b_local = (pcd_b - centres_b[centre_idx_b]) * self.scaling_factor_local
        id_k_neighbor = knn(pcd_b, pcd_b, k=self.local_neighbours)
        features_b = self.local_feature_encoder(pcd_b_local, id_k_neighbor).view(-1, 3, 3)
        setattr(data, f'pos_a_{idx}', pcd_a * self.scaling_factor_global)
        setattr(data, f'features_a_{idx}', features_a)
        setattr(data, f'centre_idx_a_{idx}', centre_idx_a)
        setattr(data, f'point_idx_a_{idx}', point_idx_a)
        setattr(data, f'centres_a_{idx}', centres_a * self.scaling_factor_global)

        setattr(data, f'pos_b_{idx}', pcd_b * self.scaling_factor_global)
        setattr(data, f'features_b_{idx}', features_b)
        setattr(data, f'centre_idx_b_{idx}', centre_idx_b)
        setattr(data, f'point_idx_b_{idx}', point_idx_b)
        setattr(data, f'centres_b_{idx}', centres_b * self.scaling_factor_global)

        setattr(data, f'pos_b_{idx}_batch', torch.zeros(len(pcd_b), dtype=torch.long))
        setattr(data, f'pos_b_{idx}_ptr', torch.tensor([0, len(pcd_b)], dtype=torch.long))
        setattr(data, f'features_b_{idx}_batch', torch.zeros(len(features_b), dtype=torch.long))
        setattr(data, f'features_b_{idx}_ptr', torch.tensor([0, len(features_b)], dtype=torch.long))
        setattr(data, f'centre_idx_b_{idx}_batch', torch.zeros(len(centre_idx_b), dtype=torch.long))
        setattr(data, f'centre_idx_b_{idx}_ptr', torch.tensor([0, len(centre_idx_b)], dtype=torch.long))
        setattr(data, f'point_idx_b_{idx}_batch', torch.zeros(len(point_idx_b), dtype=torch.long))
        setattr(data, f'point_idx_b_{idx}_ptr', torch.tensor([0, len(point_idx_b)], dtype=torch.long))
        setattr(data, f'centres_b_{idx}_batch', torch.zeros(len(centres_b), dtype=torch.long))
        setattr(data, f'centres_b_{idx}_ptr', torch.tensor([0, len(centres_b)], dtype=torch.long))

        setattr(data, f'pos_a_{idx}_batch', torch.zeros(len(pcd_a), dtype=torch.long))
        setattr(data, f'pos_a_{idx}_ptr', torch.tensor([0, len(pcd_a)], dtype=torch.long))
        setattr(data, f'features_a_{idx}_batch', torch.zeros(len(features_a), dtype=torch.long))
        setattr(data, f'features_a_{idx}_ptr', torch.tensor([0, len(features_a)], dtype=torch.long))
        setattr(data, f'centre_idx_a_{idx}_batch', torch.zeros(len(centre_idx_a), dtype=torch.long))
        setattr(data, f'centre_idx_a_{idx}_ptr', torch.tensor([0, len(centre_idx_a)], dtype=torch.long))
        setattr(data, f'point_idx_a_{idx}_batch', torch.zeros(len(point_idx_a), dtype=torch.long))
        setattr(data, f'point_idx_a_{idx}_ptr', torch.tensor([0, len(point_idx_a)], dtype=torch.long))
        setattr(data, f'centres_a_{idx}_batch', torch.zeros(len(centres_a), dtype=torch.long))
        setattr(data, f'centres_a_{idx}_ptr', torch.tensor([0, len(centres_a)], dtype=torch.long))

        return data

