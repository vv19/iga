import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from iga.models.occupancy_net import Encoder
from iga.models.graph_encoder import GraphEncoder
from iga.train_occupancy_net import load_encoder
from torch_geometric.data import DataLoader
from torch_geometric.nn import MLP
from iga.optimization.sampler import Sampler, get_sampling_method, get_sampling_scale
from iga.utils.common_utils import *
from iga.utils.nn_utils import add_spec_norm, init_weights, dfs_freeze
from iga.utils.viz_utils import Visualiser
import wandb
import matplotlib.pyplot as plt
from diffusers.training_utils import EMAModel


class EBM(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.use_ema = config['use_ema']

        if self.config['pre_trained_local_encoder']:
            self.local_encoder, encoder_config = load_encoder(self.config['local_encoder_path'], device=self.device)
            self.config['local_nn_dims'] = encoder_config['local_nn_dims']
            if self.config['frozen_local_encoder']:
                dfs_freeze(self.local_encoder)
        else:
            self.local_encoder = Encoder(local_nn_dims=config['local_nn_dims'])
        self.graph_encoder = GraphEncoder(config)

        mlp_dims = config['mlp_dims']
        mlp_dims[0] = config['hidden_dim']

        self.energy_predictor = MLP(mlp_dims,
                                    dropout=config['dropout'],
                                    norm=None,
                                    act=nn.GELU(approximate='tanh'), plain_last=True)

        if self.config['init_weights']:
            self.energy_predictor.apply(init_weights)

        # Pv plotter for debugging
        self.visualiser = Visualiser(off_screen=config['record'], save_dir=config['save_dir'], config=config)

        # Create a sampler
        self.sampler = Sampler(forward_model=self.predict_energy, perturb_func=self.graph_encoder.perturb_negatives,
                               config=config, graph_mean_func=self.graph_encoder.get_graph_means)

        # Adding spectral norm to all layers
        if config['spectral_norm']:
            add_spec_norm(self.energy_predictor)

        self.initialised = False
        self.loss_buffer = torch.zeros(config['log_every_n_steps'])
        self.energy_max_buffer = torch.zeros(config['log_every_n_steps'])
        self.energy_min_buffer = torch.zeros(config['log_every_n_steps'])

        self.best_val_loss = 1e10
        self.val_losses = []

        if self.use_ema:
            self.init_emas()

    def init_emas(self):
        self.graph_encoder = self.graph_encoder.to(self.config['device'])
        self.energy_predictor = self.energy_predictor.to(self.config['device'])
        self.ema_graph_encoder = EMAModel(parameters=self.graph_encoder.parameters(), power=0.75)
        self.ema_energy_predictor = EMAModel(parameters=self.energy_predictor.parameters(), power=0.75)


    def predict_energy(self, graph, return_idx=False):
        results = self.graph_encoder(graph, return_idx=return_idx)
        energy = self.energy_predictor(results[0].view(-1, results[0].shape[-1])).view(results[0].shape[0], -1, 1)

        return energy.squeeze(-1), *results[1:]

    def initialise(self, batch=None):
        print('Initializing model...')
        self.initialised = True
        self.graph_encoder.initialise(self.device)
        self.local_encoder.initialise(self.config['context_length'])
        self.sampler.device = self.device

        self.loss_buffer = torch.zeros(self.config['log_every_n_steps'])
        self.energy_max_buffer = torch.zeros(self.config['log_every_n_steps'])
        self.energy_min_buffer = torch.zeros(self.config['log_every_n_steps'])
        self.best_val_loss = 1e10
        self.val_losses = []

        if batch is not None:
            # Creating a single batch for the visualisation. Using first batch in the validation set.
            follow_batch = [f'pos_a_{i}' for i in range(self.config['context_length'])] \
                           + [f'centre_idx_a_{i}' for i in range(self.config['context_length'])] \
                           + [f'point_idx_a_{i}' for i in range(self.config['context_length'])] \
                           + [f'centres_a_{i}' for i in range(self.config['context_length'])] \
                           + [f'pos_b_{i}' for i in range(self.config['context_length'])] \
                           + [f'centre_idx_b_{i}' for i in range(self.config['context_length'])] \
                           + [f'point_idx_b_{i}' for i in range(self.config['context_length'])] \
                           + [f'centres_b_{i}' for i in range(self.config['context_length'])]
            loader_temp = DataLoader(batch, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                                     follow_batch=follow_batch)
            self.single_batch = next(iter(loader_temp))

    def get_energies(self, batch, T_noise=None, num_negatives=3, batch_size=1, sampling_method='der_free',
                     mode='train', return_edge_grads=False):
        # Encoding point clouds into local nodes.
        ############################################################################################################
        local_node_info = self.local_encoder.encode_sample(batch)  # batch.clone() ?
        ############################################################################################################
        graph = self.graph_encoder.create_graph(local_node_info, num_negatives=num_negatives, batch_size=batch_size)
        if T_noise is not None:
            graph = self.graph_encoder.perturb_negatives(graph, T_noise, num_negatives)
        else:
            graph = self.sampler.sample(batch=graph, batch_size=batch_size, config=self.config,
                                        sampling_mehtod=sampling_method, mode=mode)
        if return_edge_grads:
            return self.predict_energy(graph), graph.edge_grad
        return self.predict_energy(graph)

    def inference(self, batch, batch_size=1, visualise=True, progress_bar=False, ret_optim_results=False):
        # Encoding point clouds into local nodes.
        local_node_info = self.local_encoder.encode_sample(batch)
        # Creating a graph and creating negative samples.
        graph = self.graph_encoder.create_graph(local_node_info, num_negatives=self.config['num_negatives_val'],
                                                batch_size=batch_size)
        ############################################################################################################
        self.sampler.config['pos_ub'], self.sampler.config['rot_ub'] =\
            self.config['pos_ub_global'] * self.config['pcd_scaling'], self.config['rot_ub_global']
        torch.set_grad_enabled(True)
        results = self.sampler.sample_langevin(graph, batch_size=batch_size, mode='val',
                                               return_trajectory=True, perturb_init=True,
                                               progress_bar=progress_bar)
        torch.set_grad_enabled(False)
        if visualise:
            assert batch_size == 1
            # Selecting the first batch to visualise.
            traj = results[3][:, 0, ...][:, results[2][0], ...]
            graph_means = results[4][:, 0, ...][:, results[2][0], ...]
            self.visualiser.visualise_trajectory(batch, traj, graph_means, context_length=self.config['context_length'])
        if ret_optim_results:
            return results
        return se3_loss(results[1][:, results[2], ...].view(-1, 4, 4))

    def training_step(self, batch, batch_idx):
        if not self.initialised:
            self.initialise(batch)
            return
        current_batch_size = torch.max(batch.pos_a_0_batch) + 1

        sampling_method = get_sampling_method(self.config, self.global_step)
        self.sampler.config['pos_ub'], self.sampler.config['rot_ub'] = get_sampling_scale(self.config, self.global_step)

        # Predicting energy and calculating loss
        (energy, labels), edge_grads = self.get_energies(batch,
                                                         num_negatives=self.config['num_negatives_train'],
                                                         batch_size=current_batch_size,
                                                         sampling_method=sampling_method,
                                                         mode='train',
                                                         return_edge_grads=True)
        loss = self.loss_fn(-energy, labels)

        # Adding energy regularization. Stay close to zero.
        if self.config['reg_type'].lower() == 'l2':
            reg_loss = F.mse_loss(energy, torch.zeros_like(energy), reduction='mean')
            loss += self.config['reg_weight'] * reg_loss
        elif self.config['reg_type'].lower() == 'l1':
            reg_loss = F.l1_loss(energy, torch.zeros_like(energy), reduction='mean')
            loss += self.config['reg_weight'] * reg_loss

        # Adding gradient penalty.
        if self.config['add_gradient_penalty']:
            grads = torch.autograd.grad(loss, edge_grads, create_graph=True, retain_graph=True)[0]
            grad_norms = torch.norm(grads, p=2)
            loss += self.config['gradient_penalty_weight'] * grad_norms

        # Logging
        self.loss_buffer[self.global_step % self.config['log_every_n_steps']] = loss.item()
        self.energy_max_buffer[self.global_step % self.config['log_every_n_steps']] = torch.max(energy).item()
        self.energy_min_buffer[self.global_step % self.config['log_every_n_steps']] = torch.min(energy).item()
        self.log('train_loss', self.loss_buffer.mean(), on_step=True, on_epoch=False, prog_bar=True)
        self.log('energy_max', self.energy_max_buffer.mean(), on_step=True, on_epoch=False)
        self.log('energy_min', self.energy_min_buffer.mean(), on_step=True, on_epoch=False)
        return loss

    @torch.inference_mode(False)
    def validation_step(self, batch, batch_idx, visualise_traj=False, progress_bar=False):
        if not self.initialised:
            self.initialise(batch)
            return

        if self.use_ema:
            # Copy the current model to the EMA model for validation. Just a hack to reuse the same functions.
            graph_encoder_state_dict = self.graph_encoder.state_dict().copy()
            energy_predictor_state_dict = self.energy_predictor.state_dict().copy()
            self.ema_graph_encoder.copy_to(self.graph_encoder.parameters())
            self.ema_energy_predictor.copy_to(self.energy_predictor.parameters())

        current_batch_size = torch.max(batch.pos_a_0_batch) + 1
        loss_t, loss_r = self.inference(batch, batch_size=current_batch_size,
                                        visualise=visualise_traj, progress_bar=progress_bar)

        if self.config['mode'] == 'rot':
            val_loss = loss_r
        else:
            val_loss = loss_t
            
        self.val_losses.append(val_loss)

        if batch_idx == 0 and self.config['log_images']:
            self.visualiser.visualise_energy(self.get_energies,
                                             self.single_batch,
                                             resolution_xy=(20, 20),
                                             size_xy=(1.5 * self.config['pcd_scaling'],
                                                      1.5 * self.config['pcd_scaling']),
                                             device=self.device,
                                             override_meshes=True,
                                             global_step=self.global_step,
                                             )
            if self.config['record']:
                path_to_img = f"{self.visualiser.save_dir}/log_image_{self.global_step}.png"
                im = plt.imread(path_to_img)
                self.logger.experiment.log({"Energy Landscape": [wandb.Image(im, caption="Energy")]})

        if self.use_ema:
            self.graph_encoder.load_state_dict(graph_encoder_state_dict, strict=False)
            self.energy_predictor.load_state_dict(energy_predictor_state_dict, strict=False)

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.parameters(), lr=self.config['lr_init'])

        if self.config['use_scheduler']:
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimiser,
                                                           start_factor=1e-38,
                                                           total_iters=self.config['warm_up_lr_steps'])
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                                    T_max=self.config['num_steps'])

            scheduler = torch.optim.lr_scheduler.SequentialLR(optimiser,
                                                              schedulers=[scheduler1, scheduler2],
                                                              milestones=[self.config['warm_up_lr_steps']])
            return [optimiser], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        else:
            return optimiser

    def on_validation_epoch_end(self):
        if len(self.val_losses):
            mean_val_loss = sum(self.val_losses) / len(self.val_losses)
            self.log('val_loss', mean_val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.val_losses = []
            if self.config['record']:
                if mean_val_loss < self.best_val_loss:
                    # Save the best model.
                    self.save_current_model(f'{self.config["save_dir"]}/best_model.pt')

    def on_train_batch_end(self, *args, **kwargs):
        if self.global_step in self.config['save_itt'] and self.config['record']:
            self.save_current_model(f'{self.config["save_dir"]}/{self.global_step}_step.pt')

        if self.global_step % self.config['save_every_steps'] == 0 and self.config['record']:
            self.save_current_model(f'{self.config["save_dir"]}/latest_model.pt')

    def on_before_zero_grad(self, *args, **kwargs):
        if self.use_ema:
            self.ema_graph_encoder.step(self.graph_encoder.parameters())
            self.ema_energy_predictor.step(self.energy_predictor.parameters())

    def save_current_model(self, save_path):
        checkpoint = self.trainer._checkpoint_connector.dump_checkpoint(False)
        checkpoint['state_dict_graph_encoder'] = self.graph_encoder.state_dict().copy()
        checkpoint['state_dict_local_encoder'] = self.local_encoder.state_dict()
        checkpoint['state_dict_energy_predictor'] = self.energy_predictor.state_dict().copy()

        if self.use_ema:
            # A hack to save the EMA models.
            self.ema_graph_encoder.copy_to(self.graph_encoder.parameters())
            self.ema_energy_predictor.copy_to(self.energy_predictor.parameters())

            checkpoint['state_dict_ema_graph_encoder'] = self.graph_encoder.state_dict().copy()
            checkpoint['state_dict_ema_energy_predictor'] = self.energy_predictor.state_dict().copy()

            # Load the original models back.
            self.graph_encoder.load_state_dict(checkpoint['state_dict_graph_encoder'], strict=False)
            self.energy_predictor.load_state_dict(checkpoint['state_dict_energy_predictor'], strict=False)

        torch.save(checkpoint, save_path)
