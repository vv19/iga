import lightning as L
import torch
import torch.nn as nn
import json
from iga.models.occupancy_net import AutoEncoder
from iga.utils.reconstruction_dataset import ReconstructDataset
import numpy as np
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import os
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from iga.configs.reconstruction import config
import argparse


class LightningAE(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.))
        self.loss_buffer = torch.zeros(config['log_every_n_steps'])
        self.config = config

    def training_step(self, batch, batch_idx):
        y, targets = self.model.forward(batch)
        if torch.any(torch.isnan(y)):
            print('NaN in output.')
        if torch.any(torch.isnan(targets)):
            print('NaN in targets.')
        loss = self.loss_fn(y, targets)
        self.loss_buffer[self.global_step % self.config['log_every_n_steps']] = loss.item()
        self.log('train_loss', self.loss_buffer.mean(), prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=len(targets))
        return loss

    def validation_step(self, batch, batch_idx):
        y, targets = self.model.forward(batch)
        loss = self.loss_fn(y.squeeze(), targets.squeeze())

        # Compute the accuracy.
        y = torch.sigmoid(y)
        y = (y > 0.5).float()
        acc = (y == targets).float().mean()

        self.log('val_loss', loss.item(), prog_bar=True, batch_size=len(targets))
        self.log('val_acc', acc.item(), prog_bar=True, batch_size=len(targets))

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


def load_encoder(model_path, device):
    experiment_path = '/'.join(model_path.split('/')[:-1])
    with open(f"{experiment_path}/config.json", 'r') as f:
        config = json.load(f)
    model = AutoEncoder(local_nn_dims=config['local_nn_dims'], local_num_freq=config['local_num_freq'])
    lightning_model = LightningAE.load_from_checkpoint(model_path,
                                                       model=model,
                                                       config=config,
                                                       map_location=device,
                                                       strict=True, )
    return lightning_model.model.encoder, config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='rec_768')
    parser.add_argument('--record', type=bool, default=True)
    parser.add_argument('--data_root', type=str,
                        default='/home/vv19/projects/python/expecto_patronum/patronus/data_gen/data/rec')
    parser.add_argument('--num_samples', type=int, default=100000)

    args = parser.parse_args()
    run_name = args.run_name
    record = args.record
    data_root = args.data_root
    config['num_samples'] = args.num_samples
    ####################################################################################################################
    model = AutoEncoder(local_nn_dims=config['local_nn_dims'], local_num_freq=config['local_num_freq'])
    lightning_model = LightningAE(model, config)
    ####################################################################################################################
    # lightning_model = LightningAE(lightning_model.model, config)
    ####################################################################################################################
    dset = ReconstructDataset(f"{data_root}", num_samples=config['num_samples'],
                              reprocess=False, number_local_centers=config['number_local_centers'], )

    train_size = int(np.floor(len(dset) * config['train_split']))
    val_size = len(dset) - train_size
    dataset_train, dataset_val = random_split(dset, [train_size, val_size])
    follow_batch = ['points', 'queries', 'centres', 'centre_idx', 'point_idx', 'queries_idx', 'queries_centre_idx']

    train_dataloader = DataLoader(dataset_train, batch_size=config['batch_size_train'],
                                  follow_batch=follow_batch, shuffle=True, num_workers=config['num_workers'],
                                  pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=config['batch_size_val'],
                                follow_batch=follow_batch, shuffle=False, num_workers=config['num_workers'],
                                pin_memory=True)
    print(f'Num training samples: {len(dataset_train)}')
    print(f'Num validation samples: {len(dataset_val)}')
    ####################################################################################################################
    ################################################################################
    # Set up Logger and Callbacks if recording.
    if record:
        if f'./runs/{run_name}' is not None:
            os.makedirs(f'./runs/{run_name}/checkpoints', exist_ok=True)
        logger = WandbLogger(project='IGA',
                             name=run_name,
                             save_dir=f'./runs/{run_name}',
                             log_model=False,
                             config=config)

        callbacks = [
            ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                verbose=False,
                save_on_train_epoch_end=True,
                dirpath=f'./runs/{run_name}/checkpoints',
                filename="{step}",
                every_n_train_steps=100,
                save_last=True,
                save_top_k=3
            ),
            LearningRateMonitor(logging_interval='step')
        ]
        config['save_dir'] = f"./runs/{run_name}/checkpoints/"

        # Save config to wandb if the file doesn't already exist.
        if not os.path.exists(f"./runs/{run_name}/checkpoints/config.json"):
            with open(f"./runs/{run_name}/checkpoints/config.json", 'w') as f:
                json.dump(config, f)
    else:
        logger = False
        callbacks = []
    ################################################################################
    ################################################################################
    # Set up trainer.
    trainer = L.Trainer(
        limit_train_batches=1.,
        max_steps=config['num_steps'],
        enable_checkpointing=record,
        default_root_dir=f'./runs/{run_name}',
        accelerator=config['accelerator'],
        devices=config['devices'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
        val_check_interval=config['log_every_n_steps_val'],
        logger=logger,
        log_every_n_steps=config['log_every_n_steps'],
        callbacks=callbacks,
        enable_progress_bar=True,
        num_sanity_val_steps=config['num_sanity_val_steps'],
        gradient_clip_val=config['gradient_clip_val'],
    )
    ################################################################################
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    ################################################################################
