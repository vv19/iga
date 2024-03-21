import time
from iga.configs.ebm_rot import config as ebm_rot_config
from iga.configs.ebm_trans import config as ebm_trans_config
from iga.models.ebm import *
import json
import os
import pytorch_lightning as pl
from iga.utils.alignment_dataset import AlignDataset
import torch
import argparse
from torch_geometric.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_model(model_path):
    config_path = '/'.join(model_path.split('/')[:-1]) + f'/config_{mode}.json'
    with open(config_path, 'r') as f:
        config_old = json.load(f)
    config_old['pre_trained_local_encoder'] = False
    config_old['spectral_norm'] = False
    config_old['local_nn_dims'][-1] = config_old['hidden_dim']

    model = EBM(config_old)
    model.config = config
    model.sampler.config = config

    checkpoint = torch.load(model_path)
    if model.use_ema:
        model.graph_encoder.load_state_dict(checkpoint['state_dict_ema_graph_encoder'], strict=False)
        model.energy_predictor.load_state_dict(checkpoint['state_dict_ema_energy_predictor'], strict=False)
    else:
        model.graph_encoder.load_state_dict(checkpoint['state_dict_graph_encoder'], strict=False)
        model.energy_predictor.load_state_dict(checkpoint['state_dict_energy_predictor'], strict=False)

    add_spec_norm(model.energy_predictor)
    add_spec_norm(model.graph_encoder)
    model.local_encoder.load_state_dict(checkpoint['state_dict_local_encoder'], strict=False)

    # Freeze the weights, otherwise memory usage increases over time.
    dfs_freeze(model.local_encoder)
    if model.use_ema:
        model.init_emas()

    model.train()
    ################################################################################################################
    return model


if __name__ == '__main__':
    ################################################################################
    # Set up config.
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='ebm')
    parser.add_argument('--record', type=bool, default=False)
    parser.add_argument('--reprocess', type=bool, default=False)
    parser.add_argument('--data_root', type=str, default='./demo_data')
    parser.add_argument('--data_root_val', type=str, default='./demo_data')
    parser.add_argument('--mode', type=str, default='trans')
    parser.add_argument('--num_samples', type=int, default=500000)
    parser.add_argument('--log_images', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default=None)

    reprocess = parser.parse_args().reprocess
    run_name = parser.parse_args().run_name
    record = parser.parse_args().record
    data_root = parser.parse_args().data_root
    data_root_val = parser.parse_args().data_root_val
    num_samples = parser.parse_args().num_samples
    log_images = parser.parse_args().log_images
    mode = parser.parse_args().mode
    model_path = parser.parse_args().model_path
    ################################################################################
    if mode == 'rot':
        config = ebm_rot_config
        config['dof_train'] = [False, False, False, True, True, True]
        config['dof_val'] = [False, False, False, True, True, True]
    else:
        config = ebm_trans_config
        config['dof_train'] = [True, True, True, False, False, False]
        config['dof_val'] = [True, True, True, False, False, False]

        config[f'step_size_init_train'] = config[f'step_size_init_train'] * config['pcd_scaling']
        config[f'noise_scale_init_train'] = config[f'noise_scale_init_train'] * config['pcd_scaling']
        config[f'step_size_init_val'] = config[f'step_size_init_val'] * config['pcd_scaling']
        config[f'noise_scale_init_val'] = config[f'noise_scale_init_val'] * config['pcd_scaling']

    run_name = f"{run_name}_{mode}"
    print(f'Run name: {run_name}')
    config['mode'] = mode
    config['record'] = record
    config['log_images'] = log_images
    config['save_dir'] = f"./runs/{run_name}/checkpoints/" if record else None
    config['local_encoder_path'] = f"./checkpoints/encoder/last.ckpt"
    ################################################################################
    # Set up datasets and dataloaders.
    torch.set_float32_matmul_precision('medium')

    dset_train = AlignDataset(root=data_root, num_samples=num_samples,
                              processed_dir='./data/processed_iga_train',
                              shuffle_demo_order=config['shuffle_demo_order'], reprocess=reprocess,
                              scaling_factor_global=config['pcd_scaling'],
                              number_local_centers=config['number_local_centers'], random_rot=False,
                              move_live_to_mean_demo=mode == 'rot')

    dset_val = AlignDataset(root=data_root_val, num_samples=10, processed_dir='./data/processed_data_iga_val',
                            shuffle_demo_order=False, reprocess=reprocess,
                            scaling_factor_global=config['pcd_scaling'],
                            number_local_centers=config['number_local_centers'], random_rot=False,
                            move_live_to_mean_demo=mode == 'rot')

    follow_batch = [f'pos_a_{i}' for i in range(config['context_length'])] \
                   + [f'centre_idx_a_{i}' for i in range(config['context_length'])] \
                   + [f'point_idx_a_{i}' for i in range(config['context_length'])] \
                   + [f'centres_a_{i}' for i in range(config['context_length'])] \
                   + [f'pos_b_{i}' for i in range(config['context_length'])] \
                   + [f'centre_idx_b_{i}' for i in range(config['context_length'])] \
                   + [f'point_idx_b_{i}' for i in range(config['context_length'])] \
                   + [f'centres_b_{i}' for i in range(config['context_length'])]

    train_dataloader = DataLoader(dset_train, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                                  follow_batch=follow_batch, pin_memory=True)
    val_dataloader = DataLoader(dset_val, batch_size=config['batch_size_val'], shuffle=False, num_workers=4,
                                follow_batch=follow_batch, pin_memory=True)
    ################################################################################
    print(f'Num training samples: {len(dset_train)}')
    print(f'Num validation samples: {len(dset_val)}')
    print(f'Context length: {config["context_length"]}')
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
        # We same models manually, so don't need to log them.
        callbacks = [
            LearningRateMonitor(logging_interval='step')
        ]

        # Save config to wandb if the file doesn't already exist.
        if not os.path.exists(f"./runs/{run_name}/checkpoints/config.json"):
            with open(f"./runs/{run_name}/checkpoints/config.json", 'w') as f:
                json.dump(config, f)
    else:
        logger = False
        callbacks = []
    ################################################################################
    if model_path is None:
        lightning_model = EBM(config)
    else:
        lightning_model = load_model(model_path)
    ################################################################################
    if config['record']:
        logger.experiment.watch(lightning_model, log='all', log_freq=100)
    ################################################################################
    # Set up trainer.
    trainer = pl.Trainer(
        limit_train_batches=1.,
        max_steps=config['num_steps'],
        enable_checkpointing=False,  # We save the models manually.
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
        val_dataloaders=val_dataloader,
    )
    ################################################################################
