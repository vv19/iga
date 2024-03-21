import numpy as np

################################################################################
# A default config for the model.
config = {
    ################################################################################
    # Data parameters.
    ################################################################################
    'shuffle_demo_order': True,
    'batch_size': 2,
    'batch_size_val': 1,
    'num_workers': 0,
    'use_scheduler': True,
    'lr_init': 1e-4,  # LR after warm up.
    'warm_up_lr_steps': 1000,  # Number of steps before warm up ends.
    'context_length': 5,  # Include the current observation.
    ############################################################################
    # Regularisation parameters.
    'reg_type': 'L2',  # 'L2', 'L1' or 'None'
    'reg_weight': 1e-5,
    'gradient_clip_val': 1.,
    ############################################################################
    # Model parameters.
    # Local encoder parameters.
    'pre_trained_local_encoder': True,
    'frozen_local_encoder': True,
    'number_local_centers': 8,
    'local_num_freq': 10,
    'local_nn_dims': [None] + [512] * 4,  # None means use the position encoder hidden dim.
    # Position encoder parameters.
    'num_freq': 6,
    'pcd_scaling': 100.,
    # Use Spectral norm.
    'spectral_norm': True,
    'init_weights': True,
    # Exponential moving average.
    'use_ema': True,
    # Gradient penalty.
    'add_gradient_penalty': True,
    'gradient_penalty_weight': 0.5,
    # GNN parameters.
    'num_layers': 4,
    'num_heads': 8,
    'hidden_dim': 512,
    'single_head_dim': 64,
    'dropout': 0.,
    # MLP parameters.
    'mlp_dims': [None, 512, 512, 1],  # None means use the hidden_dim.
    ############################################################################
    # Sampling mode.
    'langevin_every_n_steps': 50,  # -1 means no langevin sampling.
    'random_warm_up': 2000,  # Number of random warm up steps. No Langevin sampling during warm up.
    'local_perturb_every_n_steps': 3,
    ############################################################################
    # Uniform sampling parameters. Bounds from which to sample translations and rotations.
    'pos_ub_local': 0.1,
    'rot_ub_local': np.pi / 4,
    'pos_ub_global': 0.4,
    'rot_ub_global': np.pi,
    'pos_ub': 0.4,
    'rot_ub': np.pi,
    ############################################################################
    # Langevin parameters.
    # Training parameters.
    'num_langevin_steps_train': 20,  # Number of langevin steps during training.
    'num_negatives_train': 128,  # Number of negative samples to use for training.
    'step_size_init_train': 0.1,  # Initial step size for training.
    'noise_scale_init_train': 0.05,  # Noise scale for training.
    'step_size_decay_train': 0.1,  # Decay of step size for training.
    'noise_decay_train': 0.05,  # Decay of noise scale for training.
    # Validation parameters.
    'num_langevin_steps_val': 70,  # Number of langevin steps during validation.
    'num_negatives_val': 128,  # Number of samples to use for validation.
    'step_size_init_val': 0.1,  # Initial step size for validation.
    'noise_scale_init_val': 0.02,  # Noise scale for validation.
    'step_size_decay_val': 0.1,  # Decay of step size for validation.
    'noise_decay_val': 0.5,  # Decay of noise scale for validation.
    ############################################################################
    # Lightning parameters.
    'num_steps': 300000,
    'accelerator': 'gpu',
    'device': 'cuda',
    'devices': 1,
    'num_sanity_val_steps': 1,
    'check_val_every_n_epoch': None,
    'log_every_n_steps_val': 1000,
    'log_every_n_steps': 200,
    'save_itt': [50000, 100000, 150000, 200000, 250000],
    'save_every_steps': 1000,  # Save the last model every n steps (overwrites the previous model).
    ############################################################################
}
################################################################################
