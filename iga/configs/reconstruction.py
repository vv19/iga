################################################################################
# A default config for the model.
config = {
    ################################################################################
    'batch_size_train': 2,
    'batch_size_val': 2,
    'train_split': 0.9999,
    'num_workers': 8,
    'use_scheduler': True,
    'lr_init': 1e-4,  # LR after warm up.
    'warm_up_lr_steps': 100,  # Number of steps before warm up ends.
    ############################################################################
    ############################################################################
    # Regularisation parameters.
    'gradient_clip_val': 1.,
    ############################################################################
    # Model parameters.
    # Local encoder parameters.
    'number_local_centers': 8,
    'local_ratio': 0.01,
    'local_radius': 0.05,
    'local_num_freq': 10,
    # 'local_nn_dims': [None] + [512] * 4,   #  None means use the position encoder hidden dim.
    'local_nn_dims': [None] + [512] * 3 + [768],   #  None means use the position encoder hidden dim.
    # Lightning parameters.
    'num_steps': 500000,
    'accelerator': 'gpu',
    'devices': -1,
    'num_sanity_val_steps': 0,
    'check_val_every_n_epoch': None,
    'log_every_n_steps_val': 1000,
    'log_every_n_steps': 100,
    ############################################################################
}
################################################################################
