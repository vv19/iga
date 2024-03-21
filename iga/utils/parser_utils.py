import argparse


def get_inference_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--real_data',
                        type=bool,
                        default=False,
                        help='If True, no random transformations will be applied to the live observations. '
                             'Use this for when for real experiments.')

    parser.add_argument('--data_dir',
                        type=str,
                        default='./demo_data',
                        help='Path to the directory, where the samples are stored.')

    parser.add_argument('--visualise_optimisation',
                        type=str,
                        default='pcd',
                        help='pcd - visualise the optimisation using point clouds; '
                             'graph - visualise the optimisation using graphs; '
                             'empty does not visualise.')
    parser.add_argument('--overlay_visualisation',
                        type=bool,
                        default=False,
                        help='If True, the visualisations will be overlayed on top of each other.')

    parser.add_argument('--model_dir',
                        type=str,
                        default='./checkpoints',
                        help='Path to the directory, where models are stored.')
    parser.add_argument('--num_negatives_trans',
                        type=int,
                        default=100,
                        help='Number of negative samples to use for the translation model.')
    parser.add_argument('--num_steps_trans',
                        type=int,
                        default=70,
                        help='Number of steps to use for the translation model.')
    parser.add_argument('--step_size_trans',
                        type=float,
                        default=0.1,
                        help='Step size to use for the translation model.')
    parser.add_argument('--step_size_decay_trans',
                        type=float,
                        default=0.05,
                        help='Step size decay to use for the translation model.')
    parser.add_argument('--noise_scale_init_trans',
                        type=float,
                        default=0.02,
                        help='Noise scale to use for the translation model.')
    parser.add_argument('--noise_decay_trans',
                        type=float,
                        default=0.2,
                        help='Noise decay to use for the translation model.')

    parser.add_argument('--num_negatives_rot',
                        type=int,
                        default=100,
                        help='Number of negative samples to use for the rotation model.')
    parser.add_argument('--num_steps_rot',
                        type=int,
                        default=70,
                        help='Number of steps to use for the rotation model.')
    parser.add_argument('--step_size_rot',
                        type=float,
                        default=0.1,
                        help='Step size to use for the rotation model.')
    parser.add_argument('--step_size_decay_rot',
                        type=float,
                        default=0.05,
                        help='Step size decay to use for the rotation model.')
    parser.add_argument('--noise_scale_init_rot',
                        type=float,
                        default=0.02,
                        help='Noise scale to use for the rotation model.')
    parser.add_argument('--noise_decay_rot',
                        type=float,
                        default=0.2,
                        help='Noise decay to use for the rotation model.')

    parser.add_argument('--no_x',
                        type=bool,
                        default=False,
                        help='If True, x position will not be optimised.')
    parser.add_argument('--no_y',
                        type=bool,
                        default=False,
                        help='If True, y position will not be optimised.')
    parser.add_argument('--no_z',
                        type=bool,
                        default=False,
                        help='If True, z position will not be optimised.')

    parser.add_argument('--no_rot_x',
                        type=bool,
                        default=False,
                        help='If True, rotation around x will not be optimised.')
    parser.add_argument('--no_rot_y',
                        type=bool,
                        default=False,
                        help='If True, rotation around y will not be optimised.')
    parser.add_argument('--no_rot_z',
                        type=bool,
                        default=False,
                        help='If True, rotation around z will not be optimised.')

    return parser
