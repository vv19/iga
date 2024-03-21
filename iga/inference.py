from iga.models.inference_model import IGA
import argparse
from iga.utils.common_utils import transform_pcd
from iga.utils.parser_utils import get_inference_parser
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import torch


def vis_initial_poses(pcd_target, pcd_grasped):
    iga.visualiser.clear()
    iga.visualiser.add_pcd_to_plotter(pcd_target * iga.scaling_factor_global,
                                      name='live_pcd_target', color='red',
                                      radius=0.003 * iga.scaling_factor_global)
    iga.visualiser.add_pcd_to_plotter(pcd_grasped * iga.scaling_factor_global,
                                      name='live_pcd_grasped', color='green',
                                      radius=0.003 * iga.scaling_factor_global)
    iga.visualiser.plotter.add_text('Initial Object Poses, press q to continue.',
                                    font_size=20, position='upper_left', name='text', color='k')
    iga.visualiser.plotter.show(auto_close=False)
    iga.visualiser.plotter.add_text('Optimising.',
                                    font_size=20, position='upper_left', name='text', color='k')
    iga.visualiser.clear()


def vis_final_poses(pcd_target, pcd_grasped, T_e_e_new):
    iga.visualiser.clear()
    iga.visualiser.add_pcd_to_plotter(pcd_target * iga.scaling_factor_global,
                                      name='live_pcd_target', color='red',
                                      radius=0.003 * iga.scaling_factor_global)

    iga.visualiser.add_pcd_to_plotter(pcd_grasped * iga.scaling_factor_global,
                                      name='live_pcd_grasped_old', color='yellow',
                                      radius=0.003 * iga.scaling_factor_global, opacity=0.5)
    iga.visualiser.add_pcd_to_plotter(
        transform_pcd(pcd_grasped, T_e_e_new) * iga.scaling_factor_global,
        name='live_pcd_grasped', color='green',
        radius=0.003 * iga.scaling_factor_global)
    iga.visualiser.plotter.add_text('Object Poses After Optimisation. Yellow - Initial Pose. press q to continue.',
                                    font_size=20, position='upper_left', name='text', color='k')
    iga.visualiser.plotter.show(auto_close=False)
    iga.visualiser.clear()


if __name__ == '__main__':
    ####################################################################################################################
    parser = get_inference_parser()
    data_dir = parser.parse_args().data_dir
    model_dir = parser.parse_args().model_dir
    visualise_optimisation = parser.parse_args().visualise_optimisation
    overlay_visualisation = parser.parse_args().overlay_visualisation
    num_neg_trans = parser.parse_args().num_negatives_trans
    num_steps_trans = parser.parse_args().num_steps_trans
    step_size_trans = parser.parse_args().step_size_trans
    step_size_decay_trans = parser.parse_args().step_size_decay_trans
    noise_scale_init_trans = parser.parse_args().noise_scale_init_trans
    noise_decay_trans = parser.parse_args().noise_decay_trans
    num_neg_rot = parser.parse_args().num_negatives_rot
    num_steps_rot = parser.parse_args().num_steps_rot
    step_size_rot = parser.parse_args().step_size_rot
    step_size_decay_rot = parser.parse_args().step_size_decay_rot
    noise_scale_init_rot = parser.parse_args().noise_scale_init_rot
    noise_decay_rot = parser.parse_args().noise_decay_rot

    no_x = parser.parse_args().no_x
    no_y = parser.parse_args().no_y
    no_z = parser.parse_args().no_z
    no_rot_x = parser.parse_args().no_rot_x
    no_rot_y = parser.parse_args().no_rot_y
    no_rot_z = parser.parse_args().no_rot_z

    real_data = parser.parse_args().real_data
    ####################################################################################################################
    iga = IGA(
        trans_model_path=f'{model_dir}/ebm_trans.pt',
        rot_model_path=f'{model_dir}/ebm_rot.pt',
        num_negatives_trans=num_neg_trans,
        num_steps_trans=num_steps_trans,
        step_size_trans=step_size_trans,
        step_size_decay_trans=step_size_decay_trans,
        noise_scale_init_trans=noise_scale_init_trans,
        noise_decay_trans=noise_decay_trans,
        num_negatives_rot=num_neg_rot,
        num_steps_rot=num_steps_rot,
        step_size_rot=step_size_rot,
        step_size_decay_rot=step_size_decay_rot,
        noise_scale_init_rot=noise_scale_init_rot,
        noise_decay_rot=noise_decay_rot,

        dof_rot=(False, False, False, not no_rot_x, not no_rot_y, not no_rot_z),
        dof_trans=(not no_x, not no_y, not no_z, False, False, False),

        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    ####################################################################################################################
    num_samples = len([file for file in os.listdir(data_dir) if 'sample' in file and file.endswith('.pkl')])
    gripper_pcd = pickle.load(open('./assets/franka_gripper_pcd.pkl', 'rb'))
    for k in range(num_samples):
        raw_sample = pickle.load(open(f'{data_dir}/sample_{k}.pkl', 'rb'))
        sample = {
            'demo_pcds': {
                'pcds_grasped': [raw_sample['pcds_a'][i] if raw_sample['pcds_a'][i] is not None else gripper_pcd for i
                                 in range(1, len(raw_sample['pcds_a']))],
                'pcds_target': raw_sample['pcds_b'][1:]},
            'live_pcds': {
                'pcd_grasped': raw_sample['pcds_a'][0] if raw_sample['pcds_a'][0] is not None else gripper_pcd,
                'pcd_target': raw_sample['pcds_b'][0]}
        }
        if not real_data:
            # Applying a random transformation to the live observations to simulate the real world.
            T_rand = np.eye(4)
            T_rand[:3, :3] = Rot.random().as_matrix()
            T_rand[:3, 3] = np.random.uniform(-0.3, 0.3, 3)
            sample['live_pcds']['pcd_target'] = transform_pcd(sample['live_pcds']['pcd_target'], T_rand)
        ################################################################################################################
        vis_initial_poses(sample['live_pcds']['pcd_target'], sample['live_pcds']['pcd_grasped'])
        ################################################################################################################
        T_e_e_new = iga.get_transform(sample['demo_pcds'],
                                      sample['live_pcds'],
                                      visualise=visualise_optimisation != '',
                                      overlay=overlay_visualisation,
                                      vis_graph=visualise_optimisation == 'graph')
        ################################################################################################################
        vis_final_poses(sample['live_pcds']['pcd_target'], sample['live_pcds']['pcd_grasped'], T_e_e_new)
    ####################################################################################################################
