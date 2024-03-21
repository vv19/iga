from torch_geometric.data import Dataset, Data
import torch
import shutil
import os
import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import knn
from iga.models.vn_layers import LocalFeatureEncoder
from iga.utils.reconstruction_dataset import create_clustered_data_sample


class AlignDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, reprocess=True, num_samples=1000, local_neighbours=20,
                 processed_dir='./processed', shuffle_demo_order=False, number_local_centers=8,
                 random_rot=False, scaling_factor_local=100, scaling_factor_global=10,
                 move_live_to_mean_demo=False):

        self.move_live_to_mean_demo = move_live_to_mean_demo
        self.scaling_factor_local = scaling_factor_local
        self.scaling_factor_global = scaling_factor_global
        self.local_feature_encoder = LocalFeatureEncoder(aggr='mean')
        self.local_neighbours = local_neighbours

        self.number_local_centers = number_local_centers
        self.reprocess = reprocess
        self.processed_ = processed_dir

        self.raw_names = [file_name for file_name in os.listdir(root) if 'sample' in file_name][:num_samples]
        self.raw_names.sort()
        self.processed_ = f'{self.processed_}'

        if self.reprocess and os.path.exists(self.processed_):
            shutil.rmtree(self.processed_)

        if self.reprocess:
            self.dataset_length = 0
        else:
            self.dataset_length = len(os.listdir(self.processed_)) - 2
        super(AlignDataset, self).__init__(root, transform, pre_transform)
        self.dataset_length = min(len(os.listdir(self.processed_dir)) - 2, num_samples)  # -2 for transforms

        self.shuffle_demo_order = shuffle_demo_order
        self.random_rot = random_rot
        self.context_length = 5  # Default value. Will be updated in process().

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return self.raw_names

    @property
    def processed_file_names(self):
        if self.reprocess:
            return [' ']
        return [f'data_{i}.pt' for i in range(self.len())]

    @property
    def processed_dir(self) -> str:
        return self.processed_

    def process(self):

        # Caching how many conditional demos we have per sample
        self.context_length = len(pickle.load(open(self.raw_paths[0], 'rb'))['pcds_a'])

        i = 0
        for raw_path in tqdm(self.raw_paths, leave=False):
            # Read data from `raw_path`.
            sample = pickle.load(open(raw_path, 'rb'))

            data = Data()
            for j in range(len(sample['pcds_a'])):
                pcd_a = torch.tensor(sample['pcds_a'][j], dtype=torch.float)
                pcd_b = torch.tensor(sample['pcds_b'][j], dtype=torch.float)
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
                setattr(data, f'pos_a_{j}', pcd_a * self.scaling_factor_global)
                setattr(data, f'features_a_{j}', features_a)
                setattr(data, f'centre_idx_a_{j}', centre_idx_a)
                setattr(data, f'point_idx_a_{j}', point_idx_a)
                setattr(data, f'centres_a_{j}', centres_a * self.scaling_factor_global)

                setattr(data, f'pos_b_{j}', pcd_b * self.scaling_factor_global)
                setattr(data, f'features_b_{j}', features_b)
                setattr(data, f'centre_idx_b_{j}', centre_idx_b)
                setattr(data, f'point_idx_b_{j}', point_idx_b)
                setattr(data, f'centres_b_{j}', centres_b * self.scaling_factor_global)
                ##################################################################################
                # Test for equivariance
                # R = torch.tensor(Rot.random().as_matrix(), dtype=torch.float)
                # features = self.local_feature_encoder(data.pos_a_0, id_k_neighbor).view(-1, 3, 3)
                # features_t = self.local_feature_encoder(torch.mm(data.pos_a_0, R.t()), id_k_neighbor).view(-1, 3, 3)
                # features_ot = torch.mm(features.view(-1, 3), R.t()).view(-1, 3, 3)
                # printarr(features, features_t, features_ot, features_t - features_ot)
                # raise
                ##################################################################################

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return self.dataset_length

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        # Swap the 0th and another random demo. 0th is always the live, so we want to swap it with a random demo.
        if self.shuffle_demo_order:
            # Swap a and b with 50% probability.
            if np.random.uniform() > 0.5:
                # Swap a and b.
                for j in range(self.context_length):
                    exec(f"data.pos_a_{j}, data.pos_b_{j} = data.pos_b_{j}, data.pos_a_{j}")
                    exec(f"data.features_a_{j}, data.features_b_{j} = data.features_b_{j}, data.features_a_{j}")
                    exec(f"data.centre_idx_a_{j}, data.centre_idx_b_{j} = data.centre_idx_b_{j}, data.centre_idx_a_{j}")
                    exec(f"data.point_idx_a_{j}, data.point_idx_b_{j} = data.point_idx_b_{j}, data.point_idx_a_{j}")
                    exec(f"data.centres_a_{j}, data.centres_b_{j} = data.centres_b_{j}, data.centres_a_{j}")

            swap_idx = np.random.randint(self.context_length)
            exec(f"data.pos_a_0, data.pos_a_{swap_idx} = data.pos_a_{swap_idx}, data.pos_a_0")
            exec(f"data.features_a_0, data.features_a_{swap_idx} = data.features_a_{swap_idx}, data.features_a_0")
            exec(
                f"data.centre_idx_a_0, data.centre_idx_a_{swap_idx} = data.centre_idx_a_{swap_idx}, data.centre_idx_a_0")
            exec(f"data.point_idx_a_0, data.point_idx_a_{swap_idx} = data.point_idx_a_{swap_idx}, data.point_idx_a_0")
            exec(f"data.centres_a_0, data.centres_a_{swap_idx} = data.centres_a_{swap_idx}, data.centres_a_0")

            exec(f"data.pos_b_0, data.pos_b_{swap_idx} = data.pos_b_{swap_idx}, data.pos_b_0")
            exec(f"data.features_b_0, data.features_b_{swap_idx} = data.features_b_{swap_idx}, data.features_b_0")
            exec(
                f"data.centre_idx_b_0, data.centre_idx_b_{swap_idx} = data.centre_idx_b_{swap_idx}, data.centre_idx_b_0")
            exec(f"data.point_idx_b_0, data.point_idx_b_{swap_idx} = data.point_idx_b_{swap_idx}, data.point_idx_b_0")
            exec(f"data.centres_b_0, data.centres_b_{swap_idx} = data.centres_b_{swap_idx}, data.centres_b_0")

        if self.move_live_to_mean_demo:
            move_idx = np.random.randint(self.context_length - 1)
            centering = torch.mean(data.pos_b_0, dim=0) - torch.mean(getattr(data, f'pos_b_{move_idx + 1}'), dim=0)
            data.pos_b_0 -= centering
            data.centres_b_0 -= centering

        return data
