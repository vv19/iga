from torch_geometric.data import Dataset, Data
import torch
import shutil
import os
import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import fps, knn, nearest
from iga.models.vn_layers import LocalFeatureEncoder


class ReconstructDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, reprocess=True, num_samples=1000,
                 processed_dir='processed', number_local_centers=8, local_neighbours=20, scaling_factor=100):

        self.scaling_factor = scaling_factor
        self.local_feature_encoder = LocalFeatureEncoder(aggr='mean')
        self.local_neighbours = local_neighbours
        self.number_local_centers = number_local_centers
        self.reprocess = reprocess
        self.processed_ = processed_dir

        self.raw_names = [file_name for file_name in os.listdir(root) if 'sample' in file_name][:num_samples]
        self.processed_ = f'{self.processed_}'

        if self.reprocess and os.path.exists(os.path.join(root, self.processed_)):
            shutil.rmtree(os.path.join(root, self.processed_))

        if self.reprocess:
            self.dataset_length = 0
        else:
            self.dataset_length = len(os.listdir(os.path.join(root, self.processed_))) - 2
        super(ReconstructDataset, self).__init__(root, transform, pre_transform)
        self.dataset_length = len(os.listdir(self.processed_dir)) - 2  # -2 for transforms

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
        return os.path.join(self.root, self.processed_)

    def process(self):

        i = 0
        for raw_path in tqdm(self.raw_paths, leave=False):
            # Read data from `raw_path`.
            sample = pickle.load(open(raw_path, 'rb'))

            data = Data()
            pcd = sample['points'] * self.scaling_factor
            queries = sample['queries'] * self.scaling_factor
            dist = sample['dists']

            # dists are signed, but we want to make them unsigned for the occupancy prediction task.
            dist = np.abs(dist)
            # Construct query points for occupancy net.
            occupancy = np.zeros((queries.shape[0], 1))
            occupancy[dist < 0.002] = 1.
            queries = np.concatenate([queries, pcd], axis=0)
            occupancy = np.concatenate([occupancy, np.ones((pcd.shape[0], 1))], axis=0)
            data.points = torch.tensor(pcd, dtype=torch.float)
            data.queries = torch.tensor(queries, dtype=torch.float)
            data.occupancy = torch.tensor(occupancy, dtype=torch.float).squeeze()

            centre_idx, point_idx, centres, queries_idx, queries_centre_idx = \
                create_clustered_data_sample(data.points,
                                             num_clusters=self.number_local_centers,
                                             queries=data.queries)

            data.centre_idx = centre_idx
            data.point_idx = point_idx
            data.centres = centres
            data.queries_idx = queries_idx
            data.queries_centre_idx = queries_centre_idx

            # Local features
            id_k_neighbor = knn(data.points, data.points, k=self.local_neighbours)
            features = self.local_feature_encoder(data.points - data.centres[data.centre_idx],
                                                  id_k_neighbor).view(-1, 3, 3)
            data.local_features = features

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return self.dataset_length

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


def create_clustered_data_sample(pcd, queries=None, num_clusters=8):
    ######################################################################################
    idx = fps(pcd, ratio=num_clusters / pcd.shape[0])
    cluster_centers = pcd[idx]
    cluster_ids = nearest(pcd, cluster_centers)
    # Constructing the edge index
    centre_idx = torch.tensor(cluster_ids, dtype=torch.long)  # row
    point_idx = torch.arange(0, pcd.shape[0], dtype=torch.long)  # col
    ######################################################################################
    centres = torch.tensor(cluster_centers, dtype=torch.float)
    ######################################################################################
    if queries is not None:
        # For each query point, we want to find the closest cluster center.
        # We will use the NN function from PyTorch Geometric.
        query_cluster_ids = nearest(queries, cluster_centers)
        queries_idx = torch.arange(0, queries.shape[0])
        queries_centre_idx = torch.tensor(query_cluster_ids, dtype=torch.long)
        return centre_idx, point_idx, centres, queries_idx, queries_centre_idx
    return centre_idx, point_idx, centres
    ######################################################################################
