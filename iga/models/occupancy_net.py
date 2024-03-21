from iga.models.vn_networks import *
from iga.utils.nn_utils import PositionalEncoder


class Encoder(nn.Module):
    def __init__(self, local_nn_dims):
        super(Encoder, self).__init__()
        self.vn_encoder = VNEncoder(local_nn_dims)
        self.vn_global = VNEncoder([local_nn_dims[-1], local_nn_dims[-1], local_nn_dims[-1]])
        self.conv = VNPointNetConv(self.vn_encoder, global_nn=self.vn_global, aggr='mean')

    def initialise(self, context_length):
        # Buffers for context samples.
        self.context_length = context_length
        self.contex_a_x = [None] * (self.context_length - 1)
        self.contex_a_pos = [None] * (self.context_length - 1)
        self.contex_a_batch = [None] * (self.context_length - 1)
        self.contex_b_x = [None] * (self.context_length - 1)
        self.contex_b_pos = [None] * (self.context_length - 1)
        self.contex_b_batch = [None] * (self.context_length - 1)

    def encode_sample(self, data):
        a_p_x, a_p_pos, a_p_batch = self.encode(None, data.pos_a_0, data.features_a_0, data.centres_a_0,
                                                data.centre_idx_a_0, data.centres_a_0_ptr, data.centre_idx_a_0_batch,
                                                data.centres_a_0_batch, data.point_idx_a_0,
                                                data.pos_a_0_ptr, data.point_idx_a_0_batch)

        b_p_x, b_p_pos, b_p_batch = self.encode(None, data.pos_b_0, data.features_b_0, data.centres_b_0,
                                                data.centre_idx_b_0, data.centres_b_0_ptr, data.centre_idx_b_0_batch,
                                                data.centres_b_0_batch, data.point_idx_b_0,
                                                data.pos_b_0_ptr, data.point_idx_b_0_batch)

        for i in range(1, self.context_length):
            self.contex_a_x[i - 1], self.contex_a_pos[i - 1], self.contex_a_batch[i - 1] =\
                self.encode(None,
                            getattr(data, f'pos_a_{i}'),
                            getattr(data, f'features_a_{i}'),
                            getattr(data, f'centres_a_{i}'),
                            getattr(data, f'centre_idx_a_{i}'),
                            getattr(data, f'centres_a_{i}_ptr'),
                            getattr(data, f'centre_idx_a_{i}_batch'),
                            getattr(data, f'centres_a_{i}_batch'),
                            getattr(data, f'point_idx_a_{i}'),
                            getattr(data, f'pos_a_{i}_ptr'),
                            getattr(data, f'point_idx_a_{i}_batch'))

            self.contex_b_x[i - 1], self.contex_b_pos[i - 1], self.contex_b_batch[i - 1] =\
                self.encode(None,
                            getattr(data, f'pos_b_{i}'),
                            getattr(data, f'features_b_{i}'),
                            getattr(data, f'centres_b_{i}'),
                            getattr(data, f'centre_idx_b_{i}'),
                            getattr(data, f'centres_b_{i}_ptr'),
                            getattr(data, f'centre_idx_b_{i}_batch'),
                            getattr(data, f'centres_b_{i}_batch'),
                            getattr(data, f'point_idx_b_{i}'),
                            getattr(data, f'pos_b_{i}_ptr'),
                            getattr(data, f'point_idx_b_{i}_batch'))

        # Caching context batches
        a_c_batch = torch.cat(self.contex_a_batch, dim=0)
        b_c_batch = torch.cat(self.contex_b_batch, dim=0)

        # Altering batch, so that different context samples are in different batches.
        for i in range(2, self.context_length):
            self.contex_a_batch[i - 1] = self.contex_a_batch[i - 1] + torch.max(self.contex_a_batch[i - 2]) + 1
            self.contex_b_batch[i - 1] = self.contex_b_batch[i - 1] + torch.max(self.contex_b_batch[i - 2]) + 1

        # Concatenate all the context samples.
        a_c_x = torch.cat(self.contex_a_x, dim=0)
        a_c_pos = torch.cat(self.contex_a_pos, dim=0)
        a_c_batch_demo = torch.cat(self.contex_a_batch, dim=0)
        b_c_x = torch.cat(self.contex_b_x, dim=0)
        b_c_pos = torch.cat(self.contex_b_pos, dim=0)
        b_c_batch_demo = torch.cat(self.contex_b_batch, dim=0)

        embeddings = {
            'a_p_x': a_p_x,
            'a_p_pos': a_p_pos,
            'a_p_batch': a_p_batch,
            'b_p_x': b_p_x,
            'b_p_pos': b_p_pos,
            'b_p_batch': b_p_batch,
            'a_c_x': a_c_x,
            'a_c_pos': a_c_pos,
            'a_c_batch': a_c_batch,
            'a_c_batch_demo': a_c_batch_demo,
            'b_c_x': b_c_x,
            'b_c_pos': b_c_pos,
            'b_c_batch': b_c_batch,
            'b_c_batch_demo': b_c_batch_demo,
        }
        return embeddings

    def encode(self, x, points, local_features, centres, centre_idx, centres_ptr, centre_idx_batch, centres_batch,
               point_idx, points_ptr, point_idx_batch):

        # First, adjust the batch indices. We need this because of our fancy data loader.
        centre_idx = centre_idx + centres_ptr[centre_idx_batch]
        point_idx = point_idx + points_ptr[point_idx_batch]
        edge_index = torch.stack([point_idx, centre_idx], dim=0)

        x_dst = None

        feat = local_features.view(points.shape[0], -1)
        x = self.conv((feat, x_dst), (points, centres), edge_index)

        # Cast centres to the same dtype as x. This is needed for AMP.
        centres = centres.to(x.dtype)
        pos, batch = centres, centres_batch
        return x, pos, batch


class Decoder(nn.Module):
    def __init__(self, nn_dims):
        super().__init__()

        self.linear_layers = nn.ModuleList([nn.Linear(nn_dims[i], nn_dims[i + 1]) for i in range(len(nn_dims) - 1)])
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x):
        for i, layer in enumerate(self.linear_layers):
            if i == 0 or i == len(self.linear_layers) - 1:
                x = layer(x)
            else:
                x = x + layer(x)
            if i != len(self.linear_layers) - 1:
                x = self.act(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, local_nn_dims, local_num_freq=10):
        super().__init__()

        self.local_position_encoder = PositionalEncoder(3, local_num_freq, log_space=False)
        self.encoder = Encoder(local_nn_dims)

        local_decoder_dims = local_nn_dims[::-1]
        local_decoder_dims[-1] = 1
        local_decoder_dims[0] = local_nn_dims[-1] + self.local_position_encoder.d_output

        self.local_decoder = Decoder(local_decoder_dims)

    def forward(self, data):
        # Encode the input point cloud into a set of features, and keep track of the positions.
        x, pos, batch_pos = self.encoder.encode(data.x, data.points, data.local_features, data.centres, data.centre_idx,
                                                data.centres_ptr, data.centre_idx_batch, data.centres_batch,
                                                data.point_idx, data.points_ptr, data.point_idx_batch)
        # Mean pooling.
        x = mean_pool(x, dim=-1)
        # First, adjust the batch indices. We need this because of our fancy data loader.
        data.queries_idx = data.queries_idx + data.queries_ptr[data.queries_idx_batch]
        data.queries_centre_idx = data.queries_centre_idx + data.centres_ptr[data.queries_centre_idx_batch]

        # For each feature, find which query points are in the same neighborhood.
        local_queries = self.local_position_encoder(data.queries[data.queries_idx] - pos[data.queries_centre_idx])
        query_x = torch.cat([x[data.queries_centre_idx], local_queries], dim=1)
        occupancy = self.local_decoder(query_x).squeeze()
        target_occupancy = data.occupancy[data.queries_idx].squeeze()
        return occupancy, target_occupancy
