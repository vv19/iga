import torch
from torch_geometric.utils.nested import from_nested_tensor, to_nested_tensor
from torch_geometric.utils import to_dense_batch, remove_self_loops
import torch.nn as nn
from torch_geometric.nn import to_hetero, TransformerConv, MLP
from iga.utils.nn_utils import PositionalEncoder, add_spec_norm, init_weights
from torch_geometric.data import HeteroData


class GraphEncoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.node_types = ['local', 'e']
        self.edge_types_rel = [
            ('local', 'rel', 'local'),
            ('local', 'rel_across', 'local'),
        ]
        self.edge_types_cond = [
            ('local', 'live', 'e'),
            ('local', 'cond', 'local'),
        ]
        self.edge_types = self.edge_types_rel + self.edge_types_cond

        self.num_freq = config['num_freq']
        self.position_encoder = PositionalEncoder(3, self.num_freq, log_space=True, scale=config['pcd_scaling'])
        self.edge_dim = self.position_encoder.d_output

        self.energy_embedder = nn.Embedding(1, config['local_nn_dims'][-1])
        self.edge_embedder = nn.Embedding(len(self.edge_types_cond), self.edge_dim)

        self.hetero_transformer_encoder = GraphTransformer(config['local_nn_dims'][-1],
                                                           config['single_head_dim'] * config['num_heads'],
                                                           config['hidden_dim'],
                                                           heads=config['num_heads'],
                                                           edge_dim=self.edge_dim,
                                                           num_layers=config['num_layers'],
                                                           metadata=(self.node_types, self.edge_types),
                                                           dropout=config['dropout'],
                                                           )

        # Adding spectral norm to all layers
        if config['spectral_norm']:
            add_spec_norm(self)

        # Initialise weights
        if config['init_weights']:
            self.apply(init_weights)

    def initialise(self, device):
        self.device = device
        self.energy_embedding = torch.tensor(0, dtype=torch.long, device=self.device).unsqueeze(0)
        self.edge_embeddings = {edge: torch.tensor(k, dtype=torch.long, device=self.device).unsqueeze(0) for k, edge in
                                enumerate(self.edge_types_cond)}

        self.cached_zero = torch.zeros(1, device=self.device)
        self.cached_one = torch.ones(1, device=self.device)

        self.initialised = True

    def create_graph(self, local_node_info, num_negatives, batch_size):
        '''
        Creates a graph from the local node info.
        :param local_node_info: a dictionary containing the local node info. It needs to have the following keys:
            'a_p_x' - local features of the positive example (object a)
            'a_p_pos' - local positions of the positive example (object a)
            'a_p_batch' - batch index of the positive example (object a)
            'b_p_x' - local features of the positive example (object b)
            'b_p_pos' - local positions of the positive example (object b)
            'b_p_batch' - batch index of the positive example (object b)

            'a_c_x' - local features of the context (conditional / demo) examples (object a)
            'a_c_pos' - local positions of the context (conditional / demo) examples (object a)
            'a_c_batch' - batch index of the context examples (object a).
            'a_c_batch_demo' - batch index of the context examples (object a). Each conditional example is in a separate batch.
            'b_c_x' - local features of the context (conditional / demo) examples (object b)
            'b_c_pos' - local positions of the context (conditional / demo) examples (object b)
            'b_c_batch' - batch index of the context examples (object b).
            'b_c_batch_demo' - batch index of the context examples (object b). Each conditional example is in a separate batch.
        :param batch_size: Current batch size.
        :param num_negatives: Number of negative examples to create.
        :return: A HeteroData object containing the graph. Nodes are created based on self.node_types.
                 Edges are created based on self.edge_types.
        '''

        ############################################################################################################
        # Creating copies of positive samples, that will serve as negatives, when perturbed.
        a_x, a_pos, a_batch, a_batch_demo = self.create_negatives_batched(local_node_info['a_p_x'],
                                                                          local_node_info['a_p_pos'],
                                                                          local_node_info['a_p_batch'],
                                                                          num_negatives,
                                                                          batch_size)
        b_x, b_pos, b_batch, b_batch_demo = self.create_negatives_batched(local_node_info['b_p_x'],
                                                                          local_node_info['b_p_pos'],
                                                                          local_node_info['b_p_batch'],
                                                                          num_negatives,
                                                                          batch_size)
        ############################################################################################################
        # Concatenating generated samples with the context samples
        # The order is: context 'a', positive 'a', negative 'a', context 'b', positive 'b', negative 'b'
        local_x_all = torch.cat([local_node_info['a_c_x'], a_x, local_node_info['b_c_x'], b_x], dim=0)
        local_pos_all = torch.cat([local_node_info['a_c_pos'], a_pos, local_node_info['b_c_pos'], b_pos], dim=0)
        # Concatenating the batch indices. _separate -- each subgraph is in a separate batch.
        # _grouped -- each 'a' and 'b' subgraph is in the same batch.
        # context_batch -- keeps track from which batch the context samples come from. negative values are for padding.
        a_batch_separate = torch.cat(
            [local_node_info['a_c_batch_demo'], a_batch_demo + torch.max(local_node_info['a_c_batch_demo']).item() + 1],
            dim=0)
        b_batch_separate = torch.cat(
            [local_node_info['b_c_batch_demo'], b_batch_demo + torch.max(local_node_info['b_c_batch_demo']).item() + 1],
            dim=0)
        local_batch_separate = torch.cat([a_batch_separate, b_batch_separate + torch.max(a_batch_separate).item() + 1],
                                         dim=0)
        local_batch_grouped = torch.cat([a_batch_separate, b_batch_separate], dim=0)
        context_batch = torch.cat(
            [local_node_info['a_c_batch'], (a_batch + 1) * -1, local_node_info['b_c_batch'], (b_batch + 1) * -1], dim=0)
        ############################################################################################################
        # Creating one big graph as a HeteroData object and assigning different nodes and edges to it.
        hetero_data = HeteroData()
        hetero_data['local'].x = local_x_all
        hetero_data['local'].pos = local_pos_all
        hetero_data['local'].batch = local_batch_separate  # Maybe don't need to cache this?
        hetero_data['local'].batch_grouped = local_batch_grouped  # Maybe don't need to cache this?

        hetero_data['local'].true_batch = torch.cat([local_node_info['a_c_batch'],
                                                     a_batch,
                                                     local_node_info['b_c_batch'],
                                                     b_batch], dim=0)
        # Node e, has learnable embeddings, and is used to make predictions about each subgraph.
        hetero_data['e'].x = self.energy_embedder(self.energy_embedding).repeat(torch.max(a_batch_demo).item() + 1, 1)

        # Keeping track of labels for each subgraph. first one will be positive, followed by negatives.
        labels = torch.zeros((1 + num_negatives) * batch_size, device=self.device)
        labels[:batch_size] = 1.0

        hetero_data['e'].label = labels
        # Negative labels are for finding which nodes are from positive, negative or conditional sample.
        # It has to match the length of number of unique batches. We only move object 'b', so everything that is not
        # from object 'b' is padded with -1.
        hetero_data['e'].negative_labels = -1 * torch.ones(torch.max(local_batch_separate).item() + 1,
                                                           device=self.device)
        hetero_data['e'].negative_labels[-len(labels):] = labels
        # Creating variables that keeps track of from which batch and from which demo does the 'e' node came from.
        # e_batch_demo is also used to gather back the predicted energies corresponding to same batches.
        e_batch = torch.arange(len(labels), device=self.device) + torch.max(
            local_node_info['b_c_batch_demo']).item() + 1

        e_batch_demo = torch.arange(batch_size, device=self.device).repeat(1, 1 + num_negatives).squeeze()
        hetero_data['e'].e_batch_demo = e_batch_demo
        ############################################################################################################
        # FOR VIS ONLY:
        hetero_data['e'].viz_labels = torch.cat([-1 * torch.ones(batch_size * (self.config['context_length'] - 1),
                                                                 device=self.device), labels])
        hetero_data['local'].a_b_mask = torch.zeros_like(local_batch_separate, device=self.device)
        hetero_data['local'].a_b_mask[:len(a_batch_separate)] = 1
        ############################################################################################################
        # Creating edges between nodes.
        # Dense edge index for the local - local edges.
        edge_idx = torch.cartesian_prod(
            torch.arange(hetero_data['local'].num_nodes, dtype=torch.int64, device=self.device),
            torch.arange(hetero_data['local'].num_nodes, dtype=torch.int64, device=self.device)).contiguous().t()
        if ('local', 'rel', 'local') in self.edge_types:
            # Creating the same edge index for the same edges.
            hetero_data['local', 'rel', 'local'].edge_index = \
                (edge_idx[:, local_batch_separate[edge_idx[0, :]] == local_batch_separate[edge_idx[1, :]]]).clone()

            # Adding edge attributes.
            hetero_data['local', 'rel', 'local'].edge_attr = \
                (hetero_data['local'].pos[hetero_data['local', 'rel', 'local'].edge_index[1, :]] - \
                 hetero_data['local'].pos[hetero_data['local', 'rel', 'local'].edge_index[0, :]]).clone()
            hetero_data['local', 'rel', 'local'].edge_attr = \
                self.position_encoder(hetero_data['local', 'rel', 'local'].edge_attr)
        if ('local', 'rel_across', 'local') in self.edge_types:
            mask = torch.logical_and(local_batch_separate[edge_idx[0, :]] != local_batch_separate[edge_idx[1, :]],
                                     local_batch_grouped[edge_idx[0, :]] == local_batch_grouped[edge_idx[1, :]])

            mask = torch.logical_and(mask, hetero_data['local'].a_b_mask[edge_idx[0, :]] == 1)
            hetero_data['local', 'rel_across', 'local'].edge_index = edge_idx[:, mask].clone()

            # Adding edge attributes.
            hetero_data['local', 'rel_across', 'local'].edge_attr = \
                (hetero_data['local'].pos[hetero_data['local', 'rel_across', 'local'].edge_index[1, :]] - \
                 hetero_data['local'].pos[hetero_data['local', 'rel_across', 'local'].edge_index[0, :]]).clone()
            hetero_data['local', 'rel_across', 'local'].edge_attr = \
                self.position_encoder(hetero_data['local', 'rel_across', 'local'].edge_attr)

        if ('local', 'cond', 'local') in self.edge_types:

            mask = torch.logical_and(context_batch[edge_idx[0, :]] > -1,
                                     context_batch[edge_idx[0, :]] == hetero_data['local'].true_batch[edge_idx[1, :]])

            mask = torch.logical_and(mask, context_batch[edge_idx[1, :]] < 0)

            mask = torch.logical_and(mask, hetero_data['local'].a_b_mask[edge_idx[0, :]] ==
                                     hetero_data['local'].a_b_mask[edge_idx[1, :]])

            hetero_data['local', 'cond', 'local'].edge_index = edge_idx[:, mask].clone()

        if ('local', 'live', 'e') in self.edge_types:
            edge_idx = torch.cartesian_prod(
                torch.arange(hetero_data['local'].num_nodes, dtype=torch.int64, device=self.device),
                torch.arange(hetero_data['e'].num_nodes, dtype=torch.int64, device=self.device)).contiguous().t()

            mask = local_batch_grouped[edge_idx[0, :]] == e_batch[edge_idx[1, :]]

            hetero_data['local', 'live', 'e'].edge_index = edge_idx[:, mask].clone()
            # edge_idx[:, local_batch_grouped[edge_idx[0, :]] == e_batch[edge_idx[1, :]]]
            if ('local', 'cond', 'e') in self.edge_types:
                mask = torch.logical_and(context_batch[edge_idx[0, :]] == e_batch_demo[edge_idx[1, :]],
                                         hetero_data['local'].a_b_mask[edge_idx[0, :]] == 0)
                hetero_data['local', 'cond', 'e'].edge_index = edge_idx[:, mask].clone()

        # Adding edge features (learnable embeddings) for conditional edges.
        for edge_info in self.edge_types_cond:
            hetero_data[edge_info].edge_attr = self.edge_embedder(self.edge_embeddings[edge_info]).repeat(
                hetero_data[edge_info].edge_index.shape[1], 1).clone()

        # Remove self loops.
        for edge_info in self.edge_types:
            hetero_data[edge_info].edge_index, hetero_data[edge_info].edge_attr = remove_self_loops(
                hetero_data[edge_info].edge_index, hetero_data[edge_info].edge_attr)

        return hetero_data

    def create_negatives_batched(self, x, pos, batch, num_negatives, curr_batch_size):
        '''
        Creates a batch of negative examples by repeating the positive example.
        batch is training batch index, batch_demo is the index of the conditional example (each subgraph will have a different index).
        '''
        batch_demo = (batch.repeat(num_negatives + 1, 1) + (
                torch.arange(num_negatives + 1, device=self.device) * curr_batch_size).unsqueeze(1)).flatten()
        batch = batch.repeat(num_negatives + 1, 1).flatten()
        x = x.repeat(num_negatives + 1, 1, 1)
        pos = pos.repeat(num_negatives + 1, 1)

        return x.clone(), pos.clone(), batch.clone(), batch_demo.clone()

    def get_graph_means(self, graph):
        '''
        Returns the mean positions of the 'b-live' nodes in the graph.
        '''
        b_pos_mask = graph['e'].negative_labels[graph['local'].batch] != -1
        batch_mask = graph['local'].batch[b_pos_mask]
        batch_mask -= sum(graph['e'].negative_labels == -1)
        b_pos_means, _ = to_dense_batch(graph['local'].pos[b_pos_mask], batch_mask, fill_value=torch.nan)
        b_pos_means = b_pos_means[..., :3].nanmean(dim=1)
        return b_pos_means.squeeze()

    def rotate_features(self, graph, T_noise):
        # Convert to nested tensor.
        b_pos_mask = graph['e'].negative_labels[graph['local'].batch] != -1

        batch_mask = graph['local'].batch[b_pos_mask]
        batch_mask -= sum(graph['e'].negative_labels == -1)

        R = T_noise[:, :3, :3]
        b_x = graph['local'].x[b_pos_mask]
        b_x = to_nested_tensor(b_x, batch_mask)

        og_size = b_x[0].shape[0], b_x[0].shape[1], b_x[0].shape[2]
        b_x = b_x.reshape(torch.max(batch_mask).item() + 1, og_size[0] * og_size[1], og_size[2])

        R = torch.nested.as_nested_tensor([r for r in R])
        b_x = (R @ b_x.transpose(1, 2).contiguous()).transpose(1, 2)
        b_x = b_x.reshape(torch.max(batch_mask).item() + 1, og_size[0], og_size[1], og_size[2])
        b_x = from_nested_tensor(b_x)
        graph['local'].x[b_pos_mask] = b_x
        return graph

    def perturb_negatives(self, graph, T_noise, num_negatives):
        # Convert to nested tensor.
        b_pos_mask = graph['e'].negative_labels[graph['local'].batch] != -1
        b_pos_hom = torch.cat([graph['local'].pos[b_pos_mask],
                               torch.ones(graph['local'].pos[b_pos_mask].shape[0], 1, device=self.device)], dim=1)

        batch_mask = graph['local'].batch[b_pos_mask]
        batch_mask -= sum(graph['e'].negative_labels == -1)
        #######################################################################################
        # Rotating the features. We can do this because features are SO(3) equivariant.
        R = T_noise[:, :3, :3]
        b_x = graph['local'].x[b_pos_mask]
        b_x = to_nested_tensor(b_x, batch_mask)

        og_size = b_x[0].shape[0], b_x[0].shape[1], b_x[0].shape[2]
        b_x = b_x.reshape(torch.max(batch_mask).item() + 1, og_size[0] * og_size[1], og_size[2])

        R = torch.nested.as_nested_tensor([r for r in R])
        b_x = (R @ b_x.transpose(1, 2).contiguous()).transpose(1, 2)
        b_x = b_x.reshape(torch.max(batch_mask).item() + 1, og_size[0], og_size[1], og_size[2])
        b_x = from_nested_tensor(b_x)
        graph['local'].x[b_pos_mask] = b_x
        #######################################################################################

        #######################################################################################
        # Rotating the positions.
        nested_b = to_nested_tensor(b_pos_hom, batch_mask)
        #######################################################################################
        # Rotating around center of mass.
        #######################################################################################
        # Means of the b_pos per batch.
        b_pos_means, _ = to_dense_batch(b_pos_hom, batch_mask, fill_value=torch.nan)
        b_pos_means = b_pos_means[..., :3].nanmean(dim=1)
        T_mean = torch.eye(4, device=self.device).repeat(b_pos_means.shape[0], 1, 1)
        T_mean[:, :3, 3] = b_pos_means
        T_mean = torch.nested.as_nested_tensor([T for T in T_mean]).detach()

        T_mean_inv = torch.eye(4, device=self.device).repeat(b_pos_means.shape[0], 1, 1)
        T_mean_inv[:, :3, 3] = -b_pos_means
        T_mean_inv = torch.nested.as_nested_tensor([T for T in T_mean_inv]).detach()
        #####################################################################################
        T_noise = torch.nested.as_nested_tensor([T for T in T_noise])
        nested_b = (T_mean_inv @ nested_b.transpose(1, 2).contiguous()).transpose(1, 2)
        nested_b = (T_noise @ nested_b.transpose(1, 2).contiguous()).transpose(1, 2)
        nested_b = (T_mean @ nested_b.transpose(1, 2).contiguous()).transpose(1, 2)
        #####################################################################################
        graph['local'].pos[b_pos_mask] = from_nested_tensor(nested_b)[:, :3]

        return self.update_rel_edge_attr(graph)

    def update_rel_edge_attr(self, graph):
        graph['local', 'rel_across', 'local'].edge_attr = \
            graph['local'].pos[graph['local', 'rel_across', 'local'].edge_index[1, :]] - \
            graph['local'].pos[graph['local', 'rel_across', 'local'].edge_index[0, :]]

        # Caching this in case we need to add gradient penalty for too sharpy changes in energy based on the relative positions.
        graph.edge_grad = graph['local', 'rel_across', 'local'].edge_attr
        if not graph.edge_grad.requires_grad:
            graph.edge_grad.requires_grad = True

        graph['local', 'rel_across', 'local'].edge_attr = self.position_encoder(
            graph['local', 'rel_across', 'local'].edge_attr)

        graph['local', 'rel', 'local'].edge_attr = \
            (graph['local'].pos[graph['local', 'rel', 'local'].edge_index[1, :]] - \
             graph['local'].pos[graph['local', 'rel', 'local'].edge_index[0, :]]).clone()
        graph['local', 'rel', 'local'].edge_attr = \
            self.position_encoder(graph['local', 'rel', 'local'].edge_attr)
        return graph

    def forward(self, graph, return_idx=False):
        x_old = graph['local'].x
        graph['local'].x = graph['local'].x.mean(dim=-1, keepdim=False)
        x_dict_new = self.hetero_transformer_encoder(graph.x_dict,
                                                     graph.edge_index_dict,
                                                     graph.edge_attr_dict)

        labels_sorted_idx = torch.argsort(graph['e'].e_batch_demo)
        batch_sorted = graph['e'].e_batch_demo[labels_sorted_idx]

        labels, _ = to_dense_batch(graph['e'].label[labels_sorted_idx], batch_sorted)
        e_logits, _ = to_dense_batch(x_dict_new['e'][labels_sorted_idx], batch_sorted)
        graph['local'].x = x_old
        if return_idx:
            return e_logits, labels, labels_sorted_idx, batch_sorted

        return e_logits, labels


class GraphTransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_heads=1, edge_dim=None, dropout=0.3):
        super().__init__()

        self.att = TransformerConv(in_channels, hidden_channels // n_heads, edge_dim=edge_dim, heads=n_heads,
                                   concat=True, add_self_loops=False, dropout=dropout, bias=False)

        self.lin = MLP([hidden_channels, hidden_channels, hidden_channels],
                       act=nn.GELU(approximate='tanh'),
                       plain_last=False,
                       norm=None,
                       )

    def forward(self, x, edge_index, edge_attr):
        x = self.att(x, edge_index, edge_attr)
        x = x + self.lin(x)
        return x


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, edge_dim=3, num_layers=2, metadata=None,
                 dropout=0.3):
        super().__init__()

        in_channels_list = [in_channels] + [hidden_channels] * (num_layers - 1)
        self.transformer_blocks = nn.ModuleList(
            [to_hetero(
                GraphTransformerBlock(in_channels_list[i], hidden_channels, heads, edge_dim=edge_dim, dropout=dropout),
                metadata=metadata,
                aggr='sum') for i in range(num_layers)
            ])

        self.lin = to_hetero(torch.nn.Sequential(
            MLP([hidden_channels, hidden_channels, out_channels],
                act=nn.GELU(approximate='tanh'),
                dropout=dropout,
                plain_last=False,
                norm=None  # Norms didn't play well with the spectral norm and extremely small batch sizes.
                ))
            , metadata=metadata, aggr='sum')

    def forward(self, x, edge_index, edge_attr):
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, edge_index, edge_attr)

        x = self.lin(x)

        return x
