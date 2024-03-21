from iga.models.vn_layers import *
from typing import Optional, Callable, Union
from torch_geometric.typing import OptTensor, PairOptTensor, Adj, PairTensor


def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


class VNPointNetConv(MessagePassing):

    def __init__(self, local_nn: Optional[Callable] = None, global_nn: Optional[Callable] = None, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:

        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)
        out = out.view(1, out.shape[0], -1, 3)

        if self.global_nn is not None:
            out = out.permute(0, 2, 3, 1).contiguous()
            out = self.global_nn(out).permute(0, 3, 1, 2).contiguous()

        return out.squeeze(0)

    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                pos_j: Tensor) -> Tensor:

        msg = x_j.view(-1, 3, 3)

        #  [B, N_feat, 3, N_samples, ...]
        msg = self.local_nn(msg.unsqueeze(0).permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous().view(pos_i.shape[0],
                                                                                                        -1)
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn})')


class VNEncoder(nn.Module):
    def __init__(self, nn_dims):
        super(VNEncoder, self).__init__()
        self.d_out = nn_dims[-1]
        if nn_dims[0] is None:
            nn_dims[0] = 3
        self.nn_dims = nn_dims
        self.net = nn.ModuleList([
            VNLinearAndLeakyReLU(nn_dims[i], nn_dims[i + 1], share_nonlinearity=False, use_batchnorm='none',
                                 negative_slope=0.1) for i in range(len(nn_dims) - 1)])

    def forward(self, x):
        for i, layer in enumerate(self.net):
            if self.nn_dims[i] != self.nn_dims[i + 1]:
                x = layer(x)
            else:
                x = x + layer(x)
        return x
