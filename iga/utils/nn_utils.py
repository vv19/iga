import torch
import torch.nn as nn
import torch_geometric


def add_spec_norm(model):
    for layer in get_children(model):
        if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch_geometric.nn.dense.linear.Linear):
            layer = torch.nn.utils.spectral_norm(layer, n_power_iterations=1)


def remove_spec_norm(model):
    for layer in get_children(model):
        if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch_geometric.nn.dense.linear.Linear):
            try:
                layer = torch.nn.utils.remove_spectral_norm(layer)
            except:
                pass


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch_geometric.nn.dense.linear.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def dfs_freeze(model):
    for child in get_children(model):
        for param in child.parameters():
            param.requires_grad = False


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(
            self,
            d_input: int,
            n_freqs: int,
            log_space: bool = False,
            add_original_x: bool = True,
            scale: float = 1.0,
    ):
        super().__init__()
        self.scale = scale
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space

        if add_original_x:
            self.embed_fns = [lambda x: x]
            self.d_output = d_input * (1 + 2 * self.n_freqs)
        else:
            self.embed_fns = []
            self.d_output = d_input * (2 * self.n_freqs)

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x / self.scale * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x / self.scale * freq))

    def forward(
            self,
            x
    ) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """

        non_learnable_freqs = torch.cat([f(x) for f in self.embed_fns], dim=-1)
        return non_learnable_freqs
