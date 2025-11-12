
import torch
from torch import nn
from e3nn.o3 import Irreps
from torch_geometric.data import Data
from mint.model.embedding.equilibrium_embedder import MLPWithBN

class MINTMLP(nn.Module):
    r"""
    A simple wrapper for the MLPwithBN from mint.model.embedding.equilibrium_embedder.

    :param in_dim:
        Input feature dimension.
    :type in_dim: int
    :param hidden_dims:
        Hidden layer sizes in order.
    :type hidden_dims: tuple[int, ...]
    :param out_dim:
        Output feature dimension.
    :type out_dim: int
    :param activation:
        Nonlinearity name parsed by :func:`mint.utils.parse_activation`.
    :type activation: str
    :param use_input_bn:
        If ``True``, apply BatchNorm to the inputs.
    :type use_input_bn: bool
    :param affine:
        Whether BN layers learn affine scale/shift and linears drop bias when ``True``.
    :type affine: bool
    :param track_running_stats:
        Whether BN tracks running mean/var.
    :type track_running_stats: bool
    """
    def __init__(self, in_dim, hidden_dims=(128, 128), out_dim=1, activation='relu',
                 use_input_bn=True, affine=True, track_running_stats=True):

        super().__init__()
        self.half_dim = int(out_dim/2)
        self.net = MLPWithBN(
                in_dim=in_dim,
                hidden_dims=hidden_dims,
                out_dim=out_dim,
                activation=activation,
                use_input_bn=use_input_bn,
                affine=affine,
                track_running_stats=track_running_stats,
            )

    def forward(self, batch: Data) -> Data:
        r"""
        Apply the MLP to inputs. Cats batch['x'] and batch['f']

        :param batch:
            PyG data object with 'x' and 'f'.
        :type batch: torch_geometric.data.Data
        :return:
            Output batch with keys 'b' and 'eta' added.
        :rtype: torch.Tensor
        """
        x= batch['x']
        f = batch['f']

        # get shapes right
        B = batch.num_graphs
        Dx = x.size(-1)
        Df = f.size(-1)
        N = x.size(0) // B
        x = x.view(B, N, Dx).flatten(1, 2)
        f = f.view(B, N, Df).flatten(1, 2)

        # create initial hidden state
        state = torch.cat([x,f],dim=-1) # B,N*(Dx+Df)

        # apply net
        state = self.net(state)

        # get results
        batch['b'] = state[:,:self.half_dim].reshape(B, N, Dx).reshape(B*N, Dx)
        batch['eta'] = state[:,self.half_dim:].reshape(B, N, Dx).reshape(B*N, Dx)

        batch['b_irrep'] = Irreps("1o"),
        batch['eta_irrep'] = Irreps("1o")
        return batch