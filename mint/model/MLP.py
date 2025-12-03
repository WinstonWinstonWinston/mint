
import torch
from torch import nn
from e3nn.o3 import Irreps
from torch_geometric.data import Data
from mint.model.embedding.equilibrium_embedder import MLP
from torch_geometric.utils import to_dense_batch

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
        self.out_dim = out_dim
        self.net = MLP(
                in_dim=in_dim,
                hidden_dims=hidden_dims,
                out_dim=2*out_dim,
                activation=activation
            )

    def forward(self, batch: Data) -> Data:
        r"""
        Apply the MLP to inputs. Cats batch['x_t'] and batch['f']

        :param batch:
            PyG data object with 'x_t' and 'f'.
        :type batch: torch_geometric.data.Data
        :return:
            Output batch with keys 'b' and 'eta' added.
        :rtype: torch.Tensor
        """
        x = batch['x_t']
        f = batch['f']
        B = max(batch['batch'])+1

        x, mask_x = to_dense_batch(x, batch.batch)
        f, mask_f = to_dense_batch(f, batch.batch)

        x = x.flatten(start_dim=1,end_dim=-1)
        f = f.flatten(start_dim=1,end_dim=-1)

        # create initial hidden state
        f = torch.cat([x,f],dim=-1)

        # apply net
        f = self.net(f)

        b = f[:,:self.out_dim].reshape(B,22,3)
        eta = f[:,self.out_dim:].reshape(B,22,3)

        # get results
        batch['b'] = b[mask_x]
        batch['eta'] = eta[mask_x]

        batch['b_irrep'] = Irreps("1x1e")
        batch['eta_irrep'] = Irreps("1x1e")

        batch['feature_keys'].add('b')
        batch['feature_keys'].add('eta')

        return batch