import torch
from torch import nn
from e3nn.o3 import Irreps
from torch_geometric.data import Data
from mint.model.embedding.time import TimeEmbed
from mint.utils import parse_activation

class EquilibriumEmbedder(nn.Module):
    r"""
    Node feature embedder that concatenates (i) atom-type embeddings,
    (ii) an interpolant time embedding, and (optionally) (iii) force-field
    scalars passed through an MLP.

    .. math::

        \mathbf{f}_i
        \;=\;
        \big[\, \mathbf{e}^{(\text{atom})}_{a_i}
        \;\Vert\;
        \mathbf{e}^{(t)}_{t}
        \;\Vert\;
        \mathbf{e}^{(\text{ff})}_{i} \,\big]

    where :math:`\mathbf{e}^{(\text{ff})}_{i}` is present only if
    :code:`use_ff=True`. The resulting representation is treated as
    :math:`D` copies of the scalar even irrep :math:`0e`, i.e.
    :math:`\text{Irreps}(D\times 0e)`.

    :param use_ff:
        Whether to include force-field features (charge, mass, sigma, epsilon).
    :type use_ff: bool
    :param interp_time:
        Configuration for :class:`TimeEmbed` used on the graph-level interpolant time.
    :type interp_time: Any
    :param force_field:
        Namespace/struct with fields ``in_dim``, ``hidden_dims``, ``out_dim``,
        ``activation``, ``use_input_bn``, ``affine``, ``track_running_stats``.
        Required only if :code:`use_ff=True`.
    :type force_field: Any
    :param atom_type:
        Namespace/struct with fields ``num_types`` and ``embedding_dim``.
    :type atom_type: Any
    """
    def __init__(self, use_ff, interp_time, force_field, atom_type) -> None:
        super().__init__()
        self.use_ff = use_ff

        self.interpolant_time_embedder = TimeEmbed(interp_time)

        if self.use_ff:
            self.ff_embedder = MLP(
                in_dim=force_field.in_dim,
                hidden_dims=force_field.hidden_dims,
                out_dim=force_field.out_dim,
                activation=force_field.activation,
            )

        self.atom_type_embed = nn.Embedding(
            num_embeddings=atom_type.num_types,
            embedding_dim=atom_type.embedding_dim,
        )

    def forward(self, batch : Data) -> Data:
        r"""
        Build node features :math:`\mathbf{f}\in\mathbb{R}^{(B\!\cdot\!N)\times D}`
        by concatenating available embeddings and store them into the batch.

        Expected fields in :code:`batch`:
          - ``atom_type``: Long tensor of shape :math:`(B\!\cdot\!N, )`.
          - ``t_interpolant``: Float tensor of shape :math:`(B,)`.
          - If :code:`use_ff=True`: ``charge``, ``mass``, ``sigma``, ``epsilon``
            each of shape :math:`(B\!\cdot\!N,)`.

        .. math::

            \mathbf{e}^{(\text{ff})}_i
            = \text{MLP}\!\left(
              \begin{bmatrix} q_i & m_i & \sigma_i & \epsilon_i \end{bmatrix}
            \right),
            \qquad
            D = D_{\text{atom}} + D_t \;[+\, D_{\text{ff}}]

        :param batch:
            PyG data object with per-node atom types and per-graph time.
        :type batch: torch_geometric.data.Data

        :return:
            The same :code:`batch` with two new keys:
            ``'f'`` (node features, shape :math:`(B\!\cdot\!N, D)`)
            and ``'f_irrep'`` (:class:`e3nn.o3.Irreps`, equal to :math:`D\times 0e`).
        :rtype: torch_geometric.data.Data
        """
        # Expect: atom_type (B,N), interp_time (B,), and if use_ff: charge/mass/sigma/epsilon (B,N)
        atom_ty  = batch["atom_type"].long()
        t        = batch["t_interpolant"].float()
        
        atom_emb = self.atom_type_embed(atom_ty).squeeze(dim=1)               # (B*N, D_atom)

        t_emb = self.interpolant_time_embedder(t)[batch.batch] # (B*N, D_t)

        ff_emb = None
        parts = [atom_emb, t_emb]


        if self.use_ff:
            charge  = batch["charge"].float()
            mass    = batch["mass"].float()
            sigma   = batch["sigma"].float()
            epsilon = batch["epsilon"].float()
            ff_in = torch.cat([charge, mass, sigma, epsilon], dim=1)  # (B*N, 4)
            ff_emb = self.ff_embedder(ff_in)                          # (B*N, D_ff)
            parts.append(ff_emb)

        f = torch.cat(parts, dim=-1)  # (B*N, D_atom + D_t [+ D_ff])

        batch['f'] = f
        batch['f_irrep'] =Irreps(str(len(f[-1]))+"x0e")
        # TODO: must add this to the "keys" list. 
        batch['feature_keys'].add("f")

        return batch


class MLP(nn.Module):
    r"""
    A simple MLP with BatchNorm after the input (optional) and after every hidden
    linear layer.

    Layer pattern:
    :math:`\text{[BN]} \rightarrow (\text{Lin} \rightarrow \text{BN} \rightarrow \phi)^{L} \rightarrow \text{Lin}_{\text{out}}`.

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
    """
    def __init__(self, in_dim, hidden_dims=(128, 128), out_dim=1, activation='relu'):
        super().__init__()
        layers = []

        last = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(last, h),           # bias not needed if BN affine=True
                parse_activation(activation)
            ]
            last = h

        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the MLP to inputs.

        :param x:
            Input tensor of shape :math:`(B, \text{in_dim})`.
        :type x: torch.Tensor
        :return:
            Output tensor of shape :math:`(B, \text{out_dim})`.
        :rtype: torch.Tensor
        """
        return self.net(x)