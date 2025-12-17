import torch.nn as nn
from e3nn.math import  soft_one_hot_linspace
from torch_cluster import radius_graph
import torch
from e3nn import o3
from torch import nn
from torch_geometric.nn import TransformerConv

class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim, heads=4, num_layers=3):
        super().__init__()

        self.convs = nn.ModuleList()

        # First layer: in_dim -> hidden_dim
        self.convs.append(
            TransformerConv(
                in_channels=in_dim,
                out_channels=hidden_dim // heads,  # will be multiplied by heads
                heads=heads,
                dropout=0.1,
                edge_dim = edge_dim,
            )
        )

        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    dropout=0.1,
                    edge_dim = edge_dim,
                )
            )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        # x: [num_nodes, in_dim]
        # edge_index: [2, num_edges]
        # batch: [num_nodes] (graph indices)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = x.relu()
        x = self.mlp(x)
        return x
    
class GraphTransformerInterpolantNet(nn.Module):

    def __init__(self,in_dim, 
                      in_dim_cond, hidden_dim_cond, out_dim_cond, heads_cond, num_layers_cond,
                      in_dim_b,    hidden_dim_b,    out_dim_b,    heads_b,    num_layers_b   ,
                      in_dim_eta,  hidden_dim_eta,  out_dim_eta,  heads_eta,  num_layers_eta ,
                      edge_l_max, max_radius, max_neighbors, number_of_basis, edge_basis     ):

        super().__init__()

        self.max_radius = max_radius
        self.max_neighbors = max_neighbors
        self.number_of_basis = number_of_basis
        self.edge_basis = edge_basis
        self.irreps_sh = o3.Irreps.spherical_harmonics(edge_l_max,p=1).regroup()

        edge_dim = self.number_of_basis + (edge_l_max + 1) ** 2

        self.conditioner = GraphTransformer(in_dim_cond, hidden_dim_cond, out_dim_cond, edge_dim, heads_cond, num_layers_cond)
        self.b_net       = GraphTransformer(in_dim_b,    hidden_dim_b,    out_dim_b,    edge_dim, heads_b,    num_layers_b   )
        self.eta_net     = GraphTransformer(in_dim_eta,  hidden_dim_eta,  out_dim_eta,  edge_dim, heads_eta,  num_layers_eta   )
        
        self.linear_in = nn.Linear(in_features=in_dim, out_features=in_dim_cond)
        
    def forward(self,batch):
        f = self.linear_in(batch['f'])
        batch_idx = batch['batch']
        B = max(batch_idx+1)
        pos = batch['x_t']
        edge_idx = radius_graph(x=pos, 
                                          r=self.max_radius, 
                                          batch=batch_idx, 
                                          loop=False,
                                          max_num_neighbors=self.max_neighbors)
        edge_src, edge_dst = edge_idx
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis=self.edge_basis,
            cutoff=True
        )
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5)
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization='component')
        edge_attr = torch.cat([edge_length_embedded, edge_sh], dim=-1)

        f = self.conditioner(f, edge_idx, batch_idx, edge_attr)

        batch['f_cond'] = f
        batch['f_cond_irrep'] = o3.Irreps(f'{len(f[0])}x0e')
        batch['feature_keys'].add('f_cond')

        b = self.b_net(f, edge_idx, batch_idx, edge_attr)
        eta = self.eta_net(f, edge_idx, batch_idx, edge_attr)

        batch['b'] =  b
        batch['b_irrep'] = o3.Irreps("1x1e")
        batch['feature_keys'].add('b')
        batch['eta'] = eta
        batch['eta_irrep'] = o3.Irreps("1x1e")
        batch['feature_keys'].add('eta')

        return batch