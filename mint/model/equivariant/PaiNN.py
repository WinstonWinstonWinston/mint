import torch.nn as nn
import torch
from e3nn.math import  soft_one_hot_linspace
from torch_scatter import scatter
from torch_cluster import radius_graph
from mint.utils import combine_features,channels_arr_to_string,parse_activation

import torch
from e3nn import o3

import torch


class EquivariantLayerNorm(nn.Module):
    
    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization


    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps}, eps={self.eps})"


    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input, **kwargs):
        # batch, *size, dim = node_input.shape  # TODO: deal with batch
        # node_input = node_input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # node_input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the node_input batch slices this into separate graphs
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            #field = node_input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            field = node_input.narrow(1, ix, mul*d)
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0 and ir.p == 1:
                # Compute the mean
                field_mean = torch.mean(field, dim=1, keepdim=True) # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean
                
            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)    

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]
            
            if self.affine:
                weight = self.affine_weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]
            
            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]
            
            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output

def scale_shared_channels(
    f: torch.Tensor,
    scalars: torch.Tensor,
    irrep_index: dict[int, torch.Tensor],
    channel_count: int,
):
    """
    Scale features with shared channel gates across all ranks, in-place on f.
    f:       [B, C]
    scalars: [B, C_ch], where C_ch is the number of channels per rank l.
    """
    B, C = f.shape
    B_s, C_ch = scalars.shape
    assert B_s == B
    assert C_ch == channel_count

    for l, idx in irrep_index.items():
        dim_l = 2 * l + 1
        n_l = idx.numel()
        assert n_l == channel_count * dim_l

        block = f[:, idx].view(B, channel_count, dim_l)
        block *= scalars.unsqueeze(-1)
        f[:, idx] = block.view(B, -1)

    return f


def get_rank_l_channel_count(irrep_index: dict[int, torch.Tensor], l: int) -> int:
    """
    Number of channels (multiplicity) for rank l.
    """
    if l not in irrep_index:
        return 0
    n = irrep_index[l].numel()
    d = 2 * l + 1
    assert n % d == 0
    return n // d


def assert_all_ranks_same_channel_count(irrep_index: dict[int, torch.Tensor]):
    """
    Assert that all l have the same channel count (multiplicity).
    """
    if not irrep_index:
        return

    ls = sorted(irrep_index.keys())
    base_l = ls[0]
    base_count = get_rank_l_channel_count(irrep_index, base_l)

    for l in ls[1:]:
        count = get_rank_l_channel_count(irrep_index, l)
        assert count == base_count, (
            f"Channel count mismatch: l={base_l} has {base_count}, "
            f"l={l} has {count}"
        )


def get_irrep_index(irreps: o3.Irreps) -> dict[int, torch.Tensor]:
    """
    Return {l: LongTensor[n_l]} where n_l = C_ch(l) * (2l + 1).
    """
    irrep_index: dict[int, torch.Tensor] = {}
    offset = 0

    for mul, ir in irreps:
        dim_ir = ir.dim  # 2*l + 1
        block_dim = mul * dim_ir
        idx = torch.arange(offset, offset + block_dim, dtype=torch.long)

        if ir.l in irrep_index:
            irrep_index[ir.l] = torch.cat([irrep_index[ir.l], idx], dim=0)
        else:
            irrep_index[ir.l] = idx

        offset += block_dim

    return irrep_index


def get_rank_l(f: torch.Tensor, irrep_index: dict[int, torch.Tensor], l: int):
    """
    Extract packed components of rank l from f: [B, C] -> [B, C_l].
    """
    if l not in irrep_index:
        return f.new_empty(f.size(0), 0)

    idx = irrep_index[l].to(f.device)
    return f.index_select(-1, idx)

class PaINNLayer(nn.Module):

    def __init__(self, irreps,irreps_sh,
                       mlp_act, mlp_drop,number_of_basis,
                       conv_weight_layers, update_weight_layers):

        super().__init__()

        self.irreps = irreps
        self.irreps_sh = irreps_sh
        self.number_of_basis = number_of_basis
        self.irreps_idx = get_irrep_index(irreps)
        pairs = [ (self.irreps.randn((-1)),self.irreps),
            (self.irreps.randn((-1)),self.irreps)]
        _,irreps_cat = combine_features(pairs,tidy=True)
        self.irreps_cat_idx = get_irrep_index(irreps_cat)

        assert_all_ranks_same_channel_count(self.irreps_idx)

        self.channel_count = get_rank_l_channel_count(self.irreps_idx,0)

        conv_weight_modules = []
        self.tp_conv = o3.FullyConnectedTensorProduct(self.irreps, self.irreps_sh, self.irreps, shared_weights=False)
        self.linear_conv = o3.Linear(irreps_cat, self.irreps)
        # ensure we map from edge basis count + 2*channel_count to the conv weight size
        conv_weight_layers = [self.number_of_basis+2*self.channel_count] + conv_weight_layers + [self.tp_conv.weight_numel]
        for in_features, out_features in zip(conv_weight_layers[:-1], conv_weight_layers[1:]):
            conv_weight_modules.append(nn.Linear(in_features, out_features))
            conv_weight_modules.append(mlp_act)
            conv_weight_modules.append(torch.nn.Dropout(p=mlp_drop))
        self.conv_weight = nn.Sequential(*conv_weight_modules[:-2]) # exclude dropout and activation on the last pass to put in (-infty,infty)

        update_weight_modules = []
        self.tp_square = o3.TensorSquare(self.irreps, self.irreps)
        self.linear_update = o3.Linear(irreps_cat, self.irreps)
        # ensure we map from 2*channel_count to 2*channel_count
        update_weight_layers = [2*self.channel_count] + update_weight_layers + [2*self.channel_count] 
        for in_features, out_features in zip(update_weight_layers[:-1], update_weight_layers[1:]):
            update_weight_modules.append(nn.Linear(in_features, out_features))
            update_weight_modules.append(mlp_act)
            update_weight_modules.append(torch.nn.Dropout(p=mlp_drop))
        self.update_weight = nn.Sequential(*update_weight_modules[:-2]) # exclude dropout and activation on the last pass to put in (-infty,infty)
        self.layer_norm = EquivariantLayerNorm(self.irreps)

    def message(self, f, edge_src, edge_dst, edge_sh, edge_length_embedded):
        # get scalar features
        scalar_feat_f = get_rank_l(f,self.irreps_idx,0)
        # concat scalar feats at dst, src, and embedded edge features
        scalar_feats = torch.cat([scalar_feat_f[edge_dst], scalar_feat_f[edge_src], edge_length_embedded],dim=-1) # [B, 2*C_0 + number_of_basis]
        # apply tensor product
        f_tp = self.tp_conv(f[edge_src], edge_sh, self.conv_weight(scalar_feats))
        # aggregate using mean
        f_tp = scatter(f_tp, edge_dst, dim=0, dim_size=len(f))
        pairs = [ (f,self.irreps),
                  (f_tp,self.irreps)]
        f_cat,_ = combine_features(pairs,tidy=True)
        return self.linear_conv(f_cat)
        
    def update(self, f):
        pairs = [(f,self.irreps),
                (self.tp_square(f),self.irreps)]
        f_cat,_ = combine_features(pairs,tidy=True)
        scalar_feat_f_cat = get_rank_l(f_cat,self.irreps_cat_idx,0)
        w = self.update_weight(scalar_feat_f_cat)
        f_cat = scale_shared_channels(f_cat,w,self.irreps_cat_idx,2*self.channel_count)
        return self.layer_norm(self.linear_update(f_cat))

    def forward(self, f, edge_src, edge_dst, edge_sh, edge_length_embedded):
        f = self.message(f, edge_src, edge_dst, edge_sh, edge_length_embedded)
        f = self.update(f)
        return f
    
class PaiNNLike(nn.Module):

    def __init__(self, irreps, irreps_readout, edge_l_max,
                       mlp_act, mlp_drop,number_of_basis,
                       conv_weight_layers, update_weight_layers,
                       message_update_count):

        super().__init__()

        self.irreps = o3.Irreps(channels_arr_to_string(irreps)).regroup()
        self.irreps_readout =  o3.Irreps(channels_arr_to_string(irreps_readout)).regroup()
        self.irreps_sh = o3.Irreps.spherical_harmonics(edge_l_max,p=1).regroup()
        self.number_of_basis = number_of_basis
        self.message_update_count = message_update_count

        self.net_residual = torch.nn.ModuleList()
        self.net_cat = torch.nn.ModuleList()
        self.net_linears = torch.nn.ModuleList()

        pairs = [ (self.irreps.randn((-1)),self.irreps),
                  (self.irreps.randn((-1)),self.irreps),
                  (self.irreps.randn((-1)),self.irreps)]
        
        _, linear_irreps_in = combine_features(pairs,tidy=True)

        for _ in range(message_update_count):

            self.net_residual.append(PaINNLayer(self.irreps,self.irreps_sh,
                                       parse_activation(mlp_act), mlp_drop,number_of_basis,
                                       conv_weight_layers, update_weight_layers))
            
            self.net_cat.append(PaINNLayer(self.irreps,self.irreps_sh,
                                       parse_activation(mlp_act), mlp_drop,number_of_basis,
                                       conv_weight_layers, update_weight_layers))
            
            self.net_linears.append(o3.Linear(linear_irreps_in,self.irreps))

        self.readout = o3.Linear(self.irreps, self.irreps_readout)
    
    def forward(self, f, edge_src, edge_dst, edge_sh, edge_length_embedded):
    
        for i in range(self.message_update_count):
            f_residual =  self.net_residual[i](f, edge_src, edge_dst, edge_sh, edge_length_embedded)
            f_cat =   self.net_cat[i](f+f_residual, edge_src, edge_dst, edge_sh, edge_length_embedded)
            pairs = [ (f,self.irreps),
                      (f+f_residual,self.irreps),
                      (f_cat,self.irreps)]
            f, _ = combine_features(pairs,tidy=True)
            f = self.net_linears[i](f)

        return self.readout(f)

class PaiNNLikeInterpolantNet(nn.Module):

    def __init__(self,irreps_input,irreps, irreps_readout_cond, irreps_readout, edge_l_max,
                max_radius,max_neighbors,number_of_basis,edge_basis,
                mlp_act, mlp_drop,
                conv_weight_layers, update_weight_layers,
                message_update_count_cond, message_update_count_b, message_update_count_eta):

        super().__init__()

        self.max_radius = max_radius
        self.max_neighbors = max_neighbors
        self.number_of_basis = number_of_basis
        self.edge_basis = edge_basis
        self.irreps_sh = o3.Irreps.spherical_harmonics(edge_l_max,p=1).regroup()

        self.conditioner = PaiNNLike(irreps, irreps_readout_cond, edge_l_max,
                                     mlp_act, mlp_drop,number_of_basis,
                                     conv_weight_layers, update_weight_layers,
                                     message_update_count_cond)
        
        self.b_net = PaiNNLike(irreps_readout_cond, irreps_readout, edge_l_max,
                                     mlp_act, mlp_drop,number_of_basis,
                                     conv_weight_layers, update_weight_layers,
                                     message_update_count_b)

        self.eta_net = PaiNNLike(irreps_readout_cond, irreps_readout, edge_l_max,
                                     mlp_act, mlp_drop,number_of_basis,
                                     conv_weight_layers, update_weight_layers,
                                     message_update_count_eta)
        
        self.linear_in = o3.Linear( o3.Irreps(channels_arr_to_string(irreps_input)).regroup(), self.conditioner.irreps)
        
    def forward(self,batch):
        f = self.linear_in(batch['f'])
        batch_idx = batch['batch']
        pos = batch['x_t']
        edge_src, edge_dst = radius_graph(x=pos, 
                                          r=self.max_radius, 
                                          batch=batch_idx, 
                                          loop=False,
                                          max_num_neighbors=self.max_neighbors)
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

        f = self.conditioner(f, edge_src, edge_dst, edge_sh, edge_length_embedded)

        batch['f_cond'] = f
        batch['f_cond_irrep'] = self.conditioner.irreps_readout
        batch['feature_keys'].add('f_cond')

        b = self.b_net(f, edge_src, edge_dst, edge_sh, edge_length_embedded)
        eta = self.eta_net(f, edge_src, edge_dst, edge_sh, edge_length_embedded)

        batch['b'] =  b
        batch['b_irrep'] = o3.Irreps("1x1e")
        batch['feature_keys'].add('b')
        batch['eta'] = eta
        batch['eta_irrep'] = o3.Irreps("1x1e")
        batch['feature_keys'].add('eta')

        return batch
