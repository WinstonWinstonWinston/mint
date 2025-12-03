import torch
from torch_cluster import radius_graph
from torch_scatter import scatter,scatter_max
from e3nn import o3
import e3nn.nn as enn
from e3nn.o3 import Linear, Irreps
from e3nn.nn import BatchNorm
from e3nn.math import soft_unit_step, soft_one_hot_linspace
from mint.utils import channels_arr_to_string, parse_activation
from torch_geometric.utils import softmax
import torch.nn as nn
from mint.utils import combine_features
from e3nn.math import perm
import collections

_RESCALE=True

def sort_irreps_even_first(irreps):
    Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
    out = [(ir.l, -ir.p, i, mul) for i, (mul, ir) in enumerate(irreps)]
    out = sorted(out)
    inv = tuple(i for _, _, i, _ in out)
    p = perm.inverse(inv)
    irreps = o3.Irreps([(mul, (l, -p)) for l, p, _, mul in out])
    return Ret(irreps, p, inv)

class TensorProductRescale(torch.nn.Module):
    def __init__(self,
        irreps_in1, irreps_in2, irreps_out,
        instructions,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):
        
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.rescale = rescale
        self.use_bias = bias
        
        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        self.tp = o3.TensorProduct(irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2, irreps_out=self.irreps_out, 
            instructions=instructions, normalization=normalization,
            internal_weights=internal_weights, shared_weights=shared_weights, 
            path_normalization='none')
        
        self.init_rescale_bias()

        self.weight_numel = self.tp.weight_numel
    
    
    def calculate_fan_in(self, ins):
        return {
            'uvw': (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            'uvu': self.irreps_in2[ins.i_in2].mul,
            'uvv': self.irreps_in1[ins.i_in1].mul,
            'uuw': self.irreps_in1[ins.i_in1].mul,
            'uuu': 1,
            'uvuv': 1,
            'uvu<v': 1,
            'u<vw': self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
        }[ins.connection_mode]
        
        
    def init_rescale_bias(self) -> None:
        
        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        # Determine the order for each output tensor and their dims
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slices()
        
        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split('+')]
        self.irreps_bias_parity = [irrep_str[-1] for irrep_str in str(self.irreps_bias).split('+')]
        self.irreps_bias_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(self.irreps_bias).split('+')]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if self.irreps_bias_orders[slice_idx] == 0 and self.irreps_bias_parity[slice_idx] == 'e':
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype))
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)
       
        self.slices_sqrt_k = {}
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                            fan_in if slice_idx in slices_fan_in.keys() else fan_in)
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)
                
            # Re-initialize weights in each instruction
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    #else:
                    #    sqrt_k = 1.
                    #
                    #if self.rescale:
                        #weight.data.uniform_(-sqrt_k, sqrt_k)
                    #    weight.data.mul_(sqrt_k)
                    #self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            #for (out_slice_idx, out_slice, out_bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
            #    sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
            #    out_bias.uniform_(-sqrt_k, sqrt_k)
                

    def forward_tp_rescale_bias(self, x, y, weight=None):
        
        out = self.tp(x, y, weight)
        
        #if self.rescale and self.tp.internal_weights:
        #    for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
        #        out[:, slice] /= slice_sqrt_k
        if self.use_bias:
            for (_, slice, bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                #out[:, slice] += bias
                out.narrow(1, slice.start, slice.stop - slice.start).add_(bias)
        return out
        

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out

def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output, 
    internal_weights=False, bias=True):
    '''
        The irreps of output is pre-determined. 
        `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []
    
    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))
        
    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output) #irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
            irreps_output, instructions,
            internal_weights=internal_weights,
            shared_weights=internal_weights,
            bias=bias, rescale=_RESCALE)
    return tp,irreps_output    

class EquivariantLayerNorm(nn.Module):
    
    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
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

class SE3Transformer(torch.nn.Module):
    r"""
    Transformer for 3D graphs that uses spherical-tensor edge embeddings and
    irrep-typed node features, following
    `e3nn guide <https://docs.e3nn.org/en/stable/guide/transformer.html>`_ and
    `Fuchs et al., 2020 <https://arxiv.org/pdf/2006.10503>`_.

    :param _max_radius:
        Cutoff radius for building edges; pairs with :math:`r_{ij} >` this are skipped.
    :type _max_radius: float
    :param _number_of_basis:
        Radial basis size.
    :type _number_of_basis: int
    :param _hidden_size:
        Internal hidden channel size for projections/MLPs.
    :type _hidden_size: int
    :param act:
        Activation function identifier used in internal MLPs.
    :type act: str
    :param max_neighbors:
        Maximum neighbors per node (cap during graph construction).
    :type max_neighbors: int
    :param _irreps_sh:
        Spherical-harmonics irreps (sets :math:`\ell_{\max}`) for angular embeddings.
    :type _irreps_sh: e3nn.o3.Irreps
    :param _irreps_input:
        Irreps of input node features.
    :type _irreps_input: e3nn.o3.Irreps
    :param _irreps_output:
        Target irreps of output node features. Corresponds to values irreps.
    :type _irreps_output: e3nn.o3.Irreps
    :param _irreps_key:
        Irreps used for attention keys.
    :type _irreps_key: e3nn.o3.Irreps
    :param _irreps_query:
        Irreps used for attention queries.
    :type _irreps_query: e3nn.o3.Irreps
    :param _edge_basis:
        Radial basis constructor for :math:`\mathbf{b}(r)` (e.g., ``soft_one_hot_linspace`` from e3nn).
    :type _edge_basis: Callable
    """
    def __init__(self, _max_radius, _number_of_basis, _hidden_size, act, max_neighbors,
                 _irreps_sh,             # max rank to embed edges via spherical tensors
                 _irreps_input,          # e3nn irrep corresponding to input feature
                 _irreps_output,         # desired irrep corresponding to output feature
                 _irreps_key,            # desired irrep corresponding to keys
                 _irreps_query,          # desired irrep corresponding to query
                 _edge_basis,            # basis functions to use on edge emebeddings https://docs.e3nn.org/en/latest/api/math/math.html#e3nn.math.soft_one_hot_linspace
                 ): 

        super().__init__()
        
        self.max_radius = _max_radius
        self.max_neighbors = max_neighbors
        self.number_of_basis = _number_of_basis
        self.irreps_sh = _irreps_sh
        self.irreps_input = _irreps_input
        self.irreps_output = _irreps_output
        self.irreps_key = _irreps_key
        self.irreps_query = _irreps_query
        self.edge_basis = _edge_basis
        self.act = act

        self.tp_k,self.irreps_output_tpk = DepthwiseTensorProduct(self.irreps_input, self.irreps_sh, self.irreps_key, bias=False, internal_weights=False)
        #self.tp_k = o3.FullyConnectedTensorProduct(self.irreps_input, self.irreps_sh, self.irreps_key, shared_weights=False)
        self.fc_k = enn.FullyConnectedNet([self.number_of_basis, _hidden_size, self.tp_k.weight_numel], act=self.act)
        
        self.h_q = o3.Linear(self.irreps_input, self.irreps_query)

        # self.tp_v = o3.FullyConnectedTensorProduct(self.irreps_input, self.irreps_sh, self.irreps_output, shared_weights=False)
        self.tp_v,self.irreps_output_tpv = DepthwiseTensorProduct(self.irreps_input, self.irreps_sh, self.irreps_output, bias=False, internal_weights=False)
        self.fc_v = enn.FullyConnectedNet([self.number_of_basis, _hidden_size, self.tp_v.weight_numel], act=self.act)

        self.dot = o3.FullyConnectedTensorProduct(self.irreps_query, self.irreps_output_tpk, "0e")

        # Hard coded to stop the sqrt from being zero.
        self.eps = 1e-12

    def forward(self, f: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor, edge_src, edge_dst, edge_sh, edge_length_embedded, edge_weight_cutoff) -> torch.Tensor:
        r"""
        Run one SE3Transformer pass on a 3D graph.

        :param f:
            Node features, shape :math:`(N, C_\text{in})`.
        :type f: torch.Tensor
        :param pos:
            Node coordinates, shape :math:`(N, 3)`.
        :type pos: torch.Tensor
        :param batch:
            Graph id per node (PyG-style), shape :math:`(N,)`.
        :type batch: torch.Tensor
        :return:
            Updated node features, shape :math:`(N, C_\text{out})`.
        :rtype: torch.Tensor
        """
       
        q = self.h_q(f)
        k = self.tp_k(f[edge_src], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_length_embedded))


        logits = self.dot(q[edge_dst], k)
        # max_dst, _ = scatter_max(logits, edge_dst, dim=0, dim_size=len(f))
        
        # shifted = logits - max_dst[edge_dst] # Shift by max to avoid overflow inf/inf
        # exp = edge_weight_cutoff[:, None] * shifted.exp() # type: ignore
        # z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
        # z[z == 0] = 1

        alpha = softmax(logits, edge_dst) * edge_weight_cutoff[:, None]
        # inline code to check tensors
        # to_chk = {
        #     "f": f,
        #     "pos": pos,
        #     "edge_vec": edge_vec,
        #     "edge_length": edge_length,
        #     "edge_length_embedded": edge_length_embedded,
        #     "edge_weight_cutoff": edge_weight_cutoff,
        #     "edge_sh": edge_sh,
        #     "q": q,
        #     "k": k,
        #     "v": v,
        #     "exp": exp,
        #     "z": z,
        #     "alpha": alpha,
        # }

        # nan_report = {
        #     name: {
        #         "shape": tuple(t.shape),
        #         "nan_count": torch.isnan(t).sum().item(),
        #     }
        #     for name, t in to_chk.items()
        #     if torch.isnan(t).any()
        # }

        # dot_vals = self.dot(q[edge_dst], k)                 # logits before exp
        # overflow_mask = dot_vals > 80.0                     # 80 ≈ exp(80) ~ 5e34 (float32 max~3e38)

        # if overflow_mask.any():                             # only print when we really overflow
        #     idx = overflow_mask.nonzero(as_tuple=False)[:10]      # first few offenders
        #     print(
        #         ">>> exp overflow at", idx.shape[0], "edges.",
        #         "Sample logits:", dot_vals[idx.flatten()].tolist()[:5]
        #     )

        # for name, tensor in {"exp": exp, "z": z, "z_edge": z[edge_dst]}.items():
        #     if torch.isinf(tensor).any():
        #         cnt = torch.isinf(tensor).sum().item()
        #         print(f">>> {name} has {cnt:,} inf values (shape={tuple(tensor.shape)})")


        # if nan_report:                            # print only when something is wrong
        #     print(">>> NaNs detected:", nan_report)

        return scatter((alpha.relu() + self.eps).sqrt() * v, edge_dst, dim=0, dim_size=len(f))

class MultiSE3Transformer(torch.nn.Module):
    r"""
    TODO Describe me :)
    """
    def __init__(self, input_channels,
                 readout_channels,
                 hidden_channels,
                 hidden_channels_attn,
                 hidden_channels_mlp,
                 key_channels,
                 query_channels,
                 edge_l_max,
                 edge_basis, 
                 max_radius,
                 number_of_basis, 
                 hidden_size, 
                 max_neighbors, 
                 act,
                 num_layers
                 ) -> None:
        super().__init__()
    
        self.irreps_input = o3.Irreps(channels_arr_to_string(input_channels)).regroup()
        self.irreps_readout = o3.Irreps(channels_arr_to_string(readout_channels)).regroup()
        
        irreps_hidden = o3.Irreps(channels_arr_to_string(hidden_channels)).regroup()
        irreps_hidden_attn = o3.Irreps(channels_arr_to_string(hidden_channels_attn)).regroup()
        self.irreps_hidden_attn = irreps_hidden_attn
        irreps_hidden_mlp = o3.Irreps(channels_arr_to_string(hidden_channels_mlp)).regroup()

        irreps_key = o3.Irreps(channels_arr_to_string(key_channels)).regroup()
        irreps_query = o3.Irreps(channels_arr_to_string(query_channels)).regroup()

        self.irreps_sh = o3.Irreps.spherical_harmonics(edge_l_max).regroup()
        self.irreps_hidden = irreps_hidden
        self.edge_basis = edge_basis
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.hidden_size = hidden_size
        self.max_neighbors = max_neighbors

        act = parse_activation(act)

        self.lin_in = Linear(self.irreps_input, irreps_hidden)

        assert num_layers >= 1

        self.layer_norms_a = torch.nn.ModuleList()

        self.attention_0 = torch.nn.ModuleList()
        self.attention_1 = torch.nn.ModuleList()
        self.attention_2 = torch.nn.ModuleList()
        self.attention_3 = torch.nn.ModuleList()

        self.linear_layers_a = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        self.linear_layers_b = torch.nn.ModuleList()

        self.layer_norms_b = torch.nn.ModuleList()
        
        # Loop over the rest of the layers which map from irreps_output -> irreps_output
        for _ in range(num_layers):
            
            self.layer_norms_a.append(EquivariantLayerNorm(irreps_hidden))

            self.attention_0.append(SE3Transformer(max_radius, number_of_basis, hidden_size, act, max_neighbors,
                self.irreps_sh,      # max rank to embed edges via spherical tensors
                irreps_hidden,       # e3nn irrep corresponding to input feature
                irreps_hidden_attn,  # desired irrep corresponding to output feature
                irreps_key,          # desired irrep corresponding to keys
                irreps_query,        # desired irrep corresponding to query
                edge_basis,          # basis functions to use on edge emebeddings
                ))

            self.attention_1.append(SE3Transformer(max_radius, number_of_basis, hidden_size, act, max_neighbors,
                self.irreps_sh,           # max rank to embed edges via spherical tensors
                irreps_hidden,       # e3nn irrep corresponding to input feature
                irreps_hidden_attn,       # desired irrep corresponding to output feature
                irreps_key,          # desired irrep corresponding to keys
                irreps_query,        # desired irrep corresponding to query
                edge_basis,          # basis functions to use on edge emebeddings
                ))
            
            self.attention_2.append(SE3Transformer(max_radius, number_of_basis, hidden_size, act, max_neighbors,
                self.irreps_sh,      # max rank to embed edges via spherical tensors
                irreps_hidden,       # e3nn irrep corresponding to input feature
                irreps_hidden_attn,  # desired irrep corresponding to output feature
                irreps_key,          # desired irrep corresponding to keys
                irreps_query,        # desired irrep corresponding to query
                edge_basis,          # basis functions to use on edge emebeddings
                ))

            self.attention_3.append(SE3Transformer(max_radius, number_of_basis, hidden_size, act, max_neighbors,
                self.irreps_sh,      # max rank to embed edges via spherical tensors
                irreps_hidden,       # e3nn irrep corresponding to input feature
                irreps_hidden_attn,  # desired irrep corresponding to output feature
                irreps_key,          # desired irrep corresponding to keys
                irreps_query,        # desired irrep corresponding to query
                edge_basis,          # basis functions to use on edge emebeddings
                ))

            self.linear_layers_a.append(Linear(self.irreps_hidden, irreps_hidden_mlp))

            self.acts.append(
                enn.NormActivation(
                    irreps_in=irreps_hidden_mlp,
                    scalar_nonlinearity=act,
                    normalize=True, epsilon=1e-5, bias=True,
                )
            )

            self.linear_layers_b.append(Linear(irreps_hidden_mlp, irreps_hidden))

            self.layer_norms_b.append(EquivariantLayerNorm(irreps_hidden))
    
        self.readout_b = Linear(irreps_hidden, self.irreps_readout)
        # self.readout_eta = Linear(irreps_hidden, self.irreps_readout)

    def forward(self, batch):
        batch_idx = batch['batch']      # for radius_graph
        node_feats = batch['f']
        pos = batch['x_t']

        # convert shape with linear layer
        node_feats = self.lin_in(batch['f'])

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
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))

        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization='component')

        # --------- PAY ATTENTION!!! --------- 
        for i in range(len(self.attention_0)):

            node_feats = self.layer_norms_a[i](node_feats)

            node_feat_update = []

            node_feat_update.append(self.attention_0[i](
                f=node_feats,
                pos=pos,
                batch=batch_idx,
                edge_src=edge_src, 
                edge_dst=edge_dst, 
                edge_sh=edge_sh, 
                edge_length_embedded=edge_length_embedded, 
                edge_weight_cutoff=edge_weight_cutoff
            ))

            node_feat_update.append(self.attention_1[i](
                f=node_feats,
                pos=pos,
                batch=batch_idx,
                edge_src=edge_src, 
                edge_dst=edge_dst, 
                edge_sh=edge_sh, 
                edge_length_embedded=edge_length_embedded, 
                edge_weight_cutoff=edge_weight_cutoff
            ))

            node_feat_update.append(self.attention_2[i](
                f=node_feats,
                pos=pos,
                batch=batch_idx,
                edge_src=edge_src, 
                edge_dst=edge_dst, 
                edge_sh=edge_sh, 
                edge_length_embedded=edge_length_embedded, 
                edge_weight_cutoff=edge_weight_cutoff
            ))

            node_feat_update.append(self.attention_3[i](
                f=node_feats,
                pos=pos,
                batch=batch_idx,
                edge_src=edge_src, 
                edge_dst=edge_dst, 
                edge_sh=edge_sh, 
                edge_length_embedded=edge_length_embedded, 
                edge_weight_cutoff=edge_weight_cutoff
            ))

            node_feat_update,_ = combine_features([(node_feat_update[0], self.irreps_hidden_attn),
                                                   (node_feat_update[1], self.irreps_hidden_attn),
                                                   (node_feat_update[2], self.irreps_hidden_attn),
                                                   (node_feat_update[3], self.irreps_hidden_attn)],tidy=True)
            
            node_feats = node_feats + node_feat_update 

            node_feat_update = self.linear_layers_a[i](node_feats)
            node_feat_update = self.acts[i](node_feat_update)
            node_feat_update = self.linear_layers_b[i](node_feat_update)

            node_feats = node_feats + node_feat_update

            node_feats = self.layer_norms_b[i](node_feats)
            
        batch['b'] = self.readout_b(node_feats)
        batch['eta'] = torch.zeros_like(batch['b'])

        batch['b_irrep'] = Irreps("1e"),
        batch['eta_irrep'] = Irreps("1e")

        return batch
    