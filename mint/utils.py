import torch
import numpy as np
import random
import os
from torch import nn
from torch_geometric.nn.pool import radius_graph
from e3nn import o3
from torch import Tensor

def batch2loss(batch,stratified=False):
    return batch['t_interpolant'][batch.batch][:,None], batch['x_base'], batch['x'], batch['z'], batch['b'],batch['eta']

def batch2interp(batch):
    return batch['t_interpolant'][batch.batch][:,None], batch['x_base'], batch['x']

#### Train utils ####

def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Ensuring deterministic behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#### Model utils ####

def channels_arr_to_string(query_channels):
    even, odd = query_channels

    parts = []

    if even:
        parts.extend(f"{v}x{l}e" for l, v in enumerate(even))
    if odd:
        parts.extend(f"{v}x{l}o" for l, v in enumerate(odd))

    return " + ".join(parts)

def parse_activation(spec: str) -> torch.nn.Module:
    """
    Parse an activation spec string and return an nn.Module.

    Examples
    --------
    >>> parse_activation('relu')
    >>> parse_activation('leaky_relu:0.2')
    >>> parse_activation('SiLU')
    """
    name, *tail = spec.lower().split(':', 1)

    simple = {
        'relu':   lambda: nn.ReLU(inplace=True),
        'leakyrelu':   lambda: nn.LeakyReLU(inplace=True),
        'gelu':   nn.GELU,
        'silu':   nn.SiLU,
        'swish':  nn.SiLU,
        'sigmoid': nn.Sigmoid,
        'tanh':   nn.Tanh,
        'identity': nn.Identity,
        'none':    nn.Identity,
    }
    if name in simple:
        return simple[name]()

    # one‑liner LeakyReLU with optional slope:  'leaky_relu'  or  'leaky_relu:0.05'
    if name in ('leaky_relu', 'lrelu'):
        slope = float(tail[0]) if tail else 0.01
        return nn.LeakyReLU(negative_slope=slope, inplace=True)

    raise ValueError(f"Unknown activation spec: '{spec}'")

def repeat_interleave(repeats):
        outs = [torch.full((n, ), i) for i, n in enumerate(repeats)]
        return torch.cat(outs, dim=0).to(repeats.device)

def periodic_radius_graph(pos, batch, rcut, cell_lengths):
    """
    pos:           (N,3)  atom positions
    batch:         (N,)   graph indices
    rcut:          float  cutoff distance
    cell_lengths:  (B,3)  orthorhombic box lengths for each graph
    Returns:
      edge_index:  (2,E)  [src, dst] pairs (may include multiple p‐image edges)
      edge_vec:    (E,3)  the corresponding displacement vectors
    """
    device = pos.device

    # determine how many images needed along each axis
    # for each graph in the batch, compute ceil(rcut / L_i) and then take the max
    # so we cover the worst‐case.
    per_graph = torch.ceil(rcut / cell_lengths).long()             # (B,3)
    max_shift = per_graph.max(dim=0).values                         # (3,)

    # build the complete list of shifts: range(-n … +n) for each axis
    ranges = [torch.arange(-n, n+1, device=device) for n in max_shift.tolist()]
    shifts = torch.stack(torch.meshgrid(*ranges), dim=-1).view(-1,3)   # (P,3), P = prod(2*n_i+1)

    # tile your positions and batches
    #    pos_img[i,k] = pos[i] + shifts[k] * cell_lengths[ batch[i] ]
    cell_per_node = cell_lengths[batch]                                # (N,3)
    pos_img   = pos.unsqueeze(1)   + shifts.unsqueeze(0) * cell_per_node.unsqueeze(1)  # (N,P,3)
    batch_img = batch.unsqueeze(1).repeat(1, shifts.size(0))                                       # (N,P)

    # flatten and run radius_graph
    pos_rep   = pos_img.view(-1,3)                                     # (N*P,3)
    batch_rep = batch_img.view(-1)                                     # (N*P,)
    edge_index_rep = radius_graph(pos_rep, rcut, batch_rep, loop=False) # still drops trivial self‐loop

    # map back to original node indices
    src_rep, dst_rep = edge_index_rep                                  # each (E,)
    orig_src = src_rep // shifts.size(0)
    orig_dst = dst_rep // shifts.size(0)

    # compute the true displacement vectors
    edge_vec = pos_rep[dst_rep] - pos_rep[src_rep]                     # (E,3)

    return orig_src, orig_dst, edge_vec

def _permute_for_regroup(
    tensor: Tensor,
    old_irreps: o3.Irreps,
    new_irreps: o3.Irreps,
    dim: int = -1,
):
    """Re-order the feature channels so `tensor` matches `new_irreps`."""
    # Build (irrep, slice) list for the old layout
    slices_old = []
    start = 0
    for mul, ir in old_irreps:
        w = mul * ir.dim
        slices_old.append((ir, slice(start, start + w)))
        start += w

    # Assemble new channel order: for each (l,p) in new_irreps, grab
    # all matching slices from the *old* tensor (original intra‑mul order).
    order = []
    for mul_new, ir_new in new_irreps: # type: ignore
        for ir_old, slc in slices_old:
            if ir_old == ir_new:
                order.extend(range(slc.start, slc.stop))

    idx = torch.tensor(order, device=tensor.device)
    return tensor.index_select(dim, idx)


def combine_features(
    pairs,
    *,
    dim: int = -1,
    tidy: bool = False,        # <- default keeps raw order
):
    """
    Concatenate feature blocks and return (tensor, irreps).

    If `tidy=True` the result is canonical (`regroup()`); the function
    then permutes the channels so data and metadata agree.
    """
    xs  = [x for x, _ in pairs]
    irs = [o3.Irreps(ir) for _, ir in pairs]

    # 1) raw concatenation
    tensor_cat = torch.cat(xs, dim=dim)
    irreps_cat = sum(irs, o3.Irreps())      # direct sum in given order

    if tidy:
        irreps_new = irreps_cat.regroup()   # type: ignore # sort + merge
        tensor_cat = _permute_for_regroup(tensor_cat, irreps_cat, irreps_new, dim) # type: ignore
        irreps_cat = irreps_new             # switch to canonical form

    return tensor_cat, irreps_cat

def save_xyz(
    trajectory: torch.Tensor,
    atomic_numbers: list[int] | torch.Tensor,
    prefix: str = "output",
):
    """
    Save a trajectory of shape (steps, B, N, 3) as one XYZ file per batch,
    using atomic numbers for proper element symbols.

    Parameters
    ----------
    trajectory : torch.Tensor
        Tensor of shape (steps, B, N, 3)
    atomic_numbers : list[int] or torch.Tensor
        Atomic numbers of shape (N,)
    prefix : str
        Output file prefix; files will be named '{prefix}_{b}.xyz'
    """
    steps, B, N, _ = trajectory.shape

    if isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.tolist()

    # Periodic table mapping for atomic numbers 1–20, fallback to "X"
    periodic_table = { 0: "H",
        1: "H",  2: "He", 3: "Li", 4: "Be", 5: "B",  6: "C",  7: "N",  8: "O",  9: "F", 10: "Ne",
        11: "Na",12: "Mg",13: "Al",14: "Si",15: "P",16: "S",17: "Cl",18: "Ar",19: "K", 20: "Ca",
    }

    symbols = [periodic_table.get(z, "X") for z in atomic_numbers]

    for b in range(B):
        with open(f"{prefix}_{b}.xyz", "w") as f:
            for step in range(steps):
                f.write(f"{N}\n")
                f.write(f"Frame {step}\n")
                for atom in range(N):
                    x, y, z = trajectory[step, b, atom]
                    symbol = symbols[atom]
                    f.write(f"{symbol} {x:.3f} {y:.3f} {z:.3f}\n")