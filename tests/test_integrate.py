from mint.data.loader import make_meta_collate
from mint.state import MINTState
from mint.data.ADP.ADP_dataset import ADPDataset
from mint.module import EquivariantMINTModule
from mint.experiment.train import Train
from mint.experiment.generate import Generate
from mint import utils
import torch
from torch import nn
from e3nn.o3 import Irreps
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import parmed as pmd

total_frames_train = 25600
total_frames_test = 6400
total_frames_valid = 6400

ds_train = ADPDataset(data_dir='/users/1/sull1276/mint/tests/../mint/data/ADP', 
                       data_proc_fname="AA", 
                       data_proc_ext=".pkl.zst", 
                       data_raw_fname="alanine-dipeptide-250ns-nowater", 
                       data_raw_ext=".xtc", 
                       split="train", 
                       total_frames_train=total_frames_train, 
                       total_frames_test=total_frames_test, 
                       total_frames_valid=total_frames_valid, 
                       lag= OmegaConf.create({"equilibrium": True}), 
                       normalize= OmegaConf.create({"bool": True, "t_dependent": False}), 
                       node_features= OmegaConf.create({"epsilon": True, "sigma": True, "charge": True, "mass": True}), 
                       augement_rotations=False)

ds_test = ADPDataset(data_dir='/users/1/sull1276/mint/tests/../mint/data/ADP', 
                       data_proc_fname="AA", 
                       data_proc_ext=".pkl.zst", 
                       data_raw_fname="alanine-dipeptide-250ns-nowater", 
                       data_raw_ext=".xtc", 
                       split="test", 
                       total_frames_train=total_frames_train, 
                       total_frames_test=total_frames_test, 
                       total_frames_valid=total_frames_valid, 
                       lag= OmegaConf.create({"equilibrium": True}), 
                       normalize= OmegaConf.create({"bool": True, "t_dependent": False}), 
                       node_features= OmegaConf.create({"epsilon": True, "sigma": True, "charge": True, "mass": True}), 
                       augement_rotations=False)

ds_valid = ADPDataset(data_dir='/users/1/sull1276/mint/tests/../mint/data/ADP', 
                       data_proc_fname="AA", 
                       data_proc_ext=".pkl.zst", 
                       data_raw_fname="alanine-dipeptide-250ns-nowater", 
                       data_raw_ext=".xtc", 
                       split="valid", 
                       total_frames_train=total_frames_train, 
                       total_frames_test=total_frames_test, 
                       total_frames_valid=total_frames_valid, 
                       lag= OmegaConf.create({"equilibrium": True}), 
                       normalize= OmegaConf.create({"bool": True, "t_dependent": False}), 
                       node_features= OmegaConf.create({"epsilon": True, "sigma": True, "charge": True, "mass": True}), 
                       augement_rotations=False)

module = EquivariantMINTModule(
    cfg=OmegaConf.create({
        "prior": {
            "_target_": "mint.prior.normal.NormalPrior",
            "mean": 0.0,
            "std": 0.5,
        },
        "embedder": {
            "_target_": "mint.model.embedding.equilibrium_embedder.EquilibriumEmbedder",
            "use_ff": True,
            "interp_time": {
                "embedding_dim": 256,
                "max_positions": 1000,
            },
            "force_field": {
                "in_dim": 4,
                "hidden_dims": [128, 64],
                "out_dim": 32,
                "activation": "silu",
                "use_input_bn": False,
                "affine": False,
                "track_running_stats": False,
            },
            "atom_type": {
                "num_types": 14,
                "embedding_dim": 32,
            },
        },
        "model": {
            "_target_": "mint.model.equivariant.PaiNN.PaiNNLikeInterpolantNet",
            "irreps_input":        [[320  ], [0    ]],
            "irreps":              [[32, 0], [0, 32]],
            "irreps_readout_cond": [[32, 0], [0, 32]],
            "irreps_readout":      [[0, 0],  [0, 1 ]],
            "edge_l_max": 1,
            "max_radius": 1000,
            "max_neighbors": 1000,
            "number_of_basis": 64,
            "edge_basis": "gaussian",
            "mlp_act": "silu",
            "mlp_drop": 0,
            "conv_weight_layers": [192],
            "update_weight_layers": [128],
            "message_update_count_cond": 2,
            "message_update_count_eta": 2,
            "message_update_count_b": 2,
        },
        "interpolant": {
            "_target_": "mint.interpolant.interpolants.TemporallyLinearInterpolant",
            "velocity_weight": 1.0,
            "denoiser_weight": 1.0,
            "gamma_weight": 1/10,
        },
        "validation": {
            "stratified": False,
        },
        "optim": {
            "optimizer": {
                "name": "Adam",
                "lr": 5e-4,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "name": "CosineAnnealingLR",
                "T_max": "experiment.train.trainer.max_epochs",
                "eta_min": 1e-6,
            },
        },
    })
)

class InterpolantWrapper(nn.Module):
    def __init__(self, interpolant,prior):
        super().__init__()
        self.interpolant = interpolant
        self.device = 'cuda'
        self.prior = prior

    def forward(self, batch):
        # if you really need x_t for something later:
        z = torch.randn_like(batch['x'])#torch.randn_like(batch['x'])
        t,x_0,x_1 = utils.batch2interp(batch)

        interpolant_dot = self.interpolant.interpolate_derivative(t, x_0, batch['x_target'], torch.zeros_like(z))

        batch["b"] = interpolant_dot
        batch["eta"] = z

        batch["b_irrep"] = Irreps("1e")
        batch["eta_irrep"] = Irreps("1e")

        return batch
    
model_interpolant = InterpolantWrapper(module.interpolant,module.prior)

ckpt = torch.load("logs/hydra/ckpt/epoch_148-step_29800-loss_-11911615.0000.ckpt", map_location="cuda")
module.load_state_dict(ckpt["state_dict"])

st = MINTState(
    seed=42,
    module=model_interpolant.to('cuda'),
    dataset_train=ds_train,
    dataset_valid=ds_valid,
    dataset_test=ds_test,
)

print(module)

subset = Subset(ds_test, range(64))

loader = DataLoader(
    subset,
    shuffle=False,
    batch_size=64,
    collate_fn = make_meta_collate(ds_train.meta_keys)
)
def epsilon_fn(t):
    return torch.ones_like(t)*0.1
    
generate_cfg = OmegaConf.create(
    {   "dt": 1e-4,
        "step_type": "ode", # or "sde"
        "clip_val": 1e-10,
        "save_traj": True
    }
)

gen_experiment = Generate(state=st, cfg=generate_cfg, batches = loader, epsilon=epsilon_fn)

with torch.no_grad():
    samples = gen_experiment.run()

X = [sample['x_traj'] for sample in samples][0]

print(X.size())

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
    B, N, _ = trajectory.shape

    if isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.tolist()

    # Periodic table mapping for atomic numbers 1–20, fallback to "X"
    periodic_table = { 0: "H",
        1: "H",  2: "He", 3: "Li", 4: "Be", 5: "B",  6: "C",  7: "N",  8: "O",  9: "F", 10: "Ne",
        11: "Na",12: "Mg",13: "Al",14: "Si",15: "P",16: "S",17: "Cl",18: "Ar",19: "K", 20: "Ca",
    }

    symbols = [periodic_table.get(z, "X") for z in atomic_numbers]
    with open(f"{prefix}.xyz", "w") as f:
        for b in range(B):
            f.write(f"{N}\n")
            f.write(f"Frame {b}\n")
            for atom in range(N):
                x, y, z = trajectory[b, atom]
                symbol = symbols[atom]
                f.write(f"{symbol} {x:.3f} {y:.3f} {z:.3f}\n")

atomic_numbers = [a.atomic_number for a in pmd.load_file("../mint/data/ADP/alanine-dipeptide-nowater.pdb").atoms]

def save_trajectories_per_graph(X, atomic_numbers, base_prefix="traj",max_num=3):
    T, N_total, _ = X.shape
    n_nodes = 22
    assert N_total % n_nodes == 0
    n_graphs = N_total // n_nodes             # 64

    for g in range(min(n_graphs,max_num)):
        start = g * n_nodes
        end = (g + 1) * n_nodes
        X_graph = X[:, start:end, :]          # (T, 22, 3)

        prefix = f"{base_prefix}_graph{g:03d}"
        save_xyz(X_graph, atomic_numbers, prefix=prefix)

save_trajectories_per_graph(X, atomic_numbers, base_prefix="traj")