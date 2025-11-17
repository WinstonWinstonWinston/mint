from mint.state import MINTState
from mint.data.ADP.ADP_dataset import ADPDataset
from mint.module import MINTModule
from mint.experiment.train import Train
from mint.experiment.generate import Generate

from omegaconf import OmegaConf
import logging
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import parmed as pmd

ds_train = ADPDataset(data_dir='/users/1/sull1276/mint/tests/../mint/data/ADP', 
                       data_proc_fname="AA", 
                       data_proc_ext=".pkl.zst", 
                       data_raw_fname="alanine-dipeptide-250ns-nowater", 
                       data_raw_ext=".xtc", 
                       split="train", 
                       total_frames_train=25600, 
                       total_frames_test=6400, 
                       total_frames_valid=6400, 
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
                       total_frames_train=25600, 
                       total_frames_test=6400, 
                       total_frames_valid=6400, 
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
                       total_frames_train=25600, 
                       total_frames_test=6400, 
                       total_frames_valid=6400, 
                       lag= OmegaConf.create({"equilibrium": True}), 
                       normalize= OmegaConf.create({"bool": True, "t_dependent": False}), 
                       node_features= OmegaConf.create({"epsilon": True, "sigma": True, "charge": True, "mass": True}), 
                       augement_rotations=False)

module = MINTModule(
    cfg=OmegaConf.create({
        "prior": {
            "_target_": "mint.prior.normal.NormalPrior",
            "mean": 0.0,
            "std": 1.0,
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
            "_target_": "mint.model.equivariant.transformer.MultiSE3Transformer",
            "input_channels": [[320], [0]],
            "readout_channels": [[0, 0], [0, 1]],
            "hidden_channels": [[128, 0], [0, 64]],
            "hidden_channels_attn": [[32, 0], [0, 16]],
            "hidden_channels_mlp": [[384, 0], [0, 192]],
            "key_channels":    [[32, 0], [0, 16]],
            "query_channels":  [[32, 0], [0, 16]],
            "edge_l_max": 2,
            "edge_basis": "gaussian",
            "max_radius": 1000,
            "number_of_basis": 128,
            "hidden_size": 64,
            "max_neighbors": 1000,
            "act": "leakyrelu",
            "num_layers": 6,
        },
        "interpolant": {
            "_target_": "mint.interpolant.interpolants.TemporallyLinearInterpolant",
            "velocity_weight": 1.0,
            "denoiser_weight": 1.0,
            "gamma_weight": 0,
        },
        "validation": {
            "stratified": False,
        },
        "optim": {
            "optimizer": {
                "name": "AdamW",
                "lr": 5e-4,
                "weight_decay": 5e-3,
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

ckpt = torch.load("logs/hydra/ckpt/epoch_16-step_13600-loss_-610912.4375.ckpt", map_location="cuda")
module.load_state_dict(ckpt["state_dict"])

st = MINTState(
    seed=42,
    module=module.to('cuda'),
    dataset_train=ds_train,
    dataset_valid=ds_valid,
    dataset_test=ds_test,
)

print(module)

subset = Subset(ds_test, range(64))

test_loader = DataLoader(
    subset,
    batch_size=64,
    shuffle=False,
)

def epsilon_fn(t):
    return t
    
generate_cfg = OmegaConf.create(
    {   "dt": 1e-3,
        "step_type": "ode", # or "sde"
        "clip_val": 1e-3,
        "save_traj": False
    }
)

gen_experiment = Generate(state=st, cfg=generate_cfg, batches = test_loader, epsilon=epsilon_fn)

with torch.no_grad():
    samples = gen_experiment.run()

X = [sample['x'] for sample in samples]
X = torch.stack(X)
print(X.size())

X = [sample['x'] for sample in samples]
X = torch.stack(X)
B, N, C = X.shape              # B = 5, N = 1408, C = 3
nodes = 22

T = (B * N) // nodes           # total number of graphs T
X = X.view(-1, nodes, C)  # shape [T, 22, 3]

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

save_xyz(X,atomic_numbers)