from mint.state import MINTState
from mint.data.ADP.ADP_dataset import ADPDataset
from mint.module import EquivariantMINTModule
from mint.experiment.equivariance_test import EquivarianceTest
from mint.experiment.generate import Generate
from torch.utils.data import DataLoader
from mint.data.loader import make_meta_collate
from omegaconf import OmegaConf
from torch.utils.data import Subset
import torch
import parmed as pmd
from torch_geometric.utils import to_dense_batch

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

max_epochs = 200

module = EquivariantMINTModule(
    cfg=OmegaConf.create({
        "prior": {
            "_target_": "mint.prior.normal.NormalPrior",
            "mean": 0.0,
            "std": 0.5,
            "antithetic":True,
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
            },
            "atom_type": {
                "num_types": 14,
                "embedding_dim": 32,
            },
        },
        "model": {
             "_target_": "mint.model.MLP.MINTMLP",
             "in_dim": 22*320+22*3,
             "hidden_dims": (4096, 1024),
             "out_dim": 22*3,
             "activation": "silu",
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

ckpt = torch.load("logs/hydra/ckpt/epoch_19-step_4000-loss_-1.3269.ckpt", map_location="cuda")
module.load_state_dict(ckpt["state_dict"])

    
st = MINTState(
    seed=42,
    module=module,
    dataset_train=ds_train,
    dataset_valid=ds_valid,
    dataset_test=ds_test,
)

eqv_test_cfg = OmegaConf.create({"split":"train",
                                 "batch_size": 3,
                                 "number_of_trials":5,
                                 "tolerance_dict": {"x": 1e-6,
                                                    "charge":1e-6,
                                                    "atom_type":1e-6,
                                                    "mass":1e-6,
                                                    "sigma":1e-6,
                                                    "epsilon":1e-6,
                                                    "x_base":1e-6,
                                                    "t_interpolant":1e-6,
                                                    "x_t":1e-6,
                                                    "z":1e-6,
                                                    "f": 1e-3,
                                                    "f_cond":1e-3,
                                                    "b":1e-3,
                                                    "eta":1e-3},
                                 

                })

eqv_test = EquivarianceTest(st, eqv_test_cfg)
results = eqv_test.run()

row_fmt = "{:<15} {:<6} {:>10} {:>10} {:>10} {:>10} {:>14} {:>14}"

# header
print(row_fmt.format(
    "name", "status", "mean", "std", "max", "tol", "norm_before", "norm_after"
))
print("-" * 96)

# rows
for k, v in sorted(results.items()):
    mean = f"{v['mean'].item():.4g}"
    std = f"{v['std'].item():.4g}"
    max_ = f"{v['max'].item():.4g}"
    tol = f"{v['tol'].item():.4g}"
    nb = f"{v['norm_before'].item():.4g}"
    na = f"{v['norm_after'].item():.4g}"
    status = "OK" if v["all_true"].item() else "FAIL"

    print(row_fmt.format(k, status, mean, std, max_, tol, nb, na))


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
    {   "dt": 1e-3,
        "step_type": "ode", # or "sde"
        "clip_val": 1e-3,
        "save_traj": False
    }
)

gen_experiment = Generate(state=st, cfg=generate_cfg, batches = loader, epsilon=epsilon_fn)

with torch.no_grad():
    samples = gen_experiment.run()

X = [sample['x'] for sample in samples]

for b in loader:
    batch = b['batch']
    break

sample = X[0]
sample_dense, mask = to_dense_batch(sample,batch)

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

save_xyz(sample_dense,atomic_numbers)