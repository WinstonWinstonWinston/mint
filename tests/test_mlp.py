from mint.state import MINTState
from mint.data.ADP.ADP_dataset import ADPDataset
from mint.module import EquivariantMINTModule
from mint.experiment.train import Train
from mint.experiment.equivariance_test import EquivarianceTest
import torch
from omegaconf import OmegaConf
import logging
from pytorch_lightning.utilities.rank_zero import rank_zero_only

#!/bin/bash
#SBATCH -J mint-test
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -p v100                  # same partition
#SBATCH -t 08:00:00              # same time limit
#SBATCH -N 1                     # same number of nodes
#SBATCH --ntasks-per-node=1      # same tasks per node
#SBATCH --cpus-per-task=1        # matches srun default
#SBATCH --mem-per-cpu=20G        # same memory per CPU
#SBATCH --gres=gpu:1             # same GPU request

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
                "lr": 5e-3,
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

ckpt = torch.load("logs/hydra/ckpt_MLP_1/epoch_199-step_40000-loss_-13056556.0000.ckpt", map_location="cuda")
module.load_state_dict(ckpt["state_dict"])

print(module)
    
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


train_cfg = OmegaConf.create({
    "trainer": {
        "overfit_batches": 0,
        "min_epochs": 1,
        "max_epochs": max_epochs,
        "accelerator": "gpu",
        "log_every_n_steps": 10,
        "deterministic": False,
        # "strategy": "ddp_notebook",
        "val_check_interval": 1.0,
        "check_val_every_n_epoch": 1,
        "accumulate_grad_batches": 1,
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "norm",
        "precision": "32-true",
    },
    "checkpointer": {
        "dirpath": "/users/1/sull1276/mint/tests/logs/hydra/ckpt",
        "save_last": True,
        "save_top_k": 5,
        "monitor": "val/loss",
        "filename": "epoch_{epoch}-step_{step}-loss_{val/loss:.4f}",
        "auto_insert_metric_name": False,
        "mode": "min",
    },
    "wandb": {
        "name": "mint",
        "project": "mint",
        "save_dir": "/users/1/sull1276/mint/tests/logs/wandb",
    },
    "wandb_watch": {
        "log": "all",
        "log_freq": 500,
    },
    "warm_start": None,
    "warm_start_cfg_override": True,
    "loader": {
        "num_workers": 8,
        "prefetch_factor": 2,
        "batch_size": {
            "train": 128,
            "valid": 128,
            "test": 128
        },
    },
    "num_device": 1,
    "project": {"name": "mint"}
})

logger = logging.getLogger(__name__)

logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
for level in logging_levels:
    setattr(logger, level, rank_zero_only(getattr(logger, level)))
    
trainer = Train(st, train_cfg, logger)
trainer.run()

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