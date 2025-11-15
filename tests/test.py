from mint.state import MINTState
from mint.data.ADP.ADP_dataset import ADPDataset
from mint.module import MINTModule
from mint.experiment.train import Train

from omegaconf import OmegaConf
import logging
from pytorch_lightning.utilities.rank_zero import rank_zero_only

ds_train = ADPDataset(data_dir='/users/1/sull1276/mint/tests/../mint/data/ADP', 
                       data_proc_fname="AA", 
                       data_proc_ext=".pkl.zst", 
                       data_raw_fname="alanine-dipeptide-250ns-nowater", 
                       data_raw_ext=".xtc", 
                       split="train", 
                       total_frames_train=25000, 
                       total_frames_test=5000, 
                       total_frames_valid=5000, 
                       lag= OmegaConf.create({"equilibrium": True}), 
                       normalize= OmegaConf.create({"bool": False, "t_dependent": False}), 
                       node_features= OmegaConf.create({"epsilon": True, "sigma": True, "charge": True, "mass": True}), 
                       augement_rotations=False)

ds_test = ADPDataset(data_dir='/users/1/sull1276/mint/tests/../mint/data/ADP', 
                       data_proc_fname="AA", 
                       data_proc_ext=".pkl.zst", 
                       data_raw_fname="alanine-dipeptide-250ns-nowater", 
                       data_raw_ext=".xtc", 
                       split="test", 
                       total_frames_train=25000, 
                       total_frames_test=5000, 
                       total_frames_valid=5000, 
                       lag= OmegaConf.create({"equilibrium": True}), 
                       normalize= OmegaConf.create({"bool": False, "t_dependent": False}), 
                       node_features= OmegaConf.create({"epsilon": True, "sigma": True, "charge": True, "mass": True}), 
                       augement_rotations=False)

ds_valid = ADPDataset(data_dir='/users/1/sull1276/mint/tests/../mint/data/ADP', 
                       data_proc_fname="AA", 
                       data_proc_ext=".pkl.zst", 
                       data_raw_fname="alanine-dipeptide-250ns-nowater", 
                       data_raw_ext=".xtc", 
                       split="valid", 
                       total_frames_train=25000, 
                       total_frames_test=5000, 
                       total_frames_valid=5000, 
                       lag= OmegaConf.create({"equilibrium": True}), 
                       normalize= OmegaConf.create({"bool": False, "t_dependent": False}), 
                       node_features= OmegaConf.create({"epsilon": True, "sigma": True, "charge": True, "mass": True}), 
                       augement_rotations=False)
module = MINTModule(
    cfg=OmegaConf.create({
        "prior": {
            "_target_": "mint.prior.normal.NormalPrior",
            "mean": 0.0,
            "std": 0.25,
        },
        "embedder": {
            "_target_": "mint.model.embedding.equilibrium_embedder.EquilibriumEmbedder",
            "use_ff": True,
            "interp_time": {
                "embedding_dim": 64,
                "max_positions": 1000,
            },
            "force_field": {
                "in_dim": 4,
                "hidden_dims": [128, 64],
                "out_dim": 32,
                "activation": "relu",
                "use_input_bn": False,
                "affine": False,
                "track_running_stats": True,
            },
            "atom_type": {
                "num_types": 14,
                "embedding_dim": 32,
            },
        },
        "model": {
            "_target_": "mint.model.equivariant.transformer.MultiSE3Transformer",
            "input_channels": [[128], [0]],
            "readout_channels": [[0, 0], [0, 1]],
            "hidden_channels": [[8, 8], [8, 8]],
            "key_channels": [[8, 8], [8, 8]],
            "query_channels": [[8, 8], [8, 8]],
            "edge_l_max": 2,
            "edge_basis": "smooth_finite",
            "max_radius": 10,
            "number_of_basis": 64,
            "hidden_size": 128,
            "max_neighbors": 10000,
            "act": "silu",
            "num_layers": 4,
            "bn": False,
        },
        "interpolant": {
            "_target_": "mint.interpolant.interpolants.TemporallyLinearInterpolant",
            "velocity_weight": 1.0,
            "denoiser_weight": 1.0,
            "gamma_weight": 0.1,
        },
        "validation": {
            "stratified": False,
        },
        "optim": {
            "optimizer": {
                "name": "Adam",
                "lr": 3e-4,
                "weight_decay": 0.01,
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
    
st = MINTState(
    seed=42,
    module=module,
    dataset_train=ds_train,
    dataset_valid=ds_valid,
    dataset_test=ds_test,
)

train_cfg = OmegaConf.create({
    "trainer": {
        "overfit_batches": 0,
        "min_epochs": 1,
        "max_epochs": 200,
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
            "train": 64,
            "valid": 64,
            "test": 64,
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