from mint.state import MINTState
from mint.data.ADP.ADP_dataset import ADPDataset
from mint.module import EquivariantMINTModule
from mint.experiment.train import Train
from mint.experiment.equivariance_test import EquivarianceTest

from omegaconf import OmegaConf
import logging
from pytorch_lightning.utilities.rank_zero import rank_zero_only

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

max_epochs = 200

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
            "_target_": "mint.model.equivariant.PaINNLike.PaiNNLikeInterpolantNet",
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
            "mlp_drop": 0.2,
            "conv_weight_layers": [192],
            "update_weight_layers": [128],
            "message_update_count_cond": 4,
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
                "lr": 1e-2,
            },
            "scheduler": {
                "name": "LinearLR",
                "start_factor": 1.0,
                "end_factor": 0.1,
                "total_iters": max_epochs,
            },
        },
    })
)

print(module)
    
st = MINTState(
    seed=42,
    module=module,
    dataset_train=ds_train,
    dataset_valid=ds_valid,
    dataset_test=ds_test,
)


eqv_test_cfg = OmegaConf.create({"split":"train",
                                 "batch_size":5,
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
                                                    "f": 1e-6,
                                                    "f_cond":1e-6,
                                                    "b":1e-6,
                                                    "eta":1e-6},
                                 

                })


eqv_test = EquivarianceTest(st,eqv_test_cfg)
results =eqv_test.run()

print(results)

# train_cfg = OmegaConf.create({
#     "trainer": {
#         "overfit_batches": 0,
#         "min_epochs": 1,
#         "max_epochs": max_epochs,
#         "accelerator": "gpu",
#         "log_every_n_steps": 10,
#         "deterministic": False,
#         # "strategy": "ddp_notebook",
#         "val_check_interval": 1.0,
#         "check_val_every_n_epoch": 1,
#         "accumulate_grad_batches": 1,
#         "gradient_clip_val": 1.0,
#         "gradient_clip_algorithm": "norm",
#         "precision": "32-true",
#     },
#     "checkpointer": {
#         "dirpath": "/users/1/sull1276/mint/tests/logs/hydra/ckpt",
#         "save_last": True,
#         "save_top_k": 5,
#         "monitor": "val/loss",
#         "filename": "epoch_{epoch}-step_{step}-loss_{val/loss:.4f}",
#         "auto_insert_metric_name": False,
#         "mode": "min",
#     },
#     "wandb": {
#         "name": "mint",
#         "project": "mint",
#         "save_dir": "/users/1/sull1276/mint/tests/logs/wandb",
#     },
#     "wandb_watch": {
#         "log": "all",
#         "log_freq": 500,
#     },
#     "warm_start": None,
#     "warm_start_cfg_override": True,
#     "loader": {
#         "num_workers": 8,
#         "prefetch_factor": 2,
#         "batch_size": {
#             "train": 32,
#             "valid": 32,
#             "test": 32
#         },
#     },
#     "num_device": 1,
#     "project": {"name": "mint"}
# })

# logger = logging.getLogger(__name__)
# logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
# for level in logging_levels:
#     setattr(logger, level, rank_zero_only(getattr(logger, level)))
    
# trainer = Train(st, train_cfg, logger)

# trainer.run()