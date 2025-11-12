import torch
from omegaconf import DictConfig,OmegaConf
import os
import logging
import hydra
from pytorch_lightning.utilities.rank_zero import rank_zero_only #type: ignore
from mint.experiment.train import Train

logger = logging.getLogger(__name__)
logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
for level in logging_levels:
    setattr(logger, level, rank_zero_only(getattr(logger, level)))

@hydra.main(config_path="configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    torch.autograd.set_detect_anomaly(True)
    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.exp_train.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        logger.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Train(cfg,logger)
    exp.run()

if __name__ == "__main__":
    main()