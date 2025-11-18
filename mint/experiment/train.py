import os
import GPUtil
from omegaconf import DictConfig

# lightning
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only # type: ignore
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# mint
from mint.experiment.abstract import Experiment
from mint.state import MINTState
from mint.data.loader import MINTDatamodule

class Train(Experiment):
    """
    TODO: Comment me
    """
    def __init__(self, state: MINTState, train_cfg: DictConfig, logger) -> None:
        # Split configuration up

        super().__init__()

        self.train_cfg = train_cfg
        self.loader_cfg = self.train_cfg.loader
        self.train_cfg.pop('loader')
        self.state = state

        self.datamodule =  MINTDatamodule(**self.loader_cfg,
                                            train_dataset = state.dataset_train, 
                                            valid_dataset = state.dataset_valid, 
                                            test_dataset  = state.dataset_test)
        
        

        # Determine available gpus
        self.train_device_ids = GPUtil.getAvailable(order='memory', limit = 8)[:self.train_cfg.num_device]
        logger.info(f"Training with devices: {self.train_device_ids}")

        self.logger = logger

    def run(self):
        callbacks = []
       
        # Setup lightning wandb connection
        wbLogger = WandbLogger(
            **self.train_cfg.wandb,
        )
        
        wbLogger.watch(
            self.state.module,
            log=self.train_cfg.wandb_watch.log,
            log_freq=self.train_cfg.wandb_watch.log_freq
        )

        # Checkpoint directory.
        ckpt_dir = self.train_cfg.checkpointer.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        self.logger.info(f"Checkpoints saved to {ckpt_dir}")
        
        # Model checkpoints
        callbacks.append(ModelCheckpoint(**self.train_cfg.checkpointer))

        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        
        trainer = Trainer(
            **self.train_cfg.trainer,
            callbacks=callbacks,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=1 if self.train_cfg.trainer.accelerator == "cpu" else self.train_device_ids,
            logger=wbLogger,
            # detect_anomaly=True
        )

        trainer.fit(
            model=self.state.module,
            datamodule=self.datamodule,
            ckpt_path=self.train_cfg.warm_start
        )