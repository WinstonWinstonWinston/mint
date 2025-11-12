from omegaconf import DictConfig
from torch import optim
import torch
from pytorch_lightning import LightningModule
import hydra
from mint import utils
from e3nn.o3 import Irreps

class MINTModule(LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.prior = hydra.utils.instantiate(cfg.prior)
        self.embedder = hydra.utils.instantiate(cfg.embedder)
        self.model = hydra.utils.instantiate(cfg.model)
        self.interpolant = hydra.utils.instantiate(cfg.interpolant)
        # self.experiment = hydra.utils.instantiate(self.cfg.experiment)x``

        self.save_hyperparameters()

    def forward(self, batch):
        """
        TODO: Finish return param typing here
        Implements a forward pass through the embedders and model.

        :param batch:
            A torch batch of geometric data objects which come from a data loader. 
        :type batch: torch_geometric.data.Data

        :return:
            A new batch object with modified keys containing velocity, score, denoised point etc.
        :rtype: torch_geometric.data.Data??
        """
        f = self.embedder.forward(batch)
        f = self.model.forward(f)
        return f

    def configure_optimizers(self):
        """
        Parses configuration for the optimizer for lightning 

        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers

        """
        ocfg = self.cfg.optim["optimizer"]
        opt_cls = getattr(optim, ocfg["name"])
        optimizer = opt_cls(self.parameters(), **{k:v for k,v in ocfg.items() if k!="name"})

        # optional scheduler
        if "scheduler" in ocfg:
            scfg = self.cfg.optim["scheduler"]
            sch_cls = getattr(optim.lr_scheduler, scfg["name"])
            scheduler = sch_cls(optimizer, **{k:v for k,v in scfg.items() if k not in ("name","monitor")})
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scfg.get("monitor", "val/loss"),
                },
            }
        return optimizer

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        """
        Implements a training step.

            1) corrupt batch appropriately using interpolant
            2) call forward
            3) compute loss

        :param batch:
            A torch batch of geometric data objects which come from a data loader. 
        :type batch: torch_geometric.data.Data

        :return:
            A dictionary of loss values, loss, loss_velocity, and loss_denoiser
        :rtype: dict[str, torch.Tensor]
        """
        batch = self.prior.sample(batch)
        x_t, z = self.interpolant.interpolate(*utils.batch2interp(batch))
        batch['x_t'] = x_t
        batch['x_t_irrep'] = Irreps("1o"),
        batch['z'] = z
        batch['z_irrep'] = Irreps("1o")
        batch = self.forward(batch)
        loss = self.interpolant.loss(*utils.batch2loss(batch))
        self.log_dict({f"train/{k}": v for k, v in loss.items()},on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        """
        Implements a validation step. 

            1) corrupt batch appropriately using interpolant
                1a) do so stratified on [0,1] 
            2) call forward
            3) compute loss

        :param batch:
            A torch batch of geometric data objects which come from a data loader. 
        :type batch: torch_geometric.data.Datax

        :return:
            A dictionary of loss values, loss, loss_velocity, and loss_denoiser
        :rtype: dict[str, torch.Tensor]
        """
        batch = self.prior.sample(batch, stratified=self.cfg.validation.stratified)
        x_t, z = self.interpolant.interpolate(*utils.batch2interp(batch))
        batch['x_t'] = x_t
        batch['x_t_irrep'] = Irreps("1o"),
        batch['z'] = z
        batch['z_irrep'] = Irreps("1o")
        batch = self.forward(batch)
        # TODO: Fix interpolant to have stratified flag
        loss = self.interpolant.loss(*utils.batch2loss(batch,stratified=self.cfg.validation.stratified))
        self.log_dict({f"val/{k}": v for k, v in loss.items()},on_step=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch) -> None:
        """
        Use the batch of data to perform experiments on the model based off of config

        1) parse experiments from config and instantiate experiment objects
        2) prepare model for experiment (disable dropout, training depedent layers, etc. )
        3) run experiment
        4) go back to 2

        :param batch:
            A torch batch of geometric data objects which come from a data loader. 
        :type batch: torch_geometric.data.Data
        """
        # raise NotImplementedError
        pass
        # self.experiment.run(batch)