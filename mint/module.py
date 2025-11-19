from omegaconf import DictConfig
from torch import optim
import torch
from pytorch_lightning import LightningModule
import hydra
from mint import utils
from e3nn.o3 import Irreps
from omegaconf import DictConfig
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from typing import Dict,Any
import torch
from e3nn import o3
from copy import copy

class MINTModule(LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.prior = hydra.utils.instantiate(cfg.prior)
        self.embedder = hydra.utils.instantiate(cfg.embedder)
        self.model = hydra.utils.instantiate(cfg.model)
        self.interpolant = hydra.utils.instantiate(cfg.interpolant)

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
        x_t, z = self.interpolant.interpolate(*utils.batch2interp(batch)) # x_t = I(x_0,x_1,t) + \gamma(t)*z
        batch['x_t'] = x_t
        batch['x_t_irrep'] = Irreps("1o")
        batch['feature_keys'].add("x_t")
        batch['z'] = z
        batch['z_irrep'] = Irreps("1o")
        batch['feature_keys'].add("z")
        batch = self.forward(batch)
        loss = self.interpolant.loss(*utils.batch2loss(batch))
        self.log_dict({f"train/{k}": v for k, v in loss.items()}, on_step=True, prog_bar=True, logger=True)
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
        batch['x_t_irrep'] = Irreps("1o")
        batch['feature_keys'].add("x_t")
        batch['z'] = z
        batch['z_irrep'] = Irreps("1o")
        batch['feature_keys'].add("z")
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

class EquivariantMINTModule(MINTModule):

    def __init__(self,cfg: DictConfig) -> None:
        super().__init__(cfg)

    def test_equivariance(self, batch: Data, tolerance_dict: Dict):
        """
        Applies random rotations to the data and forward passes through the model.

        :param batch:
            A torch batch of geometric data objects which come from a data loader. 
        :type batch: torch_geometric.data.Data
        :param tolerance_dict:
            Key value pairs corresponding to the tolerance of a commutator to pass the test.
        :type tolerance_dict: dict
 
        :return:
            A boolean condition on whether or not the model is equivariant
        :rtype: bool
        """
        batch = self.prior.sample(batch)
        x_t, z = self.interpolant.interpolate(*utils.batch2interp(batch))
        batch['x_t'] = x_t
        batch['x_t_irrep'] = Irreps("1o")
        batch['feature_keys'].add("x_t")
        batch['z'] = z
        batch['z_irrep'] = Irreps("1o")
        batch['feature_keys'].add("z")

        # sample a set of random rotations for the given batch
        R =self.random_group_action(batch)

        # rotate the batch
        roated_batch= self.apply_group_action(R,batch)

        out_before = self.forward(roated_batch)
        out_after = self.apply_group_action(R, self.forward(batch))
        
        max_errors,mean_errors,min_errors,norm_before,norm_after = self.commutator_error(out_before, out_after)
        passes_test = dict()

        for key in batch['feature_keys']:
            tol = tolerance_dict[key]
            passes_test[key] = max_errors[key] < tol

        return passes_test,max_errors,mean_errors,min_errors,norm_before,norm_after,tolerance_dict

    def commutator_error(self, out_before: Data, out_after: Data):
        """
        Returns dictionary of key value pairs which maps a key to its commutator error between a batch out_before which was
        acted on by apply_group_action before .forward() and out_after which was acted on after.

        
        :param out_before:
            A torch batch of geometric data objects which was rotated before forward.
        :type batch: torch_geometric.data.Data
        :param out_after:
            A torch batch of geometric data objects which was rotated after forward.
        :type batch: torch_geometric.data.Data

        :return:
            A dictionary of max absolute value of error values.
        :rtype: dict
        """
        max_errors = dict()
        mean_errors = dict()
        std_errors = dict()
        norm_before = dict()
        norm_after = dict()
        for key in out_before['feature_keys']:
            max_errors[key] = torch.max(abs(out_before[key] - out_after[key]))
            mean_errors[key] = torch.mean(abs(out_before[key] - out_after[key]))
            std_errors[key] = torch.std(abs(out_before[key] - out_after[key]))
            norm_before[key] = torch.mean(torch.norm(out_before[key],dim=-1))
            norm_after[key] = torch.mean(torch.norm(out_after[key],dim=-1))
        return max_errors, mean_errors ,std_errors,norm_before,norm_after
    
    def random_group_action(self, batch: Data) -> Any:
        """
        Samples a random group action for the system in question.

        :param batch:
            A torch batch of geometric data objects which come from a data loader. 
        :type batch: torch_geometric.data.Data

        :return:
            An object specifying the group action. 
        :rtype: dict
        """
        size = max(batch['batch']+1)
        R = o3.rand_matrix(size)
        return R
    
    def apply_group_action(self, R: Any, batch: Data) -> Data:
        """"
        Applies a group action R to the batch. It does so by taking advantage of the get_irreps method.

        :param R:
            An object specifying the group action. 
        :type R: Any
        :param batch:
            A torch batch of geometric data objects which come from a data loader. 
        :type batch: torch_geometric.data.Data

        :return:
            A torch batch of geometric data objects which have been acted on by the group action R.
        :type batch: torch_geometric.data.Data
        """
        rotated_batch = copy(batch)
        for key in batch['feature_keys']:
            irrep : o3.Irreps = batch[key+"_irrep"]
            D = irrep.D_from_matrix(R)
            x_dense, mask =to_dense_batch(batch[key],batch['batch'])
            rotated_batch[key] = torch.einsum('Bij,BNj->BNi',D,x_dense)[mask]

        return rotated_batch