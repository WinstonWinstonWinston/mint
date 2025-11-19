from torch import Tensor
from torch_geometric.data import Data
from mint.prior.abstract import MINTPrior
from e3nn.o3 import Irreps
import torch
import math

class NormalPrior(MINTPrior):
    """
    Abstract interface for prior samplers used in MINT workflows.

    :param cfg:
        Configuration object (dataclass/dict/omegaconf) holding prior hyperparameters.
    :type cfg: Any

    ANTITHETIC BROKEN NUMNUTZ IDIOT DUMBASS THAT IS NOOOOT WHAT IT IS WHAT WERE U THINKING
    """ 

    def __init__(self, mean:float, std: float, antithetic = False) -> None:
        super().__init__()
        self.std = std
        self.mean = mean
        self.antithetic = antithetic

    def sample(self, batch: Data, stratified: bool = False) -> Data:
        """
        Draw samples from the prior and return the same batch object type with updated keys.
        """
        x_base = torch.randn_like(batch['x'])*self.std + self.mean
        B = max(batch['batch']) + 1 # shift by 1, zero indexing
        device = batch['x'].device
        t_interpolant = torch.rand(B, device=device)

        batch['x_base'] = x_base
        batch['x_base_irrep'] = Irreps("1o")
        batch['feature_keys'].add("x_base")
        batch['t_interpolant'] = t_interpolant
        return batch

    def log_prob(self, batch: Data) -> Tensor:
        """
        Compute log-probabilities under the prior for variables referenced by batch.

        :param batch:
            A torch batch of geometric data objects coming from a data loader.
        :type batch: torch_geometric.data.Data

        :return:
            Per-example log p(x) with a leading batch dimension.
        :rtype: torch.Tensor
        """
        x = batch["x_base"] # (B*N, 3)
        B = batch.num_graphs
        z = (x - self.mean) / self.std
        z = z.view(B, -1, 3) # requires contiguous nodes per graph
        B, N, D = z.shape
        return -0.5*torch.sum(z*z, dim=(1,2)) - 0.5*N*D*math.log(2*math.pi) - N*D*math.log(float(self.std))