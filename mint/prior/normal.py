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
    """

    def __init__(self, mean:float, std: float, antithetic = False) -> None:
        super().__init__()
        self.std = std
        self.mean = mean
        self.antithetic = antithetic

    def sample(self, batch: Data, stratified: bool = False) -> Data:
        """
        Draw samples from the prior and return the same batch object type with updated keys.

        :param batch:
            A torch batch of geometric data objects coming from a data loader.
        :type batch: torch_geometric.data.Data
        :param stratified:
            Whether to use stratified sampling over the time variable
        :type stratified: bool

        :return:
            The input batch type with modified keys.
        :rtype: torch_geometric.data.Data
        """
        x_base = torch.randn_like(batch['x'])*self.std + self.mean
        B = max(batch['batch']) + 1 # shift by 1, zero indexing
        device = batch['x'].device
        if self.antithetic:
            if not stratified:
                # number of independent t's we need (each generates a pair t, 1 - t)
                n_pairs = (B + 1) // 2
                t_base = torch.rand(n_pairs, device=device)
            else:
                n_pairs = (B + 1) // 2
                # stratified sampling in [0, 1] for n_pairs values
                t_base = torch.cat([
                    (i + torch.rand(n_pairs // 4 + (i < n_pairs % 4), device=device)) / 4
                    for i in range(4)
                ])[:n_pairs]

            # make pairs (t, 1 - t)
            t_interpolant = torch.cat([t_base, 1.0 - t_base], dim=0)[:B]

        else:
            t_interpolant = torch.rand(B, device=device) if not stratified else torch.cat([(i + torch.rand(B//4 + (i < B%4), device=device))/4 for i in range(4)])


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