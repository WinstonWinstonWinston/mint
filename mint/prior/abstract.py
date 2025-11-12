from abc import ABC, abstractmethod
from typing import Any
from torch import Tensor
from torch_geometric.data import Data


class MINTPrior(ABC):
    """
    Abstract interface for prior samplers used in MINT workflows.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def sample(self, batch: Data, stratified: bool) -> Data:
        """
        Draw samples from the prior and return the same batch object type with updated keys.

        :param batch:
            A torch batch of geometric data objects coming from a data loader.
        :type batch: torch_geometric.data.Data
        :param stratified:
            Whether to use stratified sampling over the time variable
        :type stratified: bool

        :return:
            The input batch type with modified keys (e.g., velocity, score, denoised point).
        :rtype: torch_geometric.data.Data
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError