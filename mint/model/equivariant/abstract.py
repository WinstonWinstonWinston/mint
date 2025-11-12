from omegaconf import DictConfig
from mint.module import MINTModule
from torch_geometric.data import Data
from typing import Dict,Any
import torch
from e3nn import o3
from copy import copy

class EquivariantMINTModule(MINTModule):

    def __init__(self,cfg: DictConfig) -> None:
        super().__init__(cfg)

    def test_equivariance(self,batch: Data, tolerance_dict: Dict) -> Dict[str, bool]:
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
        # sample a set of random rotations for the given batch
        R =self.random_group_action(batch)

        # rotate the batch
        roated_batch= self.apply_group_action(R,batch)
        out_before = self.forward(roated_batch)
        out_after = self.apply_group_action(R, self.forward(batch))
        
        errors = self.commutator_error(out_before, out_after)
        passes_test = dict()

        for key in batch['keys']:
            tol = tolerance_dict[key]
            passes_test[key] = errors[key] > tol

        return passes_test

    def commutator_error(self, out_before: Data, out_after: Data) -> Dict[str, float]:
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
            A dictionary of mean absolute value of error values.
        :rtype: dict
        """
        errors = dict()
        for key in out_before['keys']:
            errors[key] = torch.mean(abs(out_before[key] - out_after[key]))
        return errors
    
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
        size = max(batch['batch'])
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
        for key in batch['keys']:
            irrep : o3.Irreps = batch[key+"_irrep"]
            D = irrep.D_from_matrix(R)
            rotated_batch[key] = torch.einsum('BNij,BNj->BNi',D,batch[key])
        
        return rotated_batch