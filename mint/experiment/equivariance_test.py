# management 
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from torch.utils.data import Subset

from mint.experiment.abstract import Experiment
from mint.state import MINTState
from mint.module import EquivariantMINTModule
from mint.data.loader import make_meta_collate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch

class EquivarianceTest(Experiment):
    """
    Generate samples by pushing forward data from the base distribution into the target using a model stored in the state.
    """
    def __init__(self,  state: MINTState, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.state = state
        assert isinstance(state.module, EquivariantMINTModule), f"state.module must be a EquivariantMINTModule, got {type(state.module)}"
    
    def summarize_results(self,results):
        if not results:
            return {}

        bool_dicts  =      [b        for (b,  _   , _    , _    ,_ ,_ ,_) in results]
        max_value_dicts =  [vmax     for (_, vmax ,_     ,_     ,_ ,_ ,_) in results]
        mean_value_dicts = [vmean    for (_, _    ,vmean ,_     ,_ ,_ ,_) in results]
        std_value_dicts =  [vstd     for (_, _    , _    ,vstd  ,_ ,_ ,_) in results]
        norm_b_value_dicts =  [na    for (_, _    , _    ,_     ,na,_ ,_)  in results]
        norm_a_value_dicts =  [nb    for (_, _    , _    ,_     ,_ ,nb,_) in results]
        tolerance_dicts =  [tol      for (_, _    , _    ,_     ,_ , _,tol) in results]

        keys = bool_dicts[0].keys()
        summary = {}

        for k in keys:
            # stack scalar tensors
            bool_vals = torch.stack([d[k] for d in bool_dicts]) 
            max_vals  = torch.stack([d[k] for d in max_value_dicts]) 
            mean_vals = torch.stack([d[k] for d in mean_value_dicts]) 
            std_vals  = torch.stack([d[k] for d in std_value_dicts])
            nb_vals  = torch.stack([d[k] for d in norm_b_value_dicts])
            na_vals  = torch.stack([d[k] for d in norm_a_value_dicts])
            tol_vals  = torch.stack([torch.tensor(d[k]) for d in tolerance_dicts])
            
            all_true = bool_vals.all()  # False if any is False
            max_val  = max_vals.max()   # maximum over all batches
            mean_val = mean_vals.mean() # mean mean over all batches
            min_val  = std_vals.mean()   # mean std over all batche
            nb_val  = nb_vals.mean()   # mean nb over all batche
            na_val  = na_vals.mean()   # mean na over all batches
            tol  = tol_vals.mean()   # mean na over all batches

            summary[k] = {
                "all_true": all_true,
                "max": max_val,
                "mean": mean_val,
                "std": min_val,
                "norm_before": nb_val,
                "norm_after": na_val,
                "tol": tol,
            }

        return summary

    def run(self):

        if self.cfg.split == "test":
            dataset = self.state.dataset_test
        elif self.cfg.split == "train":
            dataset = self.state.dataset_train
        elif self.cfg.split == "valid":
            dataset = self.state.dataset_valid

        subset = Subset(dataset, range(self.cfg.number_of_trials))

        assert self.cfg.batch_size <= self.cfg.number_of_trials, f"cfg.batch_size = {self.cfg.batch_size}, must be less than cfg.number_of_trials = {self.cfg.number_of_trials}"

        loader = DataLoader(
            subset,
            shuffle=False,
            batch_size=self.cfg.batch_size,
            collate_fn = make_meta_collate(dataset.meta_keys)
        )
        self.state.module.eval()
        with torch.no_grad():
            results = []
            for batch in tqdm(loader):
                results.append(
                    self.state.module.test_equivariance(batch, self.cfg.tolerance_dict)  # type: ignore
                )
        self.state.module.train()
        return self.summarize_results(results)