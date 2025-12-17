# management 
from omegaconf import DictConfig
from tqdm.auto import tqdm

from mint.experiment.abstract import Experiment
from mint.state import MINTState

class Generate(Experiment):
    """
    Generate samples by pushing forward data from the base distribution into the target using a model stored in the state.
    """
    def __init__(self,  state: MINTState, cfg: DictConfig, batches, epsilon=lambda t: t) -> None:
        super().__init__()

        self.generate_cfg = cfg
        self.state = state
        self.epsilon = epsilon
        self.batches = batches

    def run(self):
        samples =[]
        for batch in tqdm(self.batches, desc="Generating samples", position=0, dynamic_ncols=True):
            # Sample from the base distribution
            batch = batch.to(self.state.module.device)
            batch = self.state.module.prior.sample(batch, stratified=False)
            # Overwrite state with base sample
            batch['x_target'] = batch['x']
            batch['x_t'] = batch['x_base']
            # Forward integrate batch
            integrated_batch = self.state.module.interpolant.integrate(batch,
                                                        self.state.module,
                                                        self.generate_cfg.dt,
                                                        self.generate_cfg.step_type,
                                                        self.generate_cfg.clip_val,
                                                        self.generate_cfg.save_traj,
                                                        self.epsilon,
                                                        self.generate_cfg.b_anneal_factor)
            # Save result
            samples.append(integrated_batch)
        return samples