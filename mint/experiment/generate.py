# management 
from omegaconf import DictConfig, OmegaConf

from mint.experiment.abstract import Experiment

class Generate(Experiment):
    """
    
    """
    def __init__(self, cfg: DictConfig, logger) -> None:
        # Split configuration up

        super().__init__(cfg)


    def run(self):
        pass