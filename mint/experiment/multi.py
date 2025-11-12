

from omegaconf import DictConfig
from abc import ABC, abstractmethod
import hydra

class Experiment(ABC):
    """
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.experiments = []
        for element in cfg.elements:
            self.experiments.append(hydra.utils.instantiate(element))

    def run(self):
        
        for experiment in self.experiments:
            experiment.run()
        raise NotImplementedError