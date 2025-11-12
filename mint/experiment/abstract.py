from abc import ABC, abstractmethod

class Experiment(ABC):
    """
    Abstract base experiment class. 
    
    All experiments must
    1: Save their configuration
    2: Contain a "run"
    3: Be able to give a descriptive print statement describing the config
    

    TODO: Comment me
    """

    @abstractmethod
    def run(self):
        raise NotImplementedError