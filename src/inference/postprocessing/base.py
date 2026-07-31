from abc import ABC, abstractmethod
from src.datasets.structures import TrajectorySample
from src.inference.config import InferenceConfig

class BasePostProcessor(ABC):
    """
    Contract for pipeline post-processing plugins.
    Takes a TrajectorySample, mutates it or returns a new instance, and passes it forward.
    """
    def __init__(self, config: InferenceConfig):
        self.config = config
        
    @abstractmethod
    def process(self, sample: TrajectorySample) -> TrajectorySample:
        """Modifies or validates the trajectory sample."""
        pass
