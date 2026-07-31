from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseModelInterface(ABC):
    """
    Abstract interface for all generative AI architectures in this framework.
    """
    
    @abstractmethod
    def train_model(self, dataset: Any, config: Dict[str, Any]) -> None:
        """Trains the model on the provided dataset."""
        pass
        
    @abstractmethod
    def generate(self, text: str, style_id: str = None) -> Any:
        """Generates a coordinate representation from text."""
        pass
        
    @abstractmethod
    def evaluate(self, dataset: Any) -> Dict[str, float]:
        """Evaluates the model and returns metrics."""
        pass
        
    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the model checkpoint."""
        pass
        
    @abstractmethod
    def load(self, path: str) -> None:
        """Loads a model from a checkpoint."""
        pass
