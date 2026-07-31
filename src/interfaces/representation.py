from abc import ABC, abstractmethod
from typing import Any, Dict, List
from src.datasets.structures import TrajectorySample

class CoordinateRepresentation(ABC):
    """
    Abstract interface for coordinate representation (Continuous, Discrete, etc.).
    """
    
    @abstractmethod
    def encode(self, trajectory: TrajectorySample) -> Any:
        """Encodes a Trajectory object into the target representation."""
        pass
        
    @abstractmethod
    def decode(self, representation: Any) -> TrajectorySample:
        """Decodes the representation back into a Trajectory object."""
        pass
        
    @abstractmethod
    def statistics(self) -> Dict[str, Any]:
        """Returns statistics about the representation (e.g., vocab size, bounds)."""
        pass
