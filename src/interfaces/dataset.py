from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseDataset(ABC):
    """
    Abstract interface for Dataset implementations.
    """
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Loads the dataset into memory or sets up streaming."""
        pass
        
    @abstractmethod
    def validate(self) -> bool:
        """Validates the dataset integrity (checksums, schema)."""
        pass
        
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """Runs statistical analysis and returns metrics."""
        pass
