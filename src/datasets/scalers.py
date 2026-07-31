import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

from src.registry import Registry

class BaseScaler(ABC):
    """
    Abstract interface for Coordinate Scalers.
    Scalers handle feature-wise normalization of encoded arrays.
    """
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Computes necessary statistics (mean, std, min, max, etc.) from the data."""
        pass
        
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Applies scaling to the data."""
        pass
        
    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverses the scaling operation."""
        pass
        
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns scaler parameters for versioning and logging."""
        pass

@Registry.register_scaler("identity")
class IdentityScaler(BaseScaler):
    """A passthrough scaler that applies no normalization."""
    
    def fit(self, data: np.ndarray) -> None:
        pass
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        return data.copy()
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data.copy()
        
    def get_metadata(self) -> Dict[str, Any]:
        return {"scaler_type": "identity"}

@Registry.register_scaler("standard")
class StandardScaler(BaseScaler):
    """Zero-mean, unit-variance scaler. Ignores NaN values if present."""
    
    def __init__(self, eps: float = 1e-8):
        self.mean: np.ndarray = None
        self.std: np.ndarray = None
        self.eps = eps
        
    def fit(self, data: np.ndarray) -> None:
        if data.size == 0:
            return
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0) + self.eps
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler must be fitted before transform.")
        if data.size == 0:
            return data.copy()
        return (data - self.mean) / self.std
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler must be fitted before inverse_transform.")
        if data.size == 0:
            return data.copy()
        return (data * self.std) + self.mean
        
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "scaler_type": "standard",
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "eps": self.eps
        }

@Registry.register_scaler("minmax")
class MinMaxScaler(BaseScaler):
    """Scales data to a fixed range (typically 0, 1 or -1, 1)."""
    
    def __init__(self, feature_range: tuple = (-1, 1), eps: float = 1e-8):
        self.min_val: np.ndarray = None
        self.max_val: np.ndarray = None
        self.range_min, self.range_max = feature_range
        self.eps = eps
        
    def fit(self, data: np.ndarray) -> None:
        if data.size == 0:
            return
        self.min_val = np.nanmin(data, axis=0)
        self.max_val = np.nanmax(data, axis=0)
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_val is None or self.max_val is None:
            raise ValueError("MinMaxScaler must be fitted before transform.")
        if data.size == 0:
            return data.copy()
            
        data_range = np.maximum(self.max_val - self.min_val, self.eps)
        normalized = (data - self.min_val) / data_range
        return normalized * (self.range_max - self.range_min) + self.range_min
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_val is None or self.max_val is None:
            raise ValueError("MinMaxScaler must be fitted before inverse_transform.")
        if data.size == 0:
            return data.copy()
            
        data_range = np.maximum(self.max_val - self.min_val, self.eps)
        normalized = (data - self.range_min) / (self.range_max - self.range_min)
        return (normalized * data_range) + self.min_val
        
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "scaler_type": "minmax",
            "feature_range": [self.range_min, self.range_max],
            "min_val": self.min_val.tolist() if self.min_val is not None else None,
            "max_val": self.max_val.tolist() if self.max_val is not None else None,
            "eps": self.eps
        }
