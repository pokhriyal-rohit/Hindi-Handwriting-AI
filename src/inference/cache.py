import copy
from typing import Dict, Any, Optional
from src.datasets.structures import TrajectorySample

class InferenceCache:
    """
    Multi-level memoization cache for the InferenceSession.
    Prevents redundant model executions for repeated identical inputs.
    """
    def __init__(self):
        # Level 1: Normalized Text -> Raw Tokens
        self.text_cache: Dict[str, list[int]] = {}
        
        # Level 2: Tokens Hash -> Raw Prediction Arrays
        self.prediction_cache: Dict[str, list[list[float]]] = {}
        
        # Level 3: Normalized Text -> TrajectorySample (Bypasses levels 1 & 2)
        self.trajectory_cache: Dict[str, TrajectorySample] = {}
        
        # Level 4: Trajectory Hash -> Rendered Output Paths
        self.render_cache: Dict[str, Dict[str, str]] = {}
        
    def clear(self):
        """Flushes all caches."""
        self.text_cache.clear()
        self.prediction_cache.clear()
        self.trajectory_cache.clear()
        self.render_cache.clear()
        
    def get_trajectory(self, text: str) -> Optional[TrajectorySample]:
        if text in self.trajectory_cache:
            # Always return a deepcopy so downstream processors don't pollute the cached master
            return copy.deepcopy(self.trajectory_cache[text])
        return None
        
    def set_trajectory(self, text: str, sample: TrajectorySample):
        # Store a deepcopy to protect the master record
        self.trajectory_cache[text] = copy.deepcopy(sample)
