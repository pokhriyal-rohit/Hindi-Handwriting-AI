from typing import List, Dict, Any
from src.datasets.structures import TrajectorySample

class BaseMetric:
    """
    The canonical Evaluation Metric interface.
    Every benchmark metric must implement this contract.
    """
    
    @classmethod
    def name(cls) -> str:
        """Returns the canonical name of the metric."""
        raise NotImplementedError
        
    @classmethod
    def version(cls) -> str:
        """Returns the version of the metric implementation."""
        raise NotImplementedError
        
    @classmethod
    def description(cls) -> str:
        """Returns a short description of what the metric measures."""
        raise NotImplementedError
        
    def validate(self, prediction: TrajectorySample, target: TrajectorySample) -> bool:
        """
        Verifies the inputs are valid for this specific metric.
        Raises ValueError if invalid, or returns True/False.
        """
        if not isinstance(prediction, TrajectorySample) or not isinstance(target, TrajectorySample):
            raise TypeError("Metrics strictly require TrajectorySample objects.")
        return True

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> Dict[str, Any]:
        """
        Executes the core calculation.
        Returns a dictionary of raw scores.
        """
        raise NotImplementedError

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates a batch of evaluation results into statistical summaries (mean, std, etc.).
        """
        raise NotImplementedError
