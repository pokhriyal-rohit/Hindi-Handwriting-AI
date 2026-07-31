import math
from typing import Dict, Any, List
from src.registry import Registry
from src.datasets.structures import TrajectorySample
from src.evaluation.metrics.base import BaseMetric

def _flatten_points(sample: TrajectorySample) -> List[List[float]]:
    """Helper to flatten a TrajectorySample into an ordered list of [x, y] coordinates."""
    pts = []
    for stroke in sample.strokes:
        for pt in stroke.points:
            pts.append([pt.x, pt.y])
    return pts

@Registry.register_metric("dtw")
class DTWMetric(BaseMetric):
    @classmethod
    def name(cls) -> str:
        return "dtw"
        
    @classmethod
    def version(cls) -> str:
        return "1.0.0"
        
    @classmethod
    def description(cls) -> str:
        return "Dynamic Time Warping distance between two trajectory coordinate sequences."

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> Dict[str, Any]:
        self.validate(prediction, target)
        
        try:
            import numpy as np
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
        except ImportError:
            raise ImportError("DTWMetric requires 'fastdtw', 'scipy', and 'numpy'. pip install fastdtw scipy numpy")
            
        pred_pts = np.array(_flatten_points(prediction))
        tgt_pts = np.array(_flatten_points(target))
        
        if len(pred_pts) == 0 or len(tgt_pts) == 0:
            return {"dtw_distance": float('inf')}
            
        distance, path = fastdtw(pred_pts, tgt_pts, dist=euclidean)
        return {"dtw_distance": float(distance)}

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            import numpy as np
        except ImportError:
            return {}
            
        scores = [r["dtw_distance"] for r in results if r["dtw_distance"] != float('inf')]
        if not scores: return {}
        
        return {
            "dtw_mean": float(np.mean(scores)),
            "dtw_median": float(np.median(scores)),
            "dtw_std": float(np.std(scores)),
            "dtw_min": float(np.min(scores)),
            "dtw_max": float(np.max(scores)),
            "dtw_samples": len(scores)
        }

@Registry.register_metric("frechet")
class FrechetMetric(BaseMetric):
    @classmethod
    def name(cls) -> str:
        return "frechet"
        
    @classmethod
    def version(cls) -> str:
        return "1.0.0"
        
    @classmethod
    def description(cls) -> str:
        return "Discrete Frechet Distance measuring geometric similarity between curves."

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> Dict[str, Any]:
        self.validate(prediction, target)
        
        try:
            import numpy as np
            import similaritymeasures
        except ImportError:
            raise ImportError("FrechetMetric requires 'similaritymeasures' and 'numpy'. pip install similaritymeasures numpy")
            
        pred_pts = np.array(_flatten_points(prediction))
        tgt_pts = np.array(_flatten_points(target))
        
        if len(pred_pts) == 0 or len(tgt_pts) == 0:
            return {"frechet_distance": float('inf')}
            
        # Using similaritymeasures for Discrete Frechet
        distance = similaritymeasures.frechet_dist(pred_pts, tgt_pts)
        return {"frechet_distance": float(distance)}

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            import numpy as np
        except ImportError:
            return {}
            
        scores = [r["frechet_distance"] for r in results if r["frechet_distance"] != float('inf')]
        if not scores: return {}
        
        return {
            "frechet_mean": float(np.mean(scores)),
            "frechet_std": float(np.std(scores))
        }

@Registry.register_metric("stroke_count")
class StrokeCountDifferenceMetric(BaseMetric):
    @classmethod
    def name(cls) -> str:
        return "stroke_count"
        
    @classmethod
    def version(cls) -> str:
        return "1.0.0"
        
    @classmethod
    def description(cls) -> str:
        return "Absolute difference in the number of strokes between prediction and target."

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> Dict[str, Any]:
        self.validate(prediction, target)
        pred_strokes = len(prediction.strokes)
        tgt_strokes = len(target.strokes)
        diff = abs(pred_strokes - tgt_strokes)
        
        return {
            "pred_strokes": pred_strokes,
            "target_strokes": tgt_strokes,
            "stroke_difference": diff
        }

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            import numpy as np
        except ImportError:
            return {}
            
        diffs = [r["stroke_difference"] for r in results]
        if not diffs: return {}
        
        return {
            "stroke_difference_mean": float(np.mean(diffs)),
            "stroke_difference_max": float(np.max(diffs))
        }

@Registry.register_metric("endpoint_error")
class EndpointErrorMetric(BaseMetric):
    @classmethod
    def name(cls) -> str:
        return "endpoint_error"
        
    @classmethod
    def version(cls) -> str:
        return "1.0.0"
        
    @classmethod
    def description(cls) -> str:
        return "Euclidean distance between the final endpoints of the trajectories."

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> Dict[str, Any]:
        self.validate(prediction, target)
        
        pred_pts = _flatten_points(prediction)
        tgt_pts = _flatten_points(target)
        
        if len(pred_pts) == 0 or len(tgt_pts) == 0:
            return {"endpoint_error": float('inf')}
            
        p_end = pred_pts[-1]
        t_end = tgt_pts[-1]
        
        dist = math.sqrt((p_end[0] - t_end[0])**2 + (p_end[1] - t_end[1])**2)
        return {"endpoint_error": float(dist)}

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            import numpy as np
        except ImportError:
            return {}
            
        scores = [r["endpoint_error"] for r in results if r["endpoint_error"] != float('inf')]
        if not scores: return {}
        return {
            "endpoint_error_mean": float(np.mean(scores)),
            "endpoint_error_std": float(np.std(scores))
        }
