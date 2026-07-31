import math
from typing import Dict, Any, List
from src.registry import Registry
from src.datasets.structures import TrajectorySample
from src.evaluation.metrics.base import BaseMetric
from src.evaluation.metrics.trajectory import _flatten_points

def _calculate_path_length(pts: List[List[float]]) -> float:
    length = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        length += math.sqrt(dx**2 + dy**2)
    return length

def _calculate_bounding_box(pts: List[List[float]]) -> tuple:
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

@Registry.register_metric("path_length")
class PathLengthDifferenceMetric(BaseMetric):
    @classmethod
    def name(cls) -> str:
        return "path_length"
        
    @classmethod
    def version(cls) -> str:
        return "1.0.0"
        
    @classmethod
    def description(cls) -> str:
        return "Absolute difference in the total path length (Euclidean arc length) of the curves."

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> Dict[str, Any]:
        self.validate(prediction, target)
        
        pred_len = sum(_calculate_path_length([[pt.x, pt.y] for pt in stroke.points]) for stroke in prediction.strokes)
        tgt_len = sum(_calculate_path_length([[pt.x, pt.y] for pt in stroke.points]) for stroke in target.strokes)
        
        return {
            "pred_length": pred_len,
            "target_length": tgt_len,
            "length_difference": abs(pred_len - tgt_len)
        }

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            import numpy as np
            diffs = [r["length_difference"] for r in results]
            if not diffs: return {}
            return {
                "length_difference_mean": float(np.mean(diffs)),
                "length_difference_std": float(np.std(diffs))
            }
        except ImportError:
            return {}

@Registry.register_metric("bounding_box")
class BoundingBoxDifferenceMetric(BaseMetric):
    @classmethod
    def name(cls) -> str:
        return "bounding_box"
        
    @classmethod
    def version(cls) -> str:
        return "1.0.0"
        
    @classmethod
    def description(cls) -> str:
        return "Difference in width, height, and area of the coordinate bounding boxes."

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> Dict[str, Any]:
        self.validate(prediction, target)
        
        p_minx, p_miny, p_maxx, p_maxy = _calculate_bounding_box(_flatten_points(prediction))
        t_minx, t_miny, t_maxx, t_maxy = _calculate_bounding_box(_flatten_points(target))
        
        p_w, p_h = p_maxx - p_minx, p_maxy - p_miny
        t_w, t_h = t_maxx - t_minx, t_maxy - t_miny
        
        return {
            "width_difference": abs(p_w - t_w),
            "height_difference": abs(p_h - t_h),
            "area_difference": abs((p_w * p_h) - (t_w * t_h))
        }

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            import numpy as np
            w_diffs = [r["width_difference"] for r in results]
            h_diffs = [r["height_difference"] for r in results]
            a_diffs = [r["area_difference"] for r in results]
            if not w_diffs: return {}
            return {
                "width_diff_mean": float(np.mean(w_diffs)),
                "height_diff_mean": float(np.mean(h_diffs)),
                "area_diff_mean": float(np.mean(a_diffs))
            }
        except ImportError:
            return {}

@Registry.register_metric("smoothness")
class SmoothnessScoreMetric(BaseMetric):
    @classmethod
    def name(cls) -> str:
        return "smoothness"
        
    @classmethod
    def version(cls) -> str:
        return "1.0.0"
        
    @classmethod
    def description(cls) -> str:
        return "Variance of angular changes between consecutive segments. Lower is smoother."

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> Dict[str, Any]:
        self.validate(prediction, target)
        
        try:
            import numpy as np
        except ImportError:
            raise ImportError("SmoothnessScoreMetric requires 'numpy'.")
            
        def _get_angles(pts):
            if len(pts) < 3: return [0.0]
            pts = np.array(pts)
            v1 = pts[1:-1] - pts[:-2]
            v2 = pts[2:] - pts[1:-1]
            
            # Normalize
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            
            # Avoid division by zero
            valid = (n1 > 0) & (n2 > 0)
            if not np.any(valid): return [0.0]
            
            cos_theta = np.sum(v1[valid] * v2[valid], axis=1) / (n1[valid] * n2[valid])
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return np.arccos(cos_theta)
            
        pred_angles = []
        for stroke in prediction.strokes:
            pred_angles.extend(_get_angles([[pt.x, pt.y] for pt in stroke.points]))
            
        tgt_angles = []
        for stroke in target.strokes:
            tgt_angles.extend(_get_angles([[pt.x, pt.y] for pt in stroke.points]))
            
        p_smooth = np.var(pred_angles) if pred_angles else 0.0
        t_smooth = np.var(tgt_angles) if tgt_angles else 0.0
        
        return {
            "pred_smoothness": float(p_smooth),
            "target_smoothness": float(t_smooth),
            "smoothness_difference": abs(float(p_smooth - t_smooth))
        }

    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            import numpy as np
            diffs = [r["smoothness_difference"] for r in results]
            if not diffs: return {}
            return {
                "smoothness_diff_mean": float(np.mean(diffs)),
                "smoothness_diff_std": float(np.std(diffs))
            }
        except ImportError:
            return {}
