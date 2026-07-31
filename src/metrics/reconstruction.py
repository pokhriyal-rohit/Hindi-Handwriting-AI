import numpy as np
from typing import Dict, Tuple
from src.datasets.structures import TrajectorySample

class ReconstructionMetrics:
    """
    Computes numerical validation metrics between an original and decoded trajectory.
    """
    
    @staticmethod
    def calculate_metrics(original: TrajectorySample, decoded: TrajectorySample) -> Dict[str, float]:
        """Calculates RMSE, MAE, Max Error, and Endpoint Error."""
        orig_arr = []
        for stroke in original.strokes:
            for pt in stroke.points:
                orig_arr.append([pt.x, pt.y])
        
        dec_arr = []
        for stroke in decoded.strokes:
            for pt in stroke.points:
                dec_arr.append([pt.x, pt.y])
                
        orig_arr = np.array(orig_arr, dtype=np.float32)
        dec_arr = np.array(dec_arr, dtype=np.float32)
        
        if orig_arr.size == 0 or dec_arr.size == 0:
            return {"rmse": 0.0, "mae": 0.0, "max_error": 0.0, "endpoint_error": 0.0}
            
        # Ensure they are the same length
        min_len = min(len(orig_arr), len(dec_arr))
        orig_arr = orig_arr[:min_len]
        dec_arr = dec_arr[:min_len]
        
        diff = orig_arr - dec_arr
        dist = np.linalg.norm(diff, axis=1) # Euclidean distance per point
        
        rmse = float(np.sqrt(np.mean(dist**2)))
        mae = float(np.mean(dist))
        max_err = float(np.max(dist))
        
        endpoint_err = float(dist[-1]) if len(dist) > 0 else 0.0
        
        return {
            "rmse": rmse,
            "mae": mae,
            "max_error": max_err,
            "endpoint_error": endpoint_err
        }
