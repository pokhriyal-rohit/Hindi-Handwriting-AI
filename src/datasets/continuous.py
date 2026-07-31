import numpy as np
import uuid
import time
from typing import Dict, Any, List, Optional
from src.interfaces.representation import CoordinateRepresentation
from src.datasets.structures import TrajectorySample, Stroke, Point, DatasetMetadata
from src.registry import Registry

@Registry.register_representation("modular_continuous")
class ModularCoordinateRepresentation(CoordinateRepresentation):
    """
    Modular Continuous Coordinate Pipeline (Phase 3A Redesign).
    Supports configurable feature extraction and pluggable scalers via Registry.
    """
    def __init__(
        self, 
        features: List[str] = None,
        scaler_name: str = "standard",
        version: str = "2.0"
    ):
        if features is None:
            features = ["dx", "dy", "pen_state"]
        self.features = features
        self.scaler = Registry.get("scalers", scaler_name)() if scaler_name else None
        self.version = version
        
        # Determine which features are continuous (need scaling) vs discrete
        self.discrete_features = {"pen_state", "stroke_id"}
        self.continuous_indices = [
            i for i, f in enumerate(self.features) if f not in self.discrete_features
        ]
        
    def _extract_raw_features(self, trajectory: TrajectorySample) -> np.ndarray:
        """Extracts requested features into a raw N x F array."""
        abs_arr = []
        for stroke in trajectory.strokes:
            for pt in stroke.points:
                row = []
                for f in self.features:
                    if f == "x" or f == "dx": row.append(pt.x)
                    elif f == "y" or f == "dy": row.append(pt.y)
                    elif f == "pen_state": row.append(float(pt.pen_state))
                    elif f == "pressure": row.append(pt.pressure if pt.pressure is not None else 0.0)
                    elif f == "timestamp": row.append(pt.timestamp if pt.timestamp is not None else 0.0)
                    else: row.append(0.0)
                abs_arr.append(row)
                
        abs_arr = np.array(abs_arr, dtype=np.float32)
        if abs_arr.size == 0:
            return abs_arr
            
        # Convert absolute to relative (delta) where requested
        for i, f in enumerate(self.features):
            if f in ("dx", "dy", "dt"):
                deltas = np.zeros_like(abs_arr[:, i])
                deltas[1:] = abs_arr[1:, i] - abs_arr[:-1, i]
                abs_arr[:, i] = deltas
                
        return abs_arr
        
    def fit(self, samples: List[TrajectorySample]) -> None:
        """Fits the pluggable scaler on continuous features across the dataset."""
        if not self.scaler or not self.continuous_indices:
            return
            
        all_continuous = []
        for sample in samples:
            raw = self._extract_raw_features(sample)
            if raw.size > 0:
                all_continuous.append(raw[:, self.continuous_indices])
                
        if all_continuous:
            stacked = np.vstack(all_continuous)
            self.scaler.fit(stacked)
            
    def encode(self, trajectory: TrajectorySample) -> np.ndarray:
        """Encodes trajectory, applying scaler strictly to continuous features."""
        raw = self._extract_raw_features(trajectory)
        if raw.size == 0:
            return raw
            
        encoded = raw.copy()
        if self.scaler and self.continuous_indices:
            continuous_data = encoded[:, self.continuous_indices]
            scaled_data = self.scaler.transform(continuous_data)
            encoded[:, self.continuous_indices] = scaled_data
            
        return encoded
        
    def decode(self, representation: np.ndarray, start_pos=(0.0, 0.0)) -> TrajectorySample:
        """Decodes the representation back to absolute coordinate TrajectorySample."""
        if representation is None or representation.size == 0:
            return None
            
        decoded = representation.copy()
        
        # Inverse transform scaling
        if self.scaler and self.continuous_indices:
            scaled_data = decoded[:, self.continuous_indices]
            unscaled_data = self.scaler.inverse_transform(scaled_data)
            decoded[:, self.continuous_indices] = unscaled_data
            
        # Reconstruct absolute paths
        idx_dx = self.features.index("dx") if "dx" in self.features else -1
        idx_dy = self.features.index("dy") if "dy" in self.features else -1
        idx_pen = self.features.index("pen_state") if "pen_state" in self.features else -1
        idx_press = self.features.index("pressure") if "pressure" in self.features else -1
        
        x_abs = np.zeros(len(decoded))
        y_abs = np.zeros(len(decoded))
        
        if idx_dx != -1:
            x_abs = np.cumsum(decoded[:, idx_dx]) + start_pos[0]
        if idx_dy != -1:
            y_abs = np.cumsum(decoded[:, idx_dy]) + start_pos[1]
            
        strokes = []
        current_stroke = []
        
        for i, row in enumerate(decoded):
            pen_state = int(row[idx_pen]) if idx_pen != -1 else 1
            x = float(x_abs[i])
            y = float(y_abs[i])
            pressure = float(row[idx_press]) if idx_press != -1 else None
            
            current_stroke.append(Point(x=x, y=y, pen_state=pen_state, pressure=pressure))
            
            if pen_state == 0:
                strokes.append(Stroke(stroke_id=len(strokes), points=current_stroke))
                current_stroke = []
                
        if current_stroke:
            strokes.append(Stroke(stroke_id=len(strokes), points=current_stroke))
            
        metadata = DatasetMetadata(
            dataset_name="decoded_modular",
            dataset_version=self.version,
            is_synthetic=False,
            generator_version=None,
            scaling_factor=1.0
        )
        
        return TrajectorySample(
            sample_id=str(uuid.uuid4()),
            writer_id="decoded",
            script="unknown",
            language="unknown",
            text="",
            strokes=strokes,
            metadata=metadata
        )
        
    def statistics(self) -> Dict[str, Any]:
        """Emits versioning and scaling metadata."""
        stats = {
            "version": self.version,
            "features": self.features,
            "scaler": type(self.scaler).__name__ if self.scaler else "None",
            "encoded_at": time.time()
        }
        if self.scaler:
            stats.update(self.scaler.get_metadata())
        return stats
