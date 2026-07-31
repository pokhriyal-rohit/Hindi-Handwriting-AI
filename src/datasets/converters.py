import os
import json
from src.datasets.structures import TrajectorySample, Stroke, Point, DatasetMetadata

class CustomCollectorConverter:
    """
    Converts the raw JSON trajectories collected from the local Web UI Collector
    into the canonical TrajectorySample.
    """
    
    @staticmethod
    def from_json(filepath: str, writer_id: str = "custom_001") -> TrajectorySample:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        word = data.get("word", "")
        raw_strokes = data.get("strokes", [])
        
        sample_id = os.path.basename(filepath).replace(".json", "")
        
        canonical_strokes = []
        for raw_stroke in raw_strokes:
            points = []
            for i, p in enumerate(raw_stroke):
                # pen_state is 1.0 (down) for all points except the very last one where we might lift
                is_last = (i == len(raw_stroke) - 1)
                points.append(
                    Point(
                        x=float(p["x"]),
                        y=float(p["y"]),
                        timestamp=float(p.get("t", 0.0)),
                        pen_state=0 if is_last else 1
                    )
                )
            if points:
                canonical_strokes.append(Stroke(points=points))
                
        metadata = DatasetMetadata(
            dataset_name="custom_collector",
            dataset_version="1.0.0",
            sampling_rate_hz=60.0,  # Browsers typically fire mousemove at ~60Hz
            is_synthetic=False
        )
        
        return TrajectorySample(
            sample_id=sample_id,
            writer_id=writer_id,
            script="devanagari",
            language="hi",
            text=word,
            metadata=metadata,
            strokes=canonical_strokes
        )
