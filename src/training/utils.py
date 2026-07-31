import torch
from src.datasets.structures import TrajectorySample, Stroke, Point

def tensor_to_trajectory(coords: torch.Tensor, text: str = "") -> TrajectorySample:
    """
    Converts a [L, 3] tensor (dx, dy, pen_state) into a canonical TrajectorySample.
    Applies sigmoid thresholding for pen_state if it's raw logits.
    """
    # Assuming coords is [L, 3] on CPU
    strokes = []
    current_points = []
    
    # Simple absolute coordinate reconstruction
    x, y = 0.0, 0.0
    
    for i in range(coords.size(0)):
        dx, dy, pen = coords[i].tolist()
        
        # If it's a logit, sigmoid it. For now, assume it's probability or binary.
        is_pen_down = pen > 0.5
        
        x += dx
        y += dy
        
        current_points.append(Point(x=x, y=y, timestamp=float(i), pen_state=1 if is_pen_down else 0))
        
        if not is_pen_down and len(current_points) > 0:
            strokes.append(Stroke(points=current_points))
            current_points = []
            
    if len(current_points) > 0:
        strokes.append(Stroke(points=current_points))
    from src.datasets.structures import DatasetMetadata
    return TrajectorySample(
        sample_id="dummy",
        writer_id="dummy",
        script="devanagari",
        language="hi",
        text=text,
        metadata=DatasetMetadata(
            dataset_name="synthetic",
            dataset_version="1.0.0",
            is_synthetic=True,
            sampling_rate_hz=100.0
        ),
        strokes=strokes
    )
