import numpy as np
from pathlib import Path
from typing import List
from .structures import Point, Stroke, Trajectory

class IIITHWParser:
    """Parses raw IIIT-HW-Devanagari dataset files into Trajectory objects."""
    
    @staticmethod
    def parse_txt_file(filepath: str | Path, text_label: str = "", writer_id: str = "") -> Trajectory:
        """
        Parses a standard text file containing coordinates.
        Assuming format: x, y, pen_state (or similar).
        This is a skeleton parser to be adapted to the exact IIIT-HW format.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
            
        strokes = []
        current_stroke_points = []
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        # Normalize pen state to 0 or 1
                        p = int(float(parts[2]))
                        p = 1 if p > 0 else 0
                        
                        point = Point(x=x, y=y, p=p)
                        current_stroke_points.append(point)
                        
                        if p == 0:
                            # Pen lift indicates end of stroke
                            strokes.append(Stroke(points=current_stroke_points))
                            current_stroke_points = []
                            
                    except ValueError:
                        continue # Skip invalid lines
                        
        if current_stroke_points:
            strokes.append(Stroke(points=current_stroke_points))
            
        return Trajectory(strokes=strokes, text=text_label, writer_id=writer_id)

    @staticmethod
    def normalize_trajectory(traj: Trajectory) -> Trajectory:
        """
        Normalizes the trajectory coordinates to have zero mean and unit variance,
        or translates it so the top-left is at (0,0).
        """
        all_pts = traj.to_array()
        if not all_pts:
            return traj
            
        arr = np.array(all_pts)
        x_coords = arr[:, 0]
        y_coords = arr[:, 1]
        
        # Simple translation to origin
        min_x = np.min(x_coords)
        min_y = np.min(y_coords)
        
        normalized_strokes = []
        for stroke in traj.strokes:
            norm_pts = []
            for pt in stroke.points:
                norm_pts.append(Point(x=pt.x - min_x, y=pt.y - min_y, p=pt.p))
            normalized_strokes.append(Stroke(points=norm_pts))
            
        return Trajectory(strokes=normalized_strokes, text=traj.text, writer_id=traj.writer_id)
