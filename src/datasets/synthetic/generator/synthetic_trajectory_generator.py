import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath

from src.interfaces.dataset import BaseDataset
from src.datasets.structures import TrajectorySample, Stroke, Point, DatasetMetadata
from src.registry import Registry

class BezierSampler:
    """Utility to sample points along Bezier curves for synthetic trajectory generation."""
    @staticmethod
    def sample_quadratic(p0, p1, p2, num_points=10):
        t = np.linspace(0, 1, num_points)[:, np.newaxis]
        return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2

    @staticmethod
    def sample_cubic(p0, p1, p2, p3, num_points=10):
        t = np.linspace(0, 1, num_points)[:, np.newaxis]
        return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

@Registry.register_dataset("synthetic_font")
class SyntheticTrajectoryGenerator(BaseDataset):
    """
    Stage 1 Bootstrap Generator.
    Generates online coordinate trajectories from TTF fonts for engineering validation.
    """
    def __init__(self, font_path: str = "C:/Windows/Fonts/mangal.ttf"):
        self.font_path = font_path
        self.font_prop = FontProperties(fname=self.font_path)
        self.trajectories: List[TrajectorySample] = []
        
    def load(self, path: str = None) -> None:
        """
        Loads text lines and converts them into synthetic trajectories.
        If no path is provided, generates a default test set.
        """
        texts = ["नमस्ते", "भारत", "हिंदी", "देवनागरी"]
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
                
        for txt in texts:
            traj = self._text_to_trajectory(txt)
            if traj.strokes:
                self.trajectories.append(traj)
                
    def _text_to_trajectory(self, text: str) -> TrajectorySample:
        # Use Matplotlib's TextPath to extract glyph outlines
        text_path = TextPath((0, 0), text, prop=self.font_prop, size=100)
        vertices = text_path.vertices
        codes = text_path.codes
        
        strokes = []
        current_stroke = []
        
        if len(vertices) == 0:
            import uuid
            metadata = DatasetMetadata(
                dataset_name="synthetic_font",
                dataset_version="1.0",
                is_synthetic=True,
                font_name="mangal.ttf"
            )
            return TrajectorySample(
                sample_id=str(uuid.uuid4()),
                writer_id="synthetic_writer_1",
                script="devanagari",
                language="hi",
                text=text,
                strokes=[],
                metadata=metadata
            )
            
        i = 0
        while i < len(codes):
            code = codes[i]
            pt = vertices[i]
            
            if code == MplPath.MOVETO:
                if current_stroke:
                    strokes.append(Stroke(points=current_stroke))
                current_stroke = [Point(x=float(pt[0]), y=float(pt[1]), pen_state=1)]
                i += 1
            elif code == MplPath.LINETO:
                current_stroke.append(Point(x=float(pt[0]), y=float(pt[1]), pen_state=1))
                i += 1
            elif code == MplPath.CURVE3:
                if i + 1 < len(vertices) and current_stroke:
                    p0 = np.array([current_stroke[-1].x, current_stroke[-1].y])
                    p1 = vertices[i]
                    p2 = vertices[i+1]
                    sampled = BezierSampler.sample_quadratic(p0, p1, p2, num_points=6)
                    for spt in sampled[1:]:
                        current_stroke.append(Point(x=float(spt[0]), y=float(spt[1]), pen_state=1))
                i += 2
            elif code == MplPath.CURVE4:
                if i + 2 < len(vertices) and current_stroke:
                    p0 = np.array([current_stroke[-1].x, current_stroke[-1].y])
                    p1 = vertices[i]
                    p2 = vertices[i+1]
                    p3 = vertices[i+2]
                    sampled = BezierSampler.sample_cubic(p0, p1, p2, p3, num_points=8)
                    for spt in sampled[1:]:
                        current_stroke.append(Point(x=float(spt[0]), y=float(spt[1]), pen_state=1))
                i += 3
            elif code == MplPath.CLOSEPOLY:
                if current_stroke:
                    start_pt = current_stroke[0]
                    current_stroke.append(Point(x=float(start_pt.x), y=float(start_pt.y), pen_state=1))
                    strokes.append(Stroke(stroke_id=len(strokes), points=current_stroke))
                    current_stroke = []
                i += 1
            else:
                i += 1
                
        if current_stroke:
            strokes.append(Stroke(stroke_id=len(strokes), points=current_stroke))
            
        # Pen lift (p=0) is the last point of every stroke
        for stroke in strokes:
            if stroke.points:
                stroke.points[-1].pen_state = 0
                
        import uuid
        metadata = DatasetMetadata(
            dataset_name="synthetic_font",
            dataset_version="1.0",
            is_synthetic=True,
            generator_version="1.0",
            font_name="mangal.ttf"
        )
        
        return TrajectorySample(
            sample_id=str(uuid.uuid4()),
            writer_id="synthetic_writer_1",
            script="devanagari",
            language="hi",
            text=text,
            strokes=strokes,
            metadata=metadata
        )

    def validate(self) -> bool:
        """Validates that synthetic trajectories were successfully generated."""
        return len(self.trajectories) > 0
        
    def analyze(self) -> Dict[str, Any]:
        """Provides statistics about the synthetic trajectories."""
        total_strokes = sum(len(t.strokes) for t in self.trajectories)
        total_pts = sum(len(s.points) for t in self.trajectories for s in t.strokes)
        return {
            "total_samples": len(self.trajectories),
            "total_strokes": total_strokes,
            "total_points": total_pts,
            "synthetic_mode": True
        }
