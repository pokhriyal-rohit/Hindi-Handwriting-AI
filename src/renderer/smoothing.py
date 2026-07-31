import numpy as np
import copy
from src.registry import Registry
from src.datasets.structures import TrajectorySample, Stroke, Point
from src.renderer.config import RenderingConfig

class BaseSmoother:
    def __init__(self, config: RenderingConfig):
        self.config = config
        
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        raise NotImplementedError

@Registry.register_smoother("moving_average")
class MovingAverageSmoother(BaseSmoother):
    def __init__(self, config: RenderingConfig, window_size: int = 3):
        super().__init__(config)
        self.window_size = window_size
        
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        smoothed_geom = copy.deepcopy(geom)
        smoothed_geom.strokes = []
        
        for stroke in geom.strokes:
            pts = stroke.points
            if len(pts) < self.window_size:
                smoothed_geom.strokes.append(copy.deepcopy(stroke))
                continue
                
            new_pts = []
            # Keep first point
            new_pts.append(copy.deepcopy(pts[0]))
            
            for i in range(1, len(pts) - 1):
                start = max(0, i - self.window_size // 2)
                end = min(len(pts), i + self.window_size // 2 + 1)
                
                window = pts[start:end]
                avg_x = sum(p.x for p in window) / len(window)
                avg_y = sum(p.y for p in window) / len(window)
                
                new_pts.append(Point(x=avg_x, y=avg_y, pen_state=pts[i].pen_state, pressure=pts[i].pressure, timestamp=pts[i].timestamp))
                
            # Keep last point
            new_pts.append(copy.deepcopy(pts[-1]))
            smoothed_geom.strokes.append(Stroke(stroke_id=stroke.stroke_id, points=new_pts))
            
        return smoothed_geom

@Registry.register_smoother("bezier_fit")
class BezierSmoother(BaseSmoother):
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        smoothed_geom = copy.deepcopy(geom)
        smoothed_geom.strokes = []
        iterations = 2
        
        for stroke in geom.strokes:
            pts = stroke.points
            if len(pts) < 3:
                smoothed_geom.strokes.append(copy.deepcopy(stroke))
                continue
                
            current_pts = copy.deepcopy(pts)
            for _ in range(iterations):
                new_pts = [current_pts[0]]
                for i in range(len(current_pts) - 1):
                    p0 = current_pts[i]
                    p1 = current_pts[i+1]
                    
                    pres_0 = p0.pressure if p0.pressure is not None else 1.0
                    pres_1 = p1.pressure if p1.pressure is not None else 1.0
                    
                    q = Point(
                        x=0.75*p0.x + 0.25*p1.x, 
                        y=0.75*p0.y + 0.25*p1.y,
                        pen_state=p0.pen_state,
                        pressure=0.75*pres_0 + 0.25*pres_1
                    )
                    r = Point(
                        x=0.25*p0.x + 0.75*p1.x, 
                        y=0.25*p0.y + 0.75*p1.y,
                        pen_state=p1.pen_state,
                        pressure=0.25*pres_0 + 0.75*pres_1
                    )
                    new_pts.extend([q, r])
                new_pts.append(current_pts[-1])
                current_pts = new_pts
                
            smoothed_geom.strokes.append(Stroke(stroke_id=stroke.stroke_id, points=current_pts))
            
        return smoothed_geom
