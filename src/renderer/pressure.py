import math
import copy
from src.registry import Registry
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig

class BasePressure:
    def __init__(self, config: RenderingConfig):
        self.config = config
        
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        raise NotImplementedError

@Registry.register_pressure_model("constant")
class ConstantPressure(BasePressure):
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        new_geom = copy.deepcopy(geom)
        for stroke in new_geom.strokes:
            for pt in stroke.points:
                pt.pressure = 1.0
        return new_geom

@Registry.register_pressure_model("velocity_based")
class VelocityPressure(BasePressure):
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        """
        Simulates pressure based on distance between points.
        Assumes constant sampling rate, so longer distance = higher velocity = lighter pressure.
        """
        new_geom = copy.deepcopy(geom)
        
        def dist(p1, p2):
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            
        # We need global min/max velocity to normalize
        velocities = []
        for stroke in new_geom.strokes:
            pts = stroke.points
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                d = dist(pts[i], pts[i+1])
                velocities.append(d)
                
        if not velocities:
            return new_geom
            
        v_min, v_max = min(velocities), max(velocities)
        if v_max - v_min < 1e-5:
            v_max = v_min + 1e-5
            
        for stroke in new_geom.strokes:
            pts = stroke.points
            if len(pts) < 2:
                for pt in pts:
                    pt.pressure = 1.0
                continue
                
            # First point inherits from segment 0
            d0 = dist(pts[0], pts[1])
            norm_v0 = (d0 - v_min) / (v_max - v_min)
            # Pressure is inversely proportional to velocity
            pts[0].pressure = max(0.1, 1.0 - norm_v0)
            
            for i in range(1, len(pts)):
                d = dist(pts[i-1], pts[i])
                norm_v = (d - v_min) / (v_max - v_min)
                pressure = max(0.1, 1.0 - norm_v)
                pts[i].pressure = pressure
                
            # Smooth pressure slightly to prevent abrupt changes
            for i in range(1, len(pts)-1):
                p_prev = pts[i-1].pressure if pts[i-1].pressure is not None else 1.0
                p_curr = pts[i].pressure if pts[i].pressure is not None else 1.0
                p_next = pts[i+1].pressure if pts[i+1].pressure is not None else 1.0
                pts[i].pressure = (p_prev + p_curr + p_next) / 3.0
                
        return new_geom
