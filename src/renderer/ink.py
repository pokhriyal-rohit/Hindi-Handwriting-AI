import copy
import math
from src.registry import Registry
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig

class BaseInk:
    def __init__(self, config: RenderingConfig):
        self.config = config
        
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        raise NotImplementedError

@Registry.register_ink_model("constant")
class ConstantInk(BaseInk):
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        """
        Constant width and color.
        Ink attributes are not natively on TrajectorySample yet, 
        so we might append metadata to the stroke.
        For SVG rendering, we can rely on stroke-width.
        """
        return geom

@Registry.register_ink_model("fountain_pen")
class FountainPenInk(BaseInk):
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        """
        Calculates stroke width scaling based on drawing angle.
        Vertical strokes = thick, Horizontal strokes = thin.
        This doesn't modify geometry, but we can encode thickness variations in pressure
        if the SVG renderer knows how to use pressure for width.
        Actually, we can hijack `pressure` or add to it.
        Let's modify pressure so `SVGExporter` can use it to vary stroke width.
        """
        new_geom = copy.deepcopy(geom)
        
        for stroke in new_geom.strokes:
            pts = stroke.points
            if len(pts) < 2:
                continue
                
            for i in range(1, len(pts)):
                dx = pts[i].x - pts[i-1].x
                dy = pts[i].y - pts[i-1].y
                
                # Angle relative to vertical
                angle = math.atan2(abs(dx), abs(dy))
                
                # angle is 0 for vertical (thick), pi/2 for horizontal (thin)
                thickness_factor = 1.0 - (angle / (math.pi / 2)) * 0.7 # Between 0.3 and 1.0
                
                # Modulate existing pressure
                pres = pts[i].pressure if pts[i].pressure is not None else 1.0
                pts[i].pressure = pres * thickness_factor
                
            # First point inherits from second
            pts[0].pressure = pts[1].pressure
            
        return new_geom
