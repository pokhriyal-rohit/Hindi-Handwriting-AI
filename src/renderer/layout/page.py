import copy
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig
from src.registry import Registry

class BaseLayout:
    def __init__(self, config: RenderingConfig):
        self.config = config
        
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        raise NotImplementedError
        
@Registry.register_layout("page")
class PageLayout(BaseLayout):
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        """
        Positions the trajectory onto the canvas.
        Calculates bounding box and scales/translates to fit within margins.
        """
        new_geom = copy.deepcopy(geom)
        
        # Calculate bounding box
        pts = new_geom.to_array() # Returns List[List[float]] -> [x, y, p]
        if not pts:
            return new_geom
            
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        w = max_x - min_x
        h = max_y - min_y
        
        if w == 0 or h == 0:
            return new_geom
            
        export = self.config.export
        target_w = export.canvas_width - 2 * export.margin
        target_h = export.canvas_height - 2 * export.margin
        
        # Keep aspect ratio
        scale = min(target_w / w, target_h / h)
        
        # Center on page
        offset_x = export.margin + (target_w - w * scale) / 2 - (min_x * scale)
        offset_y = export.margin + (target_h - h * scale) / 2 - (min_y * scale)
        
        for stroke in new_geom.strokes:
            for pt in stroke.points:
                pt.x = pt.x * scale + offset_x
                pt.y = pt.y * scale + offset_y
                
        return new_geom
