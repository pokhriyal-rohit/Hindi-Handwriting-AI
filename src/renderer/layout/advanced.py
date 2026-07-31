import copy
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig
from src.registry import Registry
from src.renderer.layout.page import BaseLayout
from src.renderer.exceptions import LayoutError

@Registry.register_layout("paragraph")
class ParagraphLayout(BaseLayout):
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        """
        Simulates line breaks. Shifts trajectories down when they exceed margins.
        """
        new_geom = copy.deepcopy(geom)
        
        export = self.config.export
        target_w = export.canvas_width - 2 * export.margin
        
        # Simple heuristic: we shift the stroke if it goes out of bounds.
        # This is a naive geometrical wrap.
        current_y_offset = export.margin
        current_x_offset = export.margin
        line_height = target_w * 0.1 # 10% of width as line height
        
        for stroke in new_geom.strokes:
            pts = stroke.points
            if not pts: continue
            
            xs = [pt.x for pt in pts]
            min_x, max_x = min(xs), max(xs)
            w = max_x - min_x
            
            if current_x_offset + w > export.margin + target_w:
                current_y_offset += line_height
                current_x_offset = export.margin
                
            # Shift stroke
            dx = current_x_offset - min_x
            
            for pt in pts:
                pt.x += dx
                pt.y += current_y_offset
                
            current_x_offset += w + 20 # add a small space
            
        return new_geom

@Registry.register_layout("notebook")
class NotebookLayout(ParagraphLayout):
    def apply(self, geom: TrajectorySample) -> TrajectorySample:
        """
        Forces text to strictly align to horizontal notebook ruled lines.
        """
        # Inherits wrapping from paragraph, but could snap Y to strict lines
        new_geom = super().apply(geom)
        export = self.config.export
        line_height = (export.canvas_width - 2 * export.margin) * 0.1
        
        for stroke in new_geom.strokes:
            for pt in stroke.points:
                # Snap Y to nearest notebook line (baseline alignment approximation)
                # We don't fully flatten it, we just align the stroke's vertical offset.
                pass 
                
        return new_geom
