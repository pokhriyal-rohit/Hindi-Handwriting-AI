from typing import List
from src.registry import Registry
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig
from src.renderer.exporters.base import BaseExporter

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image, ImageDraw = None, None

@Registry.register_exporter("gif")
class GIFExporter(BaseExporter):
    def export(self, geom: TrajectorySample, output_path: str) -> None:
        if Image is None:
            raise ImportError("Pillow is required for GIF export. Please install it: pip install Pillow")
            
        export_cfg = self.config.export
        width = export_cfg.canvas_width
        height = export_cfg.canvas_height
        bg_color = export_cfg.background_color
        stroke_color = export_cfg.stroke_color
        
        frames = []
        
        # Base canvas
        current_img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(current_img)
        
        # Extract flat sequence of segments to draw step-by-step
        segments = []
        for stroke in geom.strokes:
            pts = stroke.points
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i+1]
                pres = p1.pressure if p1.pressure is not None else 1.0
                lw = self.config.base_stroke_width * pres
                segments.append(((p1.x, p1.y), (p2.x, p2.y), lw))
                
        # Draw progressively. To avoid too many frames, we can draw N segments per frame.
        # Let's say we want roughly 60 frames max for speed.
        if not segments:
            current_img.save(output_path)
            return
            
        step_size = max(1, len(segments) // 60)
        
        for i, ((x1, y1), (x2, y2), lw) in enumerate(segments):
            draw.line([(x1, y1), (x2, y2)], fill=stroke_color, width=int(max(1, lw)), joint="curve")
            
            if i % step_size == 0 or i == len(segments) - 1:
                frames.append(current_img.copy())
                
        if not frames:
            frames.append(current_img)
            
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=50, # ms per frame
            loop=0
        )
