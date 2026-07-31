import os
import subprocess
import tempfile
from typing import List
from src.registry import Registry
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig
from src.renderer.exporters.base import BaseExporter

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image, ImageDraw = None, None

@Registry.register_exporter("mp4")
class MP4Exporter(BaseExporter):
    def export(self, geom: TrajectorySample, output_path: str) -> None:
        if Image is None:
            raise ImportError("Pillow is required for MP4 frame generation. Please install it: pip install Pillow")
            
        export_cfg = self.config.export
        width = export_cfg.canvas_width
        height = export_cfg.canvas_height
        bg_color = export_cfg.background_color
        stroke_color = export_cfg.stroke_color
        
        # Ensure dimensions are even for ffmpeg x264
        width = width if width % 2 == 0 else width + 1
        height = height if height % 2 == 0 else height + 1
        
        current_img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(current_img)
        
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
                
        if not segments:
            # Create a 1-second empty video
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi', '-i', f'color=c={bg_color}:s={width}x{height}:d=1',
                output_path
            ], check=True)
            return
            
        # Write frames to ffmpeg via stdin
        ffmpeg_cmd = [
            'ffmpeg', '-y', 
            '-f', 'image2pipe', 
            '-vcodec', 'png', 
            '-r', '30', # 30 fps
            '-i', '-', 
            '-vcodec', 'libx264', 
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        step_size = max(1, len(segments) // 90) # ~3 seconds video
        
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        try:
            for i, ((x1, y1), (x2, y2), lw) in enumerate(segments):
                draw.line([(x1, y1), (x2, y2)], fill=stroke_color, width=int(max(1, lw)), joint="curve")
                
                if i % step_size == 0 or i == len(segments) - 1:
                    current_img.save(process.stdin, format='PNG')
        finally:
            process.stdin.close()
            process.wait()
