import tempfile
import os
from src.registry import Registry
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig
from src.renderer.exceptions import ExporterError
from src.renderer.exporters.base import BaseExporter
from src.renderer.exporters.svg import SVGExporter

try:
    import cairosvg
except ImportError:
    cairosvg = None

@Registry.register_exporter("png")
class PNGExporter(BaseExporter):
    def __init__(self, config: RenderingConfig):
        super().__init__(config)
        self.tmp_svg_path = None
        self.svg_exporter = None
        
    def initialize(self) -> None:
        if cairosvg is None:
            raise ExporterError("cairosvg is required for PNG export. Please install it: pip install cairosvg")
        self.svg_exporter = SVGExporter(self.config)
        self.svg_exporter.initialize()
        super().initialize()
        
    def export(self, geom: TrajectorySample, output_path: str) -> None:
        super().export(geom, output_path)
        
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
            self.tmp_svg_path = tmp_svg.name
            
        self.svg_exporter.export(geom, self.tmp_svg_path)
        
        cairosvg.svg2png(
            url=self.tmp_svg_path, 
            write_to=output_path,
            dpi=self.config.export.dpi
        )
            
    def cleanup(self) -> None:
        if self.tmp_svg_path and os.path.exists(self.tmp_svg_path):
            os.remove(self.tmp_svg_path)
            self.tmp_svg_path = None
        if self.svg_exporter:
            self.svg_exporter.cleanup()
        super().cleanup()
