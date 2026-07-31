import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional
from src.registry import Registry
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig
from src.renderer.exceptions import ExporterError
from src.renderer.exporters.base import BaseExporter

@Registry.register_exporter("svg")
class SVGExporter(BaseExporter):
    def initialize(self) -> None:
        super().initialize()
        
    def export(self, geom: TrajectorySample, output_path: str) -> None:
        super().export(geom, output_path)
        
        export_cfg = self.config.export
        
        svg = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'version': '1.1',
            'width': str(export_cfg.canvas_width),
            'height': str(export_cfg.canvas_height),
            'viewBox': f"0 0 {export_cfg.canvas_width} {export_cfg.canvas_height}"
        })
        
        # Background
        ET.SubElement(svg, 'rect', {
            'width': '100%',
            'height': '100%',
            'fill': export_cfg.background_color
        })
        
        group = ET.SubElement(svg, 'g', {
            'fill': 'none',
            'stroke': export_cfg.stroke_color,
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round'
        })
        
        for stroke in geom.strokes:
            pts = stroke.points
            if len(pts) < 2:
                continue
                
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i+1]
                pres = p1.pressure if p1.pressure is not None else 1.0
                width = self.config.base_stroke_width * pres
                
                ET.SubElement(group, 'path', {
                    'd': f"M {p1.x} {p1.y} L {p2.x} {p2.y}",
                    'stroke-width': str(width)
                })
                
        xmlstr = minidom.parseString(ET.tostring(svg)).toprettyxml(indent="  ")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xmlstr)
            
    def validate(self, output_path: str) -> bool:
        if not super().validate(output_path):
            return False
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read(100)
            return "<svg" in content
