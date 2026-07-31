from pydantic import BaseModel, Field
from typing import Literal, Optional

class ExportConfig(BaseModel):
    """Configuration for output rendering."""
    canvas_width: int = Field(default=1920, ge=100)
    canvas_height: int = Field(default=1080, ge=100)
    margin: int = Field(default=50, ge=0)
    dpi: int = Field(default=300, ge=72)
    background_color: str = Field(default="#FFFFFF")
    stroke_color: str = Field(default="#000000")
    
class RenderingConfig(BaseModel):
    """Main configuration orchestrating the Rendering Engine pipeline."""
    # Versioning
    renderer_version: str = Field(default="1.0.0")
    layout_version: str = Field(default="1.0.0")
    exporter_version: str = Field(default="1.0.0")
    plugin_versions: dict = Field(default_factory=dict)
    
    # Plugins
    layout_model: str = Field(default="page") # e.g., 'page', 'notebook'
    interpolation: Optional[str] = Field(default=None) # e.g., 'linear', 'catmull_rom'
    smoothing: Optional[str] = Field(default="moving_average") # e.g., 'moving_average', 'bezier'
    pressure_model: str = Field(default="velocity_based") # e.g., 'constant', 'velocity_based'
    ink_model: str = Field(default="fountain_pen") # e.g., 'constant', 'variable_width'
    
    # Base Stroke properties
    base_stroke_width: float = Field(default=3.0, gt=0.0)
    
    # Export specific
    export: ExportConfig = ExportConfig()
