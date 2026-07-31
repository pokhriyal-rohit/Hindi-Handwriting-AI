from pydantic import BaseModel, Field
from typing import Optional

class InferenceConfig(BaseModel):
    """
    Master configuration for a single Inference Session.
    Determines hardware usage, model checkpoints, and rendering pipelines.
    """
    device: str = Field(default="cpu", description="Hardware device, e.g., 'cpu', 'cuda', 'mps'")
    precision: str = Field(default="float32", description="Precision: 'float32', 'float16', 'bfloat16'")
    random_seed: int = Field(default=42, description="Seed for deterministic sampling")
    
    # Model Versions
    model_name: str = Field(default="default", description="Registered model name to load")
    model_version: str = Field(default="1.0.0", description="Specific checkpoint version or tag")
    
    # Rendering Versions
    renderer_version: str = Field(default="default", description="Renderer configuration name")
    layout_name: str = Field(default="linear", description="Layout engine name")
    pressure_model: str = Field(default="synthetic", description="Pressure model to apply")
    ink_model: str = Field(default="gel_pen", description="Ink renderer model")
    
    # Export formats
    export_formats: list[str] = Field(default_factory=lambda: ["svg"], description="Formats to automatically generate (svg, png)")
    
    # Post Processing
    postprocessors: list[str] = Field(default_factory=lambda: ["coordinate_clamp", "metadata_enricher"], description="Ordered list of post-processing plugins")
    
    # Caching
    enable_cache: bool = Field(default=True, description="Whether to use the Inference Cache system")
    
    # Hooks
    hooks: list[str] = Field(default_factory=lambda: ["logging_hook"], description="Ordered list of lifecycle hook plugins")
