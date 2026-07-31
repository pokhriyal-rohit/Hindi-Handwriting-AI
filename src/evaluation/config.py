from pydantic import BaseModel, Field
from typing import Dict

class EvaluationConfig(BaseModel):
    """Configuration tracking all versions for deterministic benchmark reproducibility."""
    
    evaluation_version: str = Field(default="1.0.0")
    dataset_version: str = Field(default="unknown")
    renderer_version: str = Field(default="unknown")
    representation_version: str = Field(default="unknown")
    model_version: str = Field(default="unknown")
    report_version: str = Field(default="1.0.0")
    
    # Stores metric names to their resolved versions
    metric_versions: Dict[str, str] = Field(default_factory=dict)
