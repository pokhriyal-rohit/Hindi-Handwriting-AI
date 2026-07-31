from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Point(BaseModel):
    """A single coordinate in a trajectory."""
    x: float
    y: float
    pen_state: int = Field(description="1 for pen down (drawing), 0 for pen up (lift)")
    pressure: Optional[float] = Field(default=None, description="Stylus pressure (0.0 to 1.0)")
    timestamp: Optional[float] = Field(default=None, description="Time in milliseconds since stroke start")

class Stroke(BaseModel):
    """A continuous sequence of points from pen-down to pen-lift."""
    stroke_id: int = Field(default=0)
    points: List[Point]

class DatasetMetadata(BaseModel):
    """Provenance and scaling metadata for the sample."""
    dataset_name: str
    dataset_version: str
    source_url: Optional[str] = None
    license: Optional[str] = None
    is_synthetic: bool = Field(default=False, description="CRITICAL: Must be True for font-generated data.")
    generator_version: Optional[str] = Field(default=None, description="Version of the synthetic generator, if applicable.")
    font_name: Optional[str] = Field(default=None, description="Font used if synthetic.")
    sampling_rate_hz: Optional[float] = Field(default=None, description="Hardware sampling rate for real datasets.")
    normalization: Optional[str] = Field(default="none", description="e.g., 'zero_mean_unit_variance', 'min_max'")
    scaling_factor: Optional[float] = Field(default=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TrajectorySample(BaseModel):
    """The canonical training sample."""
    sample_id: str
    writer_id: str = Field(description="Unique ID for the writer to support Style Encoders.")
    script: str = Field(description="e.g., 'devanagari', 'bengali', 'tamil'")
    language: str = Field(description="e.g., 'hi', 'bn', 'ta'")
    text: str = Field(description="The ground truth text transcript.")
    strokes: List[Stroke]
    metadata: DatasetMetadata
    
    def to_array(self) -> List[List[float]]:
        """Converts the trajectory to a raw list of [x, y, pen_state] arrays for continuous processing."""
        arr = []
        for stroke in self.strokes:
            for pt in stroke.points:
                arr.append([pt.x, pt.y, pt.pen_state])
        return arr
