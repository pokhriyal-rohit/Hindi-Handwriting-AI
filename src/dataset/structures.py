from pydantic import BaseModel, Field
from typing import List

class Point(BaseModel):
    """Represents a single coordinate in a handwriting trajectory."""
    x: float
    y: float
    p: int = Field(description="Pen state: 0 for lift, 1 for down/drawing")

class Stroke(BaseModel):
    """Represents a continuous stroke from pen down to pen lift."""
    points: List[Point]

class Trajectory(BaseModel):
    """Represents a complete handwritten word or sequence."""
    strokes: List[Stroke]
    text: str = Field(default="", description="The transcript of the handwriting.")
    writer_id: str = Field(default="", description="Unique identifier for the writer.")
    
    def to_array(self) -> List[List[float]]:
        """Converts the trajectory to a raw list of [x, y, p] arrays for training."""
        arr = []
        for stroke in self.strokes:
            for pt in stroke.points:
                arr.append([pt.x, pt.y, pt.p])
        return arr
