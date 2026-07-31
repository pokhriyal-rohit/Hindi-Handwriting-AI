from abc import ABC, abstractmethod
from src.datasets.structures import TrajectorySample
from pathlib import Path

class Renderer(ABC):
    """
    Abstract interface for handwriting renderers.
    """
    
    @abstractmethod
    def render_svg(self, trajectory: TrajectorySample, output_path: str | Path) -> None:
        """Renders the trajectory to an SVG file."""
        pass
        
    @abstractmethod
    def render_png(self, trajectory: TrajectorySample, output_path: str | Path) -> None:
        """Renders the trajectory to a PNG file."""
        pass
        
    @abstractmethod
    def render_pdf(self, trajectory: TrajectorySample, output_path: str | Path) -> None:
        """Renders the trajectory to a PDF file."""
        pass
        
    @abstractmethod
    def animate(self, trajectory: TrajectorySample, output_path: str | Path) -> None:
        """Generates an animation (e.g., GIF/MP4) of the writing process."""
        pass
