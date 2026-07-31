from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig
from src.renderer.exceptions import ExporterError

class BaseExporter:
    """The canonical Exporter Contract for the Rendering Engine."""
    def __init__(self, config: RenderingConfig):
        self.config = config
        self._initialized = False
        
    def initialize(self) -> None:
        """Allocate resources or verify dependencies."""
        self._initialized = True
        
    def export(self, geom: TrajectorySample, output_path: str) -> None:
        """The core export logic."""
        if not self._initialized:
            raise ExporterError(f"{self.__class__.__name__} was not initialized before export.")
        
    def validate(self, output_path: str) -> bool:
        """Verify the export was successful and valid."""
        import os
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
    def cleanup(self) -> None:
        """Release resources and delete temporaries."""
        pass
