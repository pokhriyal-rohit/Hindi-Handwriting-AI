import copy
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig
from src.registry import Registry
from src.renderer.layout.page import PageLayout
from src.renderer.exceptions import RendererError, InvalidTrajectoryError, LayoutError, PluginRegistrationError, ExporterError

class RenderingEngine:
    """
    Core functional pipeline for rendering geometry.
    Strictly consumes canonical absolute-coordinate TrajectorySamples.
    """
    def __init__(self, config: RenderingConfig = None):
        self.config = config if config else RenderingConfig()
        
    def _apply_layout(self, sample: TrajectorySample) -> TrajectorySample:
        layout_cls = Registry.get_layout(self.config.layout_model)
        if not layout_cls:
            # Fallback to default page layout
            layout = PageLayout(self.config)
            return layout.apply(sample)
        layout = layout_cls(self.config)
        return layout.apply(sample)
        
    def _apply_smoother(self, sample: TrajectorySample) -> TrajectorySample:
        if not self.config.smoothing:
            return sample
        
        smoother_cls = Registry.get_smoother(self.config.smoothing)
        if not smoother_cls:
            print(f"Warning: Smoother '{self.config.smoothing}' not found. Skipping.")
            return sample
            
        smoother = smoother_cls(self.config)
        return smoother.apply(sample)
        
    def _apply_pressure(self, sample: TrajectorySample) -> TrajectorySample:
        if not self.config.pressure_model:
            return sample
            
        pressure_cls = Registry.get_pressure_model(self.config.pressure_model)
        if not pressure_cls:
            print(f"Warning: Pressure model '{self.config.pressure_model}' not found. Skipping.")
            return sample
            
        pressure_model = pressure_cls(self.config)
        return pressure_model.apply(sample)
        
    def _apply_ink(self, sample: TrajectorySample) -> TrajectorySample:
        """Ink models modify appearance attributes (width, color, opacity). Geometry is unmodified."""
        if not self.config.ink_model:
            return sample
            
        ink_cls = Registry.get_ink_model(self.config.ink_model)
        if not ink_cls:
            print(f"Warning: Ink model '{self.config.ink_model}' not found. Skipping.")
            return sample
            
        ink_model = ink_cls(self.config)
        return ink_model.apply(sample)
        
    def render(self, trajectory: TrajectorySample, output_path: str, format: str = "svg") -> None:
        """
        Full pipeline from absolute canonical TrajectorySample to output file.
        """
        try:
            # 1. Coordinate Reconstruction & Bounding Validation (Ensuring canonical)
            if not isinstance(trajectory, TrajectorySample):
                raise InvalidTrajectoryError(f"RenderingEngine strictly requires TrajectorySample, got {type(trajectory)}")
                
            if not trajectory.strokes:
                raise InvalidTrajectoryError("TrajectorySample contains no strokes.")
                
            # We work on a copy to avoid mutating the original
            working_sample = copy.deepcopy(trajectory)
                
            # 2. Layout Engine (Margins, Page Positioning)
            try:
                working_sample = self._apply_layout(working_sample)
            except Exception as e:
                raise LayoutError(f"Layout engine failed: {e}")
            
            # 3. Interpolation (Optional/Future)
            
            # 4. Smoothing
            working_sample = self._apply_smoother(working_sample)
            
            # 5. Pressure Simulation
            working_sample = self._apply_pressure(working_sample)
            
            # 6. Ink Simulation (Appearance properties)
            working_sample = self._apply_ink(working_sample)
            
            # 7. Check Cache (Future/Optional before heavy rasterization)
            
            # 8. Export Generation
            exporter_cls = Registry.get_exporter(format)
            if not exporter_cls:
                raise PluginRegistrationError(f"Exporter plugin for format '{format}' is not registered.")
                
            try:
                exporter = exporter_cls(self.config)
                exporter.initialize()
                exporter.export(working_sample, output_path)
                if not exporter.validate(output_path):
                    raise ExporterError(f"Validation failed for '{format}' exporter at '{output_path}'.")
            except Exception as e:
                raise ExporterError(f"Exporter '{format}' failed during execution: {e}")
            finally:
                if 'exporter' in locals() and hasattr(exporter, 'cleanup'):
                    exporter.cleanup()
                
        except RendererError:
            raise
        except Exception as e:
            raise RendererError(f"Unexpected rendering failure: {e}")
