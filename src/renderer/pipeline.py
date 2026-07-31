import copy
from src.datasets.structures import TrajectorySample
from src.renderer.config import RenderingConfig
from src.registry import Registry
from src.renderer.layout.page import PageLayout
from src.renderer.cache import RendererCache
from src.renderer.exceptions import RendererError, InvalidTrajectoryError, LayoutError, PluginRegistrationError, ExporterError

class RenderingEngine:
    """
    Core functional pipeline for rendering geometry.
    Strictly consumes canonical absolute-coordinate TrajectorySamples.
    """
    def __init__(self, config: RenderingConfig = None):
        self.config = config if config else RenderingConfig()
        self.cache = RendererCache()
        
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
        import time
        from pathlib import Path
        
        times = {}
        t_start = time.perf_counter()
        
        try:
            # 1. Coordinate Reconstruction & Bounding Validation (Ensuring canonical)
            t0 = time.perf_counter()
            if not isinstance(trajectory, TrajectorySample):
                raise InvalidTrajectoryError(f"RenderingEngine strictly requires TrajectorySample, got {type(trajectory)}")
                
            if not trajectory.strokes:
                raise InvalidTrajectoryError("TrajectorySample contains no strokes.")
                
            working_sample = copy.deepcopy(trajectory)
            times['validation'] = time.perf_counter() - t0
                
            # 2. Layout Engine (Margins, Page Positioning)
            t0 = time.perf_counter()
            try:
                working_sample = self._apply_layout(working_sample)
            except Exception as e:
                raise LayoutError(f"Layout engine failed: {e}")
            times['layout'] = time.perf_counter() - t0
            
            # 3. Smoothing
            t0 = time.perf_counter()
            working_sample = self._apply_smoother(working_sample)
            times['smoothing'] = time.perf_counter() - t0
            
            # 4. Pressure Simulation
            t0 = time.perf_counter()
            working_sample = self._apply_pressure(working_sample)
            times['pressure'] = time.perf_counter() - t0
            
            # 5. Ink Simulation (Appearance properties)
            t0 = time.perf_counter()
            working_sample = self._apply_ink(working_sample)
            times['ink'] = time.perf_counter() - t0
            
            # 6. Check Cache
            t0 = time.perf_counter()
            if self.cache.serve_from_cache(trajectory, self.config, format, output_path):
                times['cache_hit'] = True
                times['cache_time'] = time.perf_counter() - t0
                self._log_profile(times, format)
                return
            times['cache_hit'] = False
            times['cache_check'] = time.perf_counter() - t0
            
            # 7. Export Generation
            t0 = time.perf_counter()
            exporter_cls = Registry.get_exporter(format)
            if not exporter_cls:
                raise PluginRegistrationError(f"Exporter plugin for format '{format}' is not registered.")
                
            try:
                exporter = exporter_cls(self.config)
                exporter.initialize()
                exporter.export(working_sample, output_path)
                times['export'] = time.perf_counter() - t0
                
                t0 = time.perf_counter()
                if not exporter.validate(output_path):
                    raise ExporterError(f"Validation failed for '{format}' exporter at '{output_path}'.")
                
                # Save successful export to cache
                self.cache.save_to_cache(trajectory, self.config, format, output_path)
                times['cache_save'] = time.perf_counter() - t0
            except Exception as e:
                raise ExporterError(f"Exporter '{format}' failed during execution: {e}")
            finally:
                if 'exporter' in locals() and hasattr(exporter, 'cleanup'):
                    exporter.cleanup()
            
            times['total'] = time.perf_counter() - t_start
            self._log_profile(times, format)
            
        except RendererError:
            raise
        except Exception as e:
            raise RendererError(f"Unexpected rendering failure: {e}")
            
    def _log_profile(self, times: dict, format: str):
        import os
        log_path = "docs/RENDERER_PROFILE.md"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Only log periodically or write a summary to not thrash the disk
        # We append a simple table row for this run
        header = False
        if not os.path.exists(log_path):
            header = True
            
        with open(log_path, "a") as f:
            if header:
                f.write("# Rendering Engine Profile\n\n")
                f.write("| Format | Layout (s) | Smoothing (s) | Pressure (s) | Ink (s) | Export (s) | Cache Hit | Total (s) |\n")
                f.write("|---|---|---|---|---|---|---|---|\n")
            
            f.write(f"| {format} | {times.get('layout',0):.4f} | {times.get('smoothing',0):.4f} | "
                    f"{times.get('pressure',0):.4f} | {times.get('ink',0):.4f} | {times.get('export',0):.4f} | "
                    f"{times.get('cache_hit', False)} | {times.get('total', 0):.4f} |\n")
