import logging
from typing import Dict, Any, List
import time
from src.inference.session import InferenceSession
from src.inference.result import InferenceResult
from src.inference.hooks import HookContext
from src.datasets.structures import TrajectorySample, DatasetMetadata, Stroke, Point

logger = logging.getLogger(__name__)

class InferencePipeline:
    """
    The master generation pipeline.
    Transforms raw input text into rendered handwritten outputs using a strict, step-by-step sequential contract.
    """
    def __init__(self, session: InferenceSession):
        self.session = session
        
    def _preprocess(self, text: str) -> str:
        # Placeholder for unicode/whitespace normalization
        return text.strip()
        
    def _tokenize(self, text: str) -> List[int]:
        # Placeholder for sub-word tokenization
        # Here we just use ascii values
        return [ord(c) for c in text]
        
    def _reconstruct_coordinates(self, raw_outputs: List[List[float]], text: str) -> TrajectorySample:
        """
        Converts differential model outputs [dx, dy, pen_state] into canonical absolute coordinate strokes.
        """
        strokes = []
        current_stroke_pts = []
        
        curr_x, curr_y = 0.0, 0.0
        
        for idx, step in enumerate(raw_outputs):
            dx, dy, pen_state = step[0], step[1], step[2]
            curr_x += dx
            curr_y += dy
            
            # Use 1 for pen down
            is_down = 1 if pen_state > 0.5 else 0
            
            pt = Point(x=curr_x, y=curr_y, pen_state=is_down, timestamp=float(idx * 10))
            current_stroke_pts.append(pt)
            
            # If pen lifts, stroke is complete
            if not is_down and current_stroke_pts:
                strokes.append(Stroke(stroke_id=len(strokes), points=current_stroke_pts))
                current_stroke_pts = []
                
        # Flush any remaining
        if current_stroke_pts:
            strokes.append(Stroke(stroke_id=len(strokes), points=current_stroke_pts))
            
        metadata = DatasetMetadata(
            dataset_name="inference",
            dataset_version="1.0.0",
            is_synthetic=True
        )
        
        return TrajectorySample(
            sample_id="generated",
            writer_id="model",
            script="unknown",
            language="unknown",
            text=text,
            strokes=strokes,
            metadata=metadata
        )
        
    def generate(self, text: str, run_dir: str = None) -> InferenceResult:
        """
        The single entrypoint for generating handwriting.
        If run_dir is provided, it exports to that directory.
        """
        logger.info(f"Generating for text: '{text}'")
        timings = {}
        cache_stats = {"trajectory_hit": False}
        t_start = time.perf_counter()
        
        ctx = HookContext(text)
        for hook in self.session.hooks: hook.before_inference(ctx)
        
        # 1. Preprocessing
        clean_text = self._preprocess(text)
        ctx.normalized_text = clean_text
        
        # --- CACHE CHECK ---
        trajectory = None
        tokens = []
        raw_outputs = []
        
        if self.session.config.enable_cache:
            trajectory = self.session.cache.get_trajectory(clean_text)
            
        if trajectory is not None:
            logger.info("Cache hit for trajectory.")
            cache_stats["trajectory_hit"] = True
        else:
            # 2. Tokenization
            tokens = self._tokenize(clean_text)
            ctx.tokens = tokens
            
            # 3. Model Prediction
            if not self.session.predictor:
                raise RuntimeError("Pipeline execution failed: Predictor is not loaded in Session.")
            raw_outputs = self.session.predictor.predict(tokens)
            ctx.metadata["raw_outputs"] = raw_outputs
            for hook in self.session.hooks: hook.after_prediction(ctx)
            
            # 4. Coordinate Reconstruction
            trajectory = self._reconstruct_coordinates(raw_outputs, clean_text)
            
            # 5. Post Processing
            for hook in self.session.hooks: hook.before_postprocessing(ctx)
            for processor in self.session.postprocessors:
                trajectory = processor.process(trajectory)
            for hook in self.session.hooks: hook.after_postprocessing(ctx)
                
            # Store in cache
            if self.session.config.enable_cache:
                self.session.cache.set_trajectory(clean_text, trajectory)
                
        ctx.trajectory = trajectory
        
        # 6. Layout & Rendering
        export_paths = {}
        for hook in self.session.hooks: hook.before_rendering(ctx)
        
        if self.session.renderer and run_dir:
            import os
            os.makedirs(run_dir, exist_ok=True)
            for fmt in self.session.config.export_formats:
                out_path = os.path.join(run_dir, f"output.{fmt}")
                self.session.renderer.render(trajectory, out_path, format=fmt)
                export_paths[fmt] = out_path
                
        for hook in self.session.hooks: hook.after_rendering(ctx)
        
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000.0
        
        res = InferenceResult(
            input_text=text,
            normalized_text=clean_text,
            trajectory=trajectory,
            configuration=self.session.config,
            timing=timings,
            cache_statistics=cache_stats,
            export_paths=export_paths,
            metadata={
                "raw_tokens": tokens,
                "raw_outputs": raw_outputs
            }
        )
        
        ctx.result = res
        for hook in self.session.hooks: hook.after_inference(ctx)
        return res
