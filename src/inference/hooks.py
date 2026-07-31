import logging
from abc import ABC
from typing import Dict, Any, Optional
from src.registry import Registry
from src.datasets.structures import TrajectorySample
from src.inference.result import InferenceResult

logger = logging.getLogger(__name__)

class HookContext:
    """State object passed through hooks, allowing plugins to inspect intermediate values."""
    def __init__(self, text: str):
        self.input_text: str = text
        self.normalized_text: Optional[str] = None
        self.tokens: list[int] = []
        self.trajectory: Optional[TrajectorySample] = None
        self.result: Optional[InferenceResult] = None
        self.metadata: Dict[str, Any] = {}

class BaseHook(ABC):
    """
    Lifecycle hooks. Empty by default so subclasses only override what they need.
    """
    def before_inference(self, ctx: HookContext) -> None: pass
    def after_prediction(self, ctx: HookContext) -> None: pass
    def before_postprocessing(self, ctx: HookContext) -> None: pass
    def after_postprocessing(self, ctx: HookContext) -> None: pass
    def before_rendering(self, ctx: HookContext) -> None: pass
    def after_rendering(self, ctx: HookContext) -> None: pass
    def after_inference(self, ctx: HookContext) -> None: pass

@Registry.register_hook("logging_hook")
class LoggingHook(BaseHook):
    """Simple plugin proving the hook architecture."""
    def before_inference(self, ctx: HookContext) -> None:
        logger.debug(f"Starting pipeline for text: {ctx.input_text}")
        
    def after_prediction(self, ctx: HookContext) -> None:
        logger.debug(f"Prediction generated {len(ctx.metadata.get('raw_outputs', []))} points.")
        
    def after_inference(self, ctx: HookContext) -> None:
        logger.debug(f"Pipeline finished.")
