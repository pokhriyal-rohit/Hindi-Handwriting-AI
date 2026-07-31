import logging
from typing import Optional, Any
from src.inference.config import InferenceConfig
from src.renderer.config import RenderingConfig
from src.renderer.pipeline import RenderingEngine

logger = logging.getLogger(__name__)

class InferenceSession:
    """
    Long-lived session holding the heavy state: Models in VRAM, Tokenizers, and Renderer pipelines.
    Reused across multiple generation requests.
    """
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.tokenizer: Any = None
        self.predictor: Any = None
        self.renderer: Optional[RenderingEngine] = None
        
        self._initialize()

    def _initialize(self):
        """Loads and initializes all required systems into memory."""
        logger.info(f"Initializing InferenceSession on device: {self.config.device} ({self.config.precision})")
        
        # Load predictor from Registry
        from src.registry import Registry
        model_cls = Registry.get_model(self.config.model_name)
        if model_cls:
            self.predictor = model_cls(self.config)
            self.predictor.load_model()
        else:
            logger.warning(f"Predictor '{self.config.model_name}' not found in registry.")
        
        # Load rendering pipeline
        render_cfg = RenderingConfig()
        # You can override config properties based on InferenceConfig here if needed
        self.renderer = RenderingEngine(render_cfg)
        
        # Load postprocessors
        self.postprocessors = []
        # Ensure validation is imported so registration runs
        import src.inference.postprocessing.validation
        for pp_name in self.config.postprocessors:
            pp_cls = Registry.get_postprocessor(pp_name)
            if pp_cls:
                self.postprocessors.append(pp_cls(self.config))
            else:
                logger.warning(f"Postprocessor '{pp_name}' not found.")
        
        logger.info("InferenceSession initialized successfully.")
        
    def warmup(self):
        """Pre-allocates buffers by running a dummy trace through the predictor."""
        if self.predictor and hasattr(self.predictor, 'warmup'):
            self.predictor.warmup()
            
    def shutdown(self):
        """Frees VRAM and closes handles."""
        logger.info("Shutting down InferenceSession...")
        if self.predictor and hasattr(self.predictor, 'shutdown'):
            self.predictor.shutdown()
        self.predictor = None
        self.renderer = None
