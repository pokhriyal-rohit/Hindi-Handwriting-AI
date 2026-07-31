from abc import ABC, abstractmethod
from typing import List, Any
from src.registry import Registry
from src.inference.config import InferenceConfig

class BasePredictor(ABC):
    """
    Abstract contract for all Inference Predictors (PyTorch, ONNX, TensorRT).
    It is strictly responsible for generating raw continuous coordinate distributions from token indices.
    """
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Loads the model weights into memory (e.g., GPU VRAM)."""
        pass

    @abstractmethod
    def predict(self, tokens: List[int]) -> List[List[float]]:
        """
        Executes a forward pass.
        Takes encoded tokens, outputs a raw list of [dx, dy, pen_state] floats.
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Pre-allocates buffers by running a dummy trace."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Unloads weights and frees hardware memory."""
        pass

@Registry.register_model("dummy_predictor")
class DummyPredictor(BasePredictor):
    """A placeholder predictor for testing the pipeline without PyTorch."""
    def load_model(self) -> None:
        self.model = "dummy_loaded"

    def predict(self, tokens: List[int]) -> List[List[float]]:
        if not self.model:
            raise RuntimeError("Model not loaded.")
        # Return a simple straight line
        # [dx, dy, pen_state]
        out = []
        for _ in tokens:
            out.append([1.0, 0.0, 1.0])
            out.append([1.0, 0.0, 1.0])
        # Lift pen at end
        out[-1][2] = 0.0
        return out

    def warmup(self) -> None:
        self.predict([0])

    def shutdown(self) -> None:
        self.model = None

# Ensure deterministic predictor is registered
import src.inference.predictor.deterministic
