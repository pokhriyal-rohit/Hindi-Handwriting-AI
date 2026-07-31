from typing import List
import math
from src.registry import Registry
from src.inference.predictor.base import BasePredictor

@Registry.register_model("deterministic_hindi")
class DeterministicHindiPredictor(BasePredictor):
    """
    A deterministic mock predictor that always generates a reproducible geometric trajectory.
    Specifically designed to provide stable test outputs for end-to-end integration.
    """
    def load_model(self) -> None:
        self.model = "deterministic_loaded"
        
    def predict(self, tokens: List[int]) -> List[List[float]]:
        if not self.model:
            raise RuntimeError("Model not loaded.")
            
        # Detect the specific "नमस्ते" token sequence (in ord: 2344, 2350, 2360, 2381, 2340, 2375)
        # But we make it generalized so any input generates a deterministic but stable shape.
        
        out = []
        for i, token in enumerate(tokens):
            # Generate a small sine-wave loop for each character based on its ord value
            length = (token % 10) + 5
            for step in range(length):
                dx = 2.0
                dy = math.sin(step) * 2.0
                pen_state = 1.0 # Pen down
                out.append([dx, dy, pen_state])
                
            # Inter-character spacing (pen up)
            out.append([5.0, 0.0, 0.0])
            
        return out

    def warmup(self) -> None:
        self.predict([2344])

    def shutdown(self) -> None:
        self.model = None
