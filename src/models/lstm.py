import torch
import torch.nn as nn
from typing import Dict, Any

from src.interfaces.model import BaseModelInterface
from src.registry import Registry
from src.models.mdn import MDNLayer

@Registry.register_model("tiny_lstm")
class CoordinateLSTM(BaseModelInterface, nn.Module):
    """
    Tiny LSTM with MDN head for unconditioned generative trajectory benchmarking.
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 256, num_layers: int = 2, num_mixtures: int = 20):
        nn.Module.__init__(self)
        BaseModelInterface.__init__(self)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # MDN Head for generative coordinate prediction
        self.mdn = MDNLayer(hidden_dim, num_mixtures)
        
    def forward(self, x: torch.Tensor, hx=None):
        """
        Args:
            x: (batch, seq_len, input_dim) -> normally [dx, dy, pen_state]
            hx: Hidden state
        """
        # x is the sequence of coordinates
        out, hx_new = self.lstm(x, hx)
        
        # Pass the LSTM hidden states to the MDN to predict distributions for the *next* step
        pi, mu1, mu2, sigma1, sigma2, rho, eos = self.mdn(out)
        
        return (pi, mu1, mu2, sigma1, sigma2, rho, eos), hx_new

    def train_model(self, dataset: Any, config: Dict[str, Any]) -> None:
        pass
        
    def generate(self, text: str = "", style_id: str = None) -> Any:
        pass
        
    def evaluate(self, dataset: Any) -> Dict[str, float]:
        pass
        
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        
    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
