import math
import torch
import torch.nn as nn
from typing import Dict, Any

from src.interfaces.model import BaseModelInterface
from src.registry import Registry
from src.models.mdn import MDNLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

@Registry.register_model("tiny_transformer")
class CoordinateTransformer(BaseModelInterface, nn.Module):
    """
    Tiny Causal Transformer Decoder with MDN head.
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 256, num_layers: int = 3, num_heads: int = 4, num_mixtures: int = 20):
        nn.Module.__init__(self)
        BaseModelInterface.__init__(self)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Projection layer to map input coordinates to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True
        )
        # We use TransformerEncoder as a causal decoder by passing a causal mask
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # MDN Head
        self.mdn = MDNLayer(hidden_dim, num_mixtures)
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, input_dim)
        """
        seq_len = x.size(1)
        
        # Create causal mask (seq_len, seq_len)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Project and add position encoding
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        
        # Pass through causal transformer
        out = self.transformer(h, mask=causal_mask, is_causal=True)
        
        # MDN output
        pi, mu1, mu2, sigma1, sigma2, rho, eos = self.mdn(out)
        
        return pi, mu1, mu2, sigma1, sigma2, rho, eos

    def train_model(self, dataset: Any, config: Dict[str, Any]) -> None:
        raise NotImplementedError("Handled externally")
        
    def generate(self, text: str = "", style_id: str = None) -> Any:
        raise NotImplementedError("Handled externally")
        
    def evaluate(self, dataset: Any) -> Dict[str, float]:
        raise NotImplementedError("Handled externally")
        
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        
    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
