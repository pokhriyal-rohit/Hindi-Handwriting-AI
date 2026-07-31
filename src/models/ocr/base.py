import torch
import torch.nn as nn
from typing import Dict, Any

class BaseOCRModel(nn.Module):
    """
    Base class for all OCR models to ensure a unified interface.
    """
    def __init__(self, vocab_size: int, config: Dict[str, Any]):
        super(BaseOCRModel, self).__init__()
        self.vocab_size = vocab_size
        self.config = config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Input: images of shape (B, C, H, W)
        Output: logits of shape (B, T, vocab_size)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def get_output_length(self, input_width: torch.Tensor) -> torch.Tensor:
        """
        Calculates the sequence length (T) of the output based on the input image width.
        This is necessary for CTCLoss.
        """
        raise NotImplementedError("Subclasses must implement get_output_length()")
