import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    """
    Bidirectional GRU for encoding discrete text sequences.
    Outputs continuous context vectors for the TrajectoryDecoder.
    """
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int = 128, 
        hidden_dim: int = 256, 
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # We use GRU for the text encoder as it is often sufficient for character-level dependencies 
        # and slightly faster than LSTM. The output decoder will be an LSTM.
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # The output of a bidirectional RNN has size 2 * hidden_dim
        self.output_dim = hidden_dim * 2
        
    def forward(self, text_tokens: torch.Tensor, text_lengths: torch.Tensor = None):
        """
        Args:
            text_tokens: (batch_size, text_seq_len) integer tokens.
            text_lengths: (batch_size,) lengths of each text sequence for packing.
            
        Returns:
            outputs: (batch_size, text_seq_len, 2 * hidden_dim) contextualized text features.
            hidden: (num_layers * 2, batch_size, hidden_dim) final hidden states.
        """
        x = self.embedding(text_tokens) # (B, S, E)
        
        # If lengths are provided, we pack the sequence to avoid processing padding tokens
        if text_lengths is not None:
            # Enforce CPU for lengths array in pack_padded_sequence
            text_lengths_cpu = text_lengths.cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, text_lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, hidden = self.rnn(packed_x)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            outputs, hidden = self.rnn(x)
            
        return outputs, hidden
