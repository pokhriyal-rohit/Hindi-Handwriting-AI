import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    """
    A minimal Sequence-to-Sequence LSTM architecture for handwriting generation.
    Takes token sequences, encodes them, and decodes into trajectory coordinates.
    For Milestone A, it outputs a fixed max sequence length to test MSE/DTW loss bounds.
    """
    def __init__(self, vocab_size: int = 100, embed_dim: int = 64, hidden_dim: int = 128, max_out_len: int = 200):
        super().__init__()
        self.max_out_len = max_out_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Decoder inputs: previous (dx, dy, pen_state) + context vector
        self.decoder = nn.LSTM(3 + (hidden_dim * 2), hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 3)
        
    def forward(self, tokens: torch.Tensor, target_len: int = None):
        """
        tokens: [B, T]
        Outputs coords: [B, target_len, 3]
        """
        B = tokens.size(0)
        if target_len is None:
            target_len = self.max_out_len
            
        # Encode
        embedded = self.embedding(tokens) # [B, T, E]
        enc_out, (hn, cn) = self.encoder(embedded) # hn: [2, B, H]
        
        # Simple context vector (global average pooling over encoder outputs)
        context = enc_out.mean(dim=1).unsqueeze(1) # [B, 1, H*2]
        
        # Decode autoregressively (in a loop) but teacher-forcing is omitted for simplicity in Milestone A
        out_coords = []
        decoder_input = torch.zeros(B, 1, 3, device=tokens.device) # Initial prev_coord = [0,0,0]
        
        # Init decoder hidden state with encoder's final state (simplified)
        h_dec = hn[-1:].contiguous() # [1, B, H]
        c_dec = cn[-1:].contiguous() # [1, B, H]
        
        for step in range(target_len):
            # Concat prev_coord and context
            dec_in = torch.cat([decoder_input, context], dim=-1) # [B, 1, 3 + H*2]
            
            dec_out, (h_dec, c_dec) = self.decoder(dec_in, (h_dec, c_dec))
            
            coord = self.fc_out(dec_out) # [B, 1, 3]
            out_coords.append(coord)
            
            decoder_input = coord # Feedback
            
        return torch.cat(out_coords, dim=1) # [B, target_len, 3]
