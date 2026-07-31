import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mdn import MDNLayer

class WindowedAttention(nn.Module):
    """
    Graves (2013) style Gaussian attention mechanism for sequence generation.
    Predicts a mixture of Gaussians over the text sequence length to determine alignment.
    This creates a monotonic, localized attention window that moves forward over time.
    """
    def __init__(self, hidden_dim: int, num_mixtures: int = 10):
        super().__init__()
        self.num_mixtures = num_mixtures
        # Predicts alpha (importance), beta (width), kappa (step size)
        self.fc = nn.Linear(hidden_dim, 3 * num_mixtures)
        
    def forward(self, h_t: torch.Tensor, kappa_prev: torch.Tensor, text_context: torch.Tensor):
        """
        Args:
            h_t: (batch, hidden_dim) - current hidden state of the LSTM decoder
            kappa_prev: (batch, num_mixtures) - previous position of attention windows
            text_context: (batch, seq_len, context_dim) - outputs from text encoder
        Returns:
            context_vector: (batch, context_dim)
            kappa_t: (batch, num_mixtures)
            phi: (batch, seq_len) - attention weights for visualization
        """
        B, U, C = text_context.size() # U = text sequence length
        
        # Calculate parameters
        params = self.fc(h_t) # (B, 3*K)
        alpha, beta, kappa_step = torch.split(params, self.num_mixtures, dim=-1)
        
        # Exponential activations to ensure they are > 0
        alpha = torch.exp(alpha) # Importance of each mixture
        beta = torch.exp(beta)   # Width of each window
        kappa_step = torch.exp(kappa_step) # How far to move the window
        
        # Update window positions
        kappa_t = kappa_prev + kappa_step # (B, K)
        
        # Create sequence of text positions (0 to U-1)
        u = torch.arange(U, device=h_t.device, dtype=torch.float32).view(1, 1, U) # (1, 1, U)
        
        # Reshape parameters for broadcasting over U
        alpha = alpha.unsqueeze(2) # (B, K, 1)
        beta = beta.unsqueeze(2)   # (B, K, 1)
        kappa = kappa_t.unsqueeze(2) # (B, K, 1)
        
        # Gaussian attention weights
        phi = alpha * torch.exp(-beta * (kappa - u)**2) # (B, K, U)
        phi = torch.sum(phi, dim=1) # (B, U)
        
        # Context vector is the weighted sum of text encodings
        phi_unsqueezed = phi.unsqueeze(2) # (B, U, 1)
        context_vector = torch.sum(phi_unsqueezed * text_context, dim=1) # (B, C)
        
        return context_vector, kappa_t, phi

class ResidualLSTMLayer(nn.Module):
    """LSTM Layer with Residual Connection and Layer Normalization."""
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        # If input size != hidden size, we need a projection for the residual connection
        self.proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x: torch.Tensor, hx: tuple):
        """
        Args:
            x: (B, input_size)
            hx: (h, c) each (B, hidden_size)
        """
        h_next, c_next = self.lstm(x, hx)
        h_norm = self.layer_norm(h_next)
        h_drop = self.dropout(h_norm)
        # Residual connection
        out = h_drop + self.proj(x)
        return out, (h_next, c_next)

class TrajectoryDecoder(nn.Module):
    """
    Production Trajectory Decoder with Attention, Residuals, LayerNorm, and MDN Head.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_dim: int,
        num_layers: int = 3,
        num_mixtures: int = 20,
        attention_mixtures: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.attention_mixtures = attention_mixtures
        
        # First layer concatenates the coordinate input AND the text context vector
        self.layer_0 = nn.LSTMCell(input_dim + context_dim, hidden_dim)
        
        # Attention mechanism calculates the context vector using the first layer's hidden state
        self.attention = WindowedAttention(hidden_dim, attention_mixtures)
        
        # Subsequent layers use residual connections
        self.layers = nn.ModuleList([
            ResidualLSTMLayer(
                input_size=hidden_dim + context_dim if i == 0 else hidden_dim, 
                hidden_size=hidden_dim, 
                dropout=dropout
            ) for i in range(num_layers - 1)
        ])
        
        # MDN Head takes the final hidden state to predict coordinate distributions
        self.mdn = MDNLayer(hidden_dim, num_mixtures)
        
    def forward_step(self, x_t: torch.Tensor, hx: list, kappa_prev: torch.Tensor, text_context: torch.Tensor):
        """
        Forward pass for a single timestep (used during autoregressive generation).
        Args:
            x_t: (B, input_dim)
            hx: list of (h, c) tuples for each layer
            kappa_prev: (B, attention_mixtures)
            text_context: (B, U, context_dim)
        """
        # We need the previous context vector to feed into layer 0. 
        # But wait, Graves 2013 computes the context vector AFTER layer 0.
        # So we feed the previous context vector or zero for the first step.
        # Alternatively, we just feed x_t and the PREVIOUS context vector.
        
        # Let's simplify: pass x_t and a dummy context vector to Layer 0, then compute new context,
        # then pass to deeper layers.
        # Wait, the simplest way is to pass x_t combined with the PREVIOUS context vector to Layer 0.
        pass
        
    def forward(self, x: torch.Tensor, text_context: torch.Tensor):
        """
        Full teacher-forced forward pass for training.
        Args:
            x: (B, seq_len, input_dim) - coordinates
            text_context: (B, U, context_dim) - text encoder outputs
        Returns:
            mdn_params, attention_weights
        """
        B, S, _ = x.size()
        device = x.device
        
        # Initialize hidden states
        hx = [(torch.zeros(B, self.hidden_dim, device=device), 
               torch.zeros(B, self.hidden_dim, device=device)) for _ in range(self.num_layers)]
        
        kappa_t = torch.zeros(B, self.attention_mixtures, device=device)
        w_t = torch.zeros(B, text_context.size(2), device=device) # previous context vector
        
        outputs = []
        attn_weights = []
        
        for t in range(S):
            x_t = x[:, t, :] # (B, input_dim)
            
            # Layer 0: combines x_t and previous context vector
            inp_0 = torch.cat([x_t, w_t], dim=-1)
            h_0, c_0 = self.layer_0(inp_0, hx[0])
            hx[0] = (h_0, c_0)
            
            # Compute new context vector using Layer 0's hidden state
            w_t, kappa_t, phi_t = self.attention(h_0, kappa_t, text_context)
            attn_weights.append(phi_t)
            
            # Deeper layers
            h_curr = h_0
            for i, layer in enumerate(self.layers):
                # We skip feeding context into every layer for simplicity, but we can pass it to Layer 1
                if i == 0:
                    inp_l = torch.cat([h_curr, w_t], dim=-1)
                else:
                    inp_l = h_curr
                    
                h_curr, hx[i+1] = layer(inp_l, hx[i+1])
                
            outputs.append(h_curr)
            
        outputs = torch.stack(outputs, dim=1) # (B, S, hidden_dim)
        
        # MDN head
        pi, mu1, mu2, sigma1, sigma2, rho, eos = self.mdn(outputs)
        
        return (pi, mu1, mu2, sigma1, sigma2, rho, eos), torch.stack(attn_weights, dim=1)
