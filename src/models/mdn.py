import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNLayer(nn.Module):
    """
    Mixture Density Network (MDN) Layer for continuous coordinate prediction.
    Outputs the parameters of a Gaussian Mixture Model (GMM) for (dx, dy) 
    and a Bernoulli distribution for pen_state.
    
    Format of output:
    - pi: Mixing coefficients (num_mixtures)
    - mu_x, mu_y: Means (num_mixtures)
    - sigma_x, sigma_y: Standard deviations (num_mixtures)
    - rho: Correlation between x and y (num_mixtures)
    - eos: Probability of pen lift/End of Stroke (1)
    """
    def __init__(self, hidden_dim: int, num_mixtures: int = 20):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.hidden_dim = hidden_dim
        
        # Output sizes:
        # pi: K (mixture weights)
        # mu_x, mu_y: K, K (means)
        # sigma_x, sigma_y: K, K (std devs)
        # rho: K (correlations)
        # eos: 1 (pen state probability)
        # Total = 6*K + 1
        
        self.z_pi = nn.Linear(hidden_dim, num_mixtures)
        self.z_mu1 = nn.Linear(hidden_dim, num_mixtures)
        self.z_mu2 = nn.Linear(hidden_dim, num_mixtures)
        self.z_sigma1 = nn.Linear(hidden_dim, num_mixtures)
        self.z_sigma2 = nn.Linear(hidden_dim, num_mixtures)
        self.z_rho = nn.Linear(hidden_dim, num_mixtures)
        self.z_eos = nn.Linear(hidden_dim, 1)
        
    def forward(self, h: torch.Tensor):
        """
        Args:
            h: Hidden state tensor of shape (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        Returns:
            pi, mu1, mu2, sigma1, sigma2, rho, eos
        """
        pi = F.softmax(self.z_pi(h), dim=-1)
        mu1 = self.z_mu1(h)
        mu2 = self.z_mu2(h)
        
        # Sigma must be > 0. using exp() ensures this.
        sigma1 = torch.exp(self.z_sigma1(h))
        sigma2 = torch.exp(self.z_sigma2(h))
        
        # Rho must be between -1 and 1. tanh ensures this.
        rho = torch.tanh(self.z_rho(h))
        
        # EOS probability (Bernoulli). sigmoid ensures [0, 1].
        eos = torch.sigmoid(self.z_eos(h))
        
        return pi, mu1, mu2, sigma1, sigma2, rho, eos

def mdn_loss(pi, mu1, mu2, sigma1, sigma2, rho, eos, target):
    """
    Computes the Negative Log-Likelihood (NLL) loss for the MDN.
    Args:
        target: Ground truth of shape (batch, seq_len, 3) where [:, :, 0]=dx, [:, :, 1]=dy, [:, :, 2]=pen_state
    """
    # Extract targets
    dx = target[..., 0:1] # (B, S, 1)
    dy = target[..., 1:2] # (B, S, 1)
    p_state = target[..., 2:3] # (B, S, 1)
    
    # MDN Loss for spatial coordinates (x, y)
    z_x = (dx - mu1) / sigma1
    z_y = (dy - mu2) / sigma2
    
    z = (z_x ** 2) + (z_y ** 2) - (2 * rho * z_x * z_y)
    num = torch.exp(-z / (2 * (1 - rho ** 2)))
    den = 2 * torch.pi * sigma1 * sigma2 * torch.sqrt(1 - rho ** 2)
    
    prob = num / (den + 1e-8) # shape (B, S, K)
    
    # Weight by mixture coefficients and sum over K
    weighted_prob = torch.sum(prob * pi, dim=-1, keepdim=True) # (B, S, 1)
    
    # Negative log-likelihood of coordinates
    loss_spatial = -torch.log(weighted_prob + 1e-8)
    
    # BCE Loss for pen state
    # We want to predict p_state. p_state=1 is pen down, p_state=0 is pen lift.
    loss_eos = F.binary_cross_entropy(eos, p_state, reduction='none')
    
    return (loss_spatial + loss_eos).mean()
