import torch
import torch.nn as nn

class TrajectoryLoss(nn.Module):
    """
    Combined MSE (for dx, dy) and BCE (for pen state) loss.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, seq_lens: torch.Tensor):
        """
        pred: [B, max_len, 3] (dx, dy, pen_logit)
        target: [B, max_len, 3] (dx, dy, pen_state)
        seq_lens: [B] actual sequence lengths
        """
        loss = 0.0
        B = pred.size(0)
        
        for i in range(B):
            l = seq_lens[i].item()
            if l == 0: continue
            
            p = pred[i, :l]
            t = target[i, :l]
            
            # MSE for dx, dy (indices 0, 1)
            coord_loss = self.mse(p[:, 0:2], t[:, 0:2])
            
            # BCE for pen_state (index 2)
            pen_loss = self.bce(p[:, 2], t[:, 2])
            
            loss += (coord_loss + pen_loss)
            
        return loss / B
