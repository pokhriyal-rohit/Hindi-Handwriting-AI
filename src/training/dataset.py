import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import math

class SyntheticTrajectoryDataset(Dataset):
    """
    Generates deterministic synthetic trajectories for 5 fixed Hindi words.
    Provides ground truth (tokens, coords, lengths) for Milestone A training.
    """
    def __init__(self):
        # 5 fixed words for deterministic overfitting test
        self.words = ["नमस्ते", "दुनिया", "भारत", "अक्षर", "भाषा"]
        
        self.samples = []
        for word in self.words:
            tokens = torch.tensor([ord(c) for c in word], dtype=torch.long)
            
            # Generate deterministic geometry (matching DeterministicHindiPredictor)
            coords = []
            for token in tokens.tolist():
                length = (token % 10) + 5
                for step in range(length):
                    dx = 2.0
                    dy = math.sin(step) * 2.0
                    pen_state = 1.0 if step < length - 1 else 0.0 # Pen down, up at end of char
                    coords.append([dx, dy, pen_state])
                # Inter-character spacing
                coords.append([5.0, 0.0, 0.0])
                
            self.samples.append((tokens, torch.tensor(coords, dtype=torch.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

def synthetic_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates (tokens, coords) pairs into padded batches.
    Returns:
        tokens_padded: [batch_size, max_seq_len]
        tokens_lens: [batch_size]
        coords_padded: [batch_size, max_traj_len, 3]
        coords_lens: [batch_size]
    """
    # Sort batch by token length descending (useful for RNNs)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    tokens = [x[0] for x in batch]
    coords = [x[1] for x in batch]
    
    tokens_lens = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    coords_lens = torch.tensor([len(c) for c in coords], dtype=torch.long)
    
    # Pad tokens with 0
    from torch.nn.utils.rnn import pad_sequence
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    
    # Pad coords with [0,0,0]
    coords_padded = pad_sequence(coords, batch_first=True, padding_value=0.0)
    
    return tokens_padded, tokens_lens, coords_padded, coords_lens
