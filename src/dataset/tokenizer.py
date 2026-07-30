import numpy as np
from typing import List
from .structures import Point, Stroke, Trajectory

class CoordinateTokenizer:
    """
    Quantizes continuous coordinates into discrete tokens for Autoregressive Transformers.
    Treats (X, Y) coordinates and PEN states as discrete vocabulary.
    """
    def __init__(self, grid_size: int = 256):
        self.grid_size = grid_size
        self.pad_token = 0
        self.eos_token = 1
        self.pen_lift_token = 2
        # Tokens 0-2 are reserved. Coordinates start from 3.
        self.coord_offset = 3
        self.vocab_size = self.grid_size + self.coord_offset

    def quantize(self, value: float, min_val: float, max_val: float) -> int:
        """Maps a continuous value to a discrete bin [0, grid_size-1]."""
        if max_val == min_val:
            return self.coord_offset
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))
        bin_idx = int(normalized * (self.grid_size - 1))
        return bin_idx + self.coord_offset

    def dequantize(self, token: int, min_val: float, max_val: float) -> float:
        """Maps a discrete token back to a continuous value."""
        if token < self.coord_offset:
            return 0.0 # Not a coordinate token
        bin_idx = token - self.coord_offset
        normalized = bin_idx / float(self.grid_size - 1)
        return min_val + normalized * (max_val - min_val)

    def tokenize_trajectory(self, traj: Trajectory) -> List[int]:
        """
        Converts a Trajectory to a 1D sequence of tokens.
        Format: [X1, Y1, X2, Y2, PEN_LIFT, X3, Y3, ..., EOS]
        """
        all_pts = traj.to_array()
        if not all_pts:
            return [self.eos_token]
            
        arr = np.array(all_pts)
        min_x, max_x = np.min(arr[:, 0]), np.max(arr[:, 0])
        min_y, max_y = np.min(arr[:, 1]), np.max(arr[:, 1])
        
        tokens = []
        for stroke in traj.strokes:
            for pt in stroke.points:
                x_tok = self.quantize(pt.x, min_x, max_x)
                y_tok = self.quantize(pt.y, min_y, max_y)
                tokens.extend([x_tok, y_tok])
                
                # If it's the last point of a stroke (p=0), add pen lift token
                # In our normalized structure, p=0 means lift.
                if pt.p == 0:
                    tokens.append(self.pen_lift_token)
                    
        tokens.append(self.eos_token)
        return tokens
