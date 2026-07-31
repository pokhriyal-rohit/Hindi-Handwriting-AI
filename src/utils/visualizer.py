import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.datasets.structures import TrajectorySample

class TrajectoryVisualizer:
    """
    Visual Debugging Utilities for Coordinate Representations.
    Functions to visualize original, encoded, decoded, and diff trajectories.
    """
    
    @staticmethod
    def plot_trajectory(traj: TrajectorySample, ax, title: str = "Trajectory", color='blue'):
        """Plots a single trajectory on the given matplotlib axis."""
        for stroke in traj.strokes:
            x = [pt.x for pt in stroke.points]
            y = [-pt.y for pt in stroke.points] # Invert Y for correct rendering
            ax.plot(x, y, color=color, linewidth=2)
        ax.set_title(title)
        ax.axis('equal')
        ax.axis('off')

    @staticmethod
    def visualize_reconstruction(
        original: TrajectorySample, 
        encoded: np.ndarray, 
        decoded: TrajectorySample, 
        save_path: str | Path
    ):
        """
        Renders a 4-panel debug visualization:
        Original | Encoded Scatter | Decoded | Overlay & Difference
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Original
        TrajectoryVisualizer.plot_trajectory(original, axes[0, 0], "Original Trajectory", color='black')
        
        # 2. Encoded Representation
        if encoded.size > 0:
            axes[0, 1].plot(encoded[:, 0], label="Feature 0", color="red", alpha=0.7)
            axes[0, 1].plot(encoded[:, 1], label="Feature 1", color="green", alpha=0.7)
            axes[0, 1].set_title("Encoded Representation (Features 0 & 1)")
            axes[0, 1].legend()
        else:
            axes[0, 1].set_title("Encoded Representation (Empty)")
            
        # 3. Decoded
        if decoded:
            TrajectoryVisualizer.plot_trajectory(decoded, axes[1, 0], "Decoded Trajectory", color='blue')
        else:
            axes[1, 0].set_title("Decoded Trajectory (None)")
            
        # 4. Overlay & Difference
        if decoded:
            TrajectoryVisualizer.plot_trajectory(original, axes[1, 1], "Overlay (Black=Orig, Red=Decoded)", color='black')
            TrajectoryVisualizer.plot_trajectory(decoded, axes[1, 1], "Overlay (Black=Orig, Red=Decoded)", color='red')
            # Plot error lines between points if lengths match
            orig_pts = []
            for stroke in original.strokes:
                for pt in stroke.points:
                    orig_pts.append([pt.x, -pt.y])
            dec_pts = []
            for stroke in decoded.strokes:
                for pt in stroke.points:
                    dec_pts.append([pt.x, -pt.y])
                    
            min_len = min(len(orig_pts), len(dec_pts))
            for i in range(min_len):
                ox, oy = orig_pts[i]
                dx, dy = dec_pts[i]
                axes[1, 1].plot([ox, dx], [oy, dy], color='orange', alpha=0.3, linewidth=1)
        else:
            axes[1, 1].set_title("Overlay (No Decoded Data)")
            
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
