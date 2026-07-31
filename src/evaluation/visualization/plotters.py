import os
from typing import Optional
from src.datasets.structures import TrajectorySample
from src.evaluation.metrics.trajectory import _flatten_points

def plot_trajectory_overlay(prediction: TrajectorySample, target: TrajectorySample, output_path: str):
    """
    Plots the prediction and target over each other using matplotlib to visually inspect
    the geometric alignment and DTW mapping (abstractly).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot target in gray
    tgt_pts = _flatten_points(target)
    if tgt_pts:
        tx, ty = zip(*tgt_pts)
        plt.plot(tx, [-y for y in ty], color='gray', linewidth=4, alpha=0.5, label='Target')
        
    # Plot prediction in blue
    pred_pts = _flatten_points(prediction)
    if pred_pts:
        px, py = zip(*pred_pts)
        plt.plot(px, [-y for y in py], color='blue', linewidth=2, label='Prediction')
        
    plt.title("Trajectory Overlay Comparison")
    plt.legend()
    plt.axis('equal')
    plt.savefig(output_path)
    plt.close()
