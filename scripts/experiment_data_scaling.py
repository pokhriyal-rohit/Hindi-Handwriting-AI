import os
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.datasets.online_dataset import CanonicalTrajectoryDataset, synthetic_collate_fn
from src.models.baseline_lstm import BaselineLSTM
from src.training.loss import TrajectoryLoss
from src.evaluation.metrics.trajectory import DTWMetric
from src.training.utils import tensor_to_trajectory

def compute_val_dtw(model, val_loader, max_samples=20, device="cpu"):
    dtw_metric = DTWMetric()
    model.eval()
    distances = []
    samples = 0
    with torch.no_grad():
        for tokens, t_lens, coords, c_lens in val_loader:
            if samples >= max_samples:
                break
            tokens, coords = tokens.to(device), coords.to(device)
            preds = model(tokens, target_len=coords.size(1))
            
            for b in range(preds.size(0)):
                if samples >= max_samples:
                    break
                pred_traj = tensor_to_trajectory(preds[b, :c_lens[b]].cpu())
                tgt_traj = tensor_to_trajectory(coords[b, :c_lens[b]].cpu())
                try:
                    res = dtw_metric.evaluate(pred_traj, tgt_traj)
                    if res["dtw_distance"] != float('inf'):
                        distances.append(res["dtw_distance"])
                except Exception as e:
                    pass
                samples += 1
    
    return np.mean(distances) if distances else float('inf')

def run_scaling_experiment():
    print("Running Data Scaling Experiment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    canonical_dir = os.path.join("data", "canonical", "online")
    train_dir = os.path.join(canonical_dir, "train")
    val_dir = os.path.join(canonical_dir, "validation")
    
    full_train_dataset = CanonicalTrajectoryDataset(data_dir=train_dir)
    val_dataset = CanonicalTrajectoryDataset(data_dir=val_dir)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=synthetic_collate_fn)
    
    fractions = [0.25, 0.50, 0.75, 1.0]
    epochs = 3
    batch_size = 16
    
    results = []
    
    for frac in fractions:
        print(f"\n--- Training on {frac*100:.0f}% of Data ---")
        
        # Subsample
        num_samples = int(len(full_train_dataset) * frac)
        indices = np.random.choice(len(full_train_dataset), num_samples, replace=False)
        subset = Subset(full_train_dataset, indices)
        
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=synthetic_collate_fn)
        
        # Reinitialize model
        model = BaselineLSTM(vocab_size=5000, embed_dim=64, hidden_dim=128, max_out_len=200).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = TrajectoryLoss().to(device)
        
        # Train for short epochs
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for tokens, t_lens, coords, c_lens in train_loader:
                tokens, coords, c_lens = tokens.to(device), coords.to(device), c_lens.to(device)
                optimizer.zero_grad()
                preds = model(tokens, target_len=coords.size(1))
                loss = loss_fn(preds, coords, c_lens)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")
            
        train_time = time.time() - t0
        
        # Evaluate DTW
        mean_dtw = compute_val_dtw(model, val_loader, max_samples=40, device=device)
        print(f"Validation DTW: {mean_dtw:.2f} | Time: {train_time:.1f}s")
        
        results.append({
            "fraction": frac,
            "samples": num_samples,
            "epochs": epochs,
            "mean_dtw": float(mean_dtw),
            "train_time_sec": train_time
        })
        
    print("\n--- Scaling Experiment Results ---")
    for r in results:
        print(f"Data: {r['fraction']*100:3.0f}% ({r['samples']} samples) | DTW: {r['mean_dtw']:.2f}")
        
    with open("data_scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to data_scaling_results.json")

if __name__ == "__main__":
    run_scaling_experiment()
