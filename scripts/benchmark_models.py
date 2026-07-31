import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.models.lstm import CoordinateLSTM
from src.models.transformer import CoordinateTransformer
from src.models.mdn import mdn_loss
from src.datasets.synthetic.generator.synthetic_trajectory_generator import SyntheticTrajectoryGenerator
from src.datasets.continuous import ModularCoordinateRepresentation
from src.registry import Registry

def generate_bootstrap_data(num_samples=100, max_len=200):
    """Generates synthetic trajectories and encodes them into tensors."""
    print("Generating Stage 1 Bootstrap Data...")
    generator = SyntheticTrajectoryGenerator(font_path="C:/Windows/Fonts/mangal.ttf")
    # For benchmarking, we just use a dummy text
    samples = []
    for _ in range(num_samples):
        traj = generator._text_to_trajectory("भारत")
        if traj:
            samples.append(traj)
            
    rep = ModularCoordinateRepresentation(features=["dx", "dy", "pen_state"], scaler_name="standard")
    rep.fit(samples)
    
    tensors = []
    for s in samples:
        encoded = rep.encode(s)
        # truncate or pad to max_len for batching simplicity in benchmark
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            pad = np.zeros((max_len - len(encoded), 3))
            encoded = np.vstack([encoded, pad])
        tensors.append(torch.tensor(encoded, dtype=torch.float32))
        
    return torch.stack(tensors), rep

def benchmark_model(model_name: str, model: torch.nn.Module, data: torch.Tensor, device: str = "cpu"):
    """Runs a benchmark loop on the model and records metrics."""
    print(f"\nBenchmarking {model_name}...")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Dummy DataLoader
    batch_size = 16
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Metrics
    metrics = {
        "model_name": model_name,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": device,
        "epochs": 10,
        "batch_size": batch_size,
    }
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    # Autoregressive setup: Input is x_t, target is x_{t+1}
    for epoch in range(metrics["epochs"]):
        model.train()
        total_loss = 0.0
        
        for batch in loader:
            seq = batch[0].to(device) # (B, S, 3)
            
            x = seq[:, :-1, :] # Inputs
            y = seq[:, 1:, :]  # Targets
            
            optimizer.zero_grad()
            
            if "lstm" in model_name.lower():
                (pi, mu1, mu2, sigma1, sigma2, rho, eos), _ = model(x)
            else:
                pi, mu1, mu2, sigma1, sigma2, rho, eos = model(x)
                
            loss = mdn_loss(pi, mu1, mu2, sigma1, sigma2, rho, eos, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
    end_time = time.time()
    
    metrics["training_time_sec"] = end_time - start_time
    metrics["samples_per_sec"] = (len(data) * metrics["epochs"]) / metrics["training_time_sec"]
    metrics["final_loss"] = total_loss / len(loader)
    
    if torch.cuda.is_available():
        metrics["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        metrics["peak_memory_mb"] = 0.0
        
    return metrics

def run_benchmarks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on {device}")
    
    try:
        data, rep = generate_bootstrap_data(num_samples=64, max_len=150)
    except Exception as e:
        print(f"Skipping benchmarking: {e}")
        return
        
    models = {
        "Tiny_LSTM": CoordinateLSTM(input_dim=3, hidden_dim=128, num_layers=2),
        "Tiny_Transformer": CoordinateTransformer(input_dim=3, hidden_dim=128, num_layers=2, num_heads=4)
    }
    
    results = []
    for name, model in models.items():
        metrics = benchmark_model(name, model, data, device)
        metrics["representation_stats"] = rep.statistics()
        results.append(metrics)
        
    # Save reproducible benchmark output
    output_path = Path("benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nBenchmarks complete. Saved to {output_path}")

if __name__ == "__main__":
    run_benchmarks()
