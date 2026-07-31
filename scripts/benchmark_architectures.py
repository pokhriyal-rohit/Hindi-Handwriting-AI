import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.lstm import CoordinateLSTM
from src.models.transformer import CoordinateTransformer
from src.models.mdn import mdn_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_architectures():
    print("Benchmarking Architectures: CoordinateLSTM vs CoordinateTransformer")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize models
    lstm_model = CoordinateLSTM(input_dim=3, hidden_dim=256, num_layers=2, num_mixtures=20).to(device)
    transformer_model = CoordinateTransformer(input_dim=3, hidden_dim=256, num_layers=3, num_heads=4, num_mixtures=20).to(device)
    
    # 2. Count Parameters
    lstm_params = count_parameters(lstm_model)
    tf_params = count_parameters(transformer_model)
    
    print(f"\n[Parameter Counts]")
    print(f"CoordinateLSTM:        {lstm_params:,}")
    print(f"CoordinateTransformer: {tf_params:,}")
    
    # 3. Dummy Data for Benchmarking
    batch_size = 32
    seq_len = 200
    
    # Dummy inputs: (B, S, 3)
    dummy_input = torch.randn(batch_size, seq_len, 3, device=device)
    # Dummy targets: (B, S, 3)
    dummy_target = torch.randn(batch_size, seq_len, 3, device=device)
    dummy_target[:, :, 2] = torch.sigmoid(dummy_target[:, :, 2]) # pen_state is probability
    
    num_iters = 50
    
    # --- Benchmark LSTM ---
    lstm_model.train()
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=1e-3)
    
    print(f"\n[Benchmarking CoordinateLSTM ({num_iters} iterations)]...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(num_iters):
        optimizer_lstm.zero_grad()
        out, _ = lstm_model(dummy_input)
        pi, mu1, mu2, sigma1, sigma2, rho, eos = out
        loss = mdn_loss(pi, mu1, mu2, sigma1, sigma2, rho, eos, dummy_target)
        loss.backward()
        optimizer_lstm.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_lstm = time.time() - t0
    print(f"LSTM Training Time: {t_lstm:.2f}s ({t_lstm/num_iters*1000:.1f} ms/batch)")
    
    # --- Benchmark Transformer ---
    transformer_model.train()
    optimizer_tf = optim.Adam(transformer_model.parameters(), lr=1e-3)
    
    print(f"\n[Benchmarking CoordinateTransformer ({num_iters} iterations)]...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(num_iters):
        optimizer_tf.zero_grad()
        out = transformer_model(dummy_input)
        pi, mu1, mu2, sigma1, sigma2, rho, eos = out
        loss = mdn_loss(pi, mu1, mu2, sigma1, sigma2, rho, eos, dummy_target)
        loss.backward()
        optimizer_tf.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_tf = time.time() - t0
    print(f"Transformer Training Time: {t_tf:.2f}s ({t_tf/num_iters*1000:.1f} ms/batch)")
    
    print("\n--- Summary ---")
    print(f"LSTM parameter count is {'larger' if lstm_params > tf_params else 'smaller'} than Transformer.")
    print(f"LSTM is {t_tf/t_lstm:.2f}x {'faster' if t_tf > t_lstm else 'slower'} than Transformer per batch.")

if __name__ == "__main__":
    benchmark_architectures()
