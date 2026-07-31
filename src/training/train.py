import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.training.dataset import SyntheticTrajectoryDataset, CustomTrajectoryDataset, synthetic_collate_fn
from src.models.baseline_lstm import BaselineLSTM
from src.training.loss import TrajectoryLoss
from src.training.experiment import ExperimentTracker
from src.training.utils import tensor_to_trajectory
from src.renderer.pipeline import RenderingEngine
from src.renderer.config import RenderingConfig

def compute_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_model(epochs: int = 100, exp_id: str = None, resume_checkpoint: str = None, val_split: float = 0.1, batch_size: int = 32):
    tracker = ExperimentTracker(exp_id=exp_id)

    import os
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "custom_hindi"))
    full_dataset = CustomTrajectoryDataset(data_dir=data_dir)

    # --- Validation split ---
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    if n_total < 2:
        raise ValueError(f"Dataset has only {n_total} sample(s) — need at least 2 to create a validation split.")
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)  # fixed seed for reproducibility
    )
    print(f"Dataset split: {n_train} train / {n_val} validation (seed=42)")

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=synthetic_collate_fn)
    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, collate_fn=synthetic_collate_fn)

    model = BaselineLSTM(vocab_size=5000, embed_dim=64, hidden_dim=128, max_out_len=200)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = TrajectoryLoss()
    
    start_epoch = 1
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from {resume_checkpoint} at epoch {start_epoch}")
        
    tracker.save_config({
        "epochs": epochs,
        "batch_size": batch_size,
        "val_split": val_split,
        "n_train": n_train,
        "n_val": n_val,
        "val_seed": 42,
        "vocab_size": 5000,
        "lr": 1e-3,
        "model": "BaselineLSTM"
    })
    
    renderer = RenderingEngine(RenderingConfig())
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        
        t0 = time.time()
        for tokens, t_lens, coords, c_lens in dataloader:
            optimizer.zero_grad()
            
            # Forward
            preds = model(tokens, target_len=coords.size(1))
            
            # Loss
            loss = loss_fn(preds, coords, c_lens)
            
            # Backward
            loss.backward()
            grad_norm = compute_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        epoch_time = time.time() - t0
        avg_loss = total_loss / len(dataloader)

        # --- Validation pass ---
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for tokens, t_lens, coords, c_lens in val_loader:
                preds = model(tokens, target_len=coords.size(1))
                val_loss_total += loss_fn(preds, coords, c_lens).item()
        avg_val_loss = val_loss_total / len(val_loader)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Grad: {grad_norm:.2f} | Time: {epoch_time:.2f}s")

        tracker.log_epoch(epoch, {
            "loss": avg_loss,
            "val_loss": avg_val_loss,
            "grad_norm": grad_norm,
            "lr": optimizer.param_groups[0]['lr'],
            "time_sec": epoch_time
        })
        
        # Save checkpoint periodically and generate SVGs
        if epoch % 10 == 0 or epoch == epochs:
            # Checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, tracker.get_checkpoint_path(epoch))
            
            # Generate qualitative outputs
            model.eval()
            with torch.no_grad():
                # Pick the first batch
                tokens, t_lens, coords, c_lens = next(iter(dataloader))
                preds = model(tokens, target_len=coords.size(1))
                
                # Take first sequence
                pred_traj = tensor_to_trajectory(preds[0, :c_lens[0]])
                target_traj = tensor_to_trajectory(coords[0, :c_lens[0]])
                
                # Render Prediction
                pred_path = tracker.get_path("predictions", f"epoch_{epoch:03d}.svg")
                renderer.render(pred_traj, pred_path, format="svg")
                
                # Render Target (Ground Truth)
                target_path = tracker.get_path("predictions", f"epoch_{epoch:03d}_target.svg")
                renderer.render(target_traj, target_path, format="svg")
                
                # Optional: Overlay (skipping explicit SVG merge for brevity, can be added later)
                
    print(f"Training completed. Outputs saved to {tracker.exp_dir}")
    return tracker.exp_dir
