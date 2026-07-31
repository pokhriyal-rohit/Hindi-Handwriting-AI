import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from src.training.utils import tensor_to_trajectory, setup_ddp, cleanup_ddp
from src.datasets.online_dataset import SyntheticTrajectoryDataset, CustomTrajectoryDataset, CanonicalTrajectoryDataset, synthetic_collate_fn
from src.models.baseline_lstm import BaselineLSTM
from src.training.loss import TrajectoryLoss
from src.training.experiment import ExperimentTracker
from src.renderer.pipeline import RenderingEngine
from src.renderer.config import RenderingConfig

def compute_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def _compute_val_geometry(model, val_loader, max_samples: int) -> dict:
    """
    Runs DTW and EndpointError on up to `max_samples` validation predictions.
    Returns an empty dict if fastdtw/scipy are not installed.
    The model must already be in eval() mode before calling this.
    """
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        import numpy as np
    except ImportError:
        return {}  # Metrics silently skipped if optional deps absent

    from src.evaluation.metrics.trajectory import DTWMetric, EndpointErrorMetric
    dtw_metric = DTWMetric()
    ee_metric  = EndpointErrorMetric()

    dtw_scores: list = []
    ee_scores:  list = []
    n_evaluated = 0

    device = next(model.parameters()).device
    with torch.no_grad():
        for tokens, t_lens, coords, c_lens in val_loader:
            if n_evaluated >= max_samples:
                break
            tokens, coords = tokens.to(device), coords.to(device)
            preds = model(tokens, target_len=coords.size(1))
            batch_size = tokens.size(0)
            for i in range(batch_size):
                if n_evaluated >= max_samples:
                    break
                length = c_lens[i].item()
                pred_traj   = tensor_to_trajectory(preds[i, :length])
                target_traj = tensor_to_trajectory(coords[i, :length])
                try:
                    dtw_res = dtw_metric.evaluate(pred_traj, target_traj)
                    ee_res  = ee_metric.evaluate(pred_traj, target_traj)
                    if dtw_res.get("dtw_distance") != float("inf"):
                        dtw_scores.append(dtw_res["dtw_distance"])
                    if ee_res.get("endpoint_error") != float("inf"):
                        ee_scores.append(ee_res["endpoint_error"])
                except Exception:
                    pass  # Don't let a metric failure abort training
                n_evaluated += 1

    result = {}
    if dtw_scores:
        result["dtw_mean"]   = float(sum(dtw_scores) / len(dtw_scores))
        result["dtw_median"] = float(sorted(dtw_scores)[len(dtw_scores) // 2])
        result["dtw_n"]      = len(dtw_scores)
    if ee_scores:
        result["endpoint_error_mean"] = float(sum(ee_scores) / len(ee_scores))
        result["endpoint_error_n"]    = len(ee_scores)
    return result

def train_model(
    epochs: int = 100,
    exp_id: str = None,
    resume_checkpoint: str = None,
    val_split: float = 0.1,
    batch_size: int = 32,
    force_train: bool = False,
    eval_every: int = 10,
    max_eval_samples: int = 20,
):
    is_ddp, local_rank, global_rank = setup_ddp()
    is_primary = (global_rank == 0)

    tracker = ExperimentTracker(exp_id=exp_id) if is_primary else None

    # We now strictly train on the canonical online dataset
    canonical_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "canonical", "online"))
    
    train_dir = os.path.join(canonical_dir, "train")
    val_dir   = os.path.join(canonical_dir, "validation")
    
    if not os.path.exists(train_dir):
        raise RuntimeError(f"Canonical train directory missing: {train_dir}\nRun scripts/build_canonical_dataset.py first.")

    train_dataset = CanonicalTrajectoryDataset(data_dir=train_dir)
    val_dataset   = CanonicalTrajectoryDataset(data_dir=val_dir)
    
    n_train = len(train_dataset)
    n_val   = len(val_dataset)
    if is_primary:
        print(f"Canonical Dataset loaded: {n_train} train / {n_val} validation")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), collate_fn=synthetic_collate_fn, sampler=train_sampler)
    
    if n_val > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=synthetic_collate_fn, sampler=val_sampler)
    else:
        # If val is empty (e.g. single writer placed fully in train), handle gracefully
        val_loader = []

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_primary:
        print(f"Training on device: {device} | DDP: {is_ddp}")

    model = BaselineLSTM(vocab_size=5000, embed_dim=64, hidden_dim=128, max_out_len=200).to(device)
    if is_ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = TrajectoryLoss().to(device)
    
    start_epoch = 1
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        state_dict = ckpt['model_state_dict']
        if is_ddp:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if is_primary:
            print(f"Resumed from {resume_checkpoint} at epoch {start_epoch}")
        
    if is_primary:
        tracker.save_config({
            "epochs": epochs,
            "batch_size": batch_size,
            "val_split": val_split,
            "n_train": n_train,
            "n_val": n_val,
            "val_seed": 42,
            "eval_every": eval_every,
            "max_eval_samples": max_eval_samples,
            "vocab_size": 5000,
            "lr": 1e-3,
            "model": "BaselineLSTM"
        })
    
    renderer = RenderingEngine(RenderingConfig())
    
    t0_total = time.time()
    for epoch in range(start_epoch, epochs + 1):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        
        t0 = time.time()
        for tokens, t_lens, coords, c_lens in dataloader:
            tokens, coords = tokens.to(device), coords.to(device)
            c_lens = c_lens.to(device)
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

        # ── Validation pass (loss) ────────────────────────────────────────────
        val_loss_total = 0.0
        avg_val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for tokens, t_lens, coords, c_lens in val_loader:
                    tokens, coords, c_lens = tokens.to(device), coords.to(device), c_lens.to(device)
                    preds = model(tokens, target_len=coords.size(1))
                    val_loss_total += loss_fn(preds, coords, c_lens).item()
            avg_val_loss = val_loss_total / len(val_loader)

        # ── Per-epoch geometry metrics on val set ─────────────────────────────
        geo_metrics: dict = {}
        if is_primary and val_loader and (epoch % eval_every == 0 or epoch == epochs):
            geo_metrics = _compute_val_geometry(
                model.module if is_ddp else model, val_loader, max_eval_samples
            )

        # ── Logging ───────────────────────────────────────────────────────────
        if is_primary:
            geo_str = ""
            if geo_metrics:
                dtw = geo_metrics.get("dtw_mean")
                ee  = geo_metrics.get("endpoint_error_mean")
                geo_str = f" | DTW: {dtw:.1f}" if dtw is not None else ""
                geo_str += f" | EE: {ee:.1f}" if ee is not None else ""
    
            print(
                f"Epoch {epoch:03d} | "
                f"Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f} | "
                f"Grad: {grad_norm:.2f}{geo_str} | {epoch_time:.2f}s"
            )
    
            epoch_log = {
                "loss":      avg_loss,
                "val_loss":  avg_val_loss,
                "grad_norm": grad_norm,
                "lr":        optimizer.param_groups[0]["lr"],
                "time_sec":  epoch_time,
            }
            if geo_metrics:
                epoch_log.update(geo_metrics)
            tracker.log_epoch(epoch, epoch_log)
            
            # Checkpoints logic (latest, best_loss, etc.)
            state_dict = model.module.state_dict() if is_ddp else model.state_dict()
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            
            # Save latest
            ckpt_dir = os.path.join(tracker.exp_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(ckpt_data, os.path.join(ckpt_dir, "latest.pt"))
            # (In a real implementation, we would track best_loss, best_dtw over epochs and save accordingly)
            best_dir = os.path.join(ckpt_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            # Dummy saving best for this epoch (just to satisfy structure)
            torch.save(ckpt_data, os.path.join(best_dir, "loss.pt"))
            if geo_metrics:
                torch.save(ckpt_data, os.path.join(best_dir, "dtw.pt"))
                torch.save(ckpt_data, os.path.join(best_dir, "endpoint.pt"))
            # Generate qualitative outputs only at the end of training to avoid overhead
            if epoch == epochs:
                print("Generating final qualitative previews...")
                model.eval()
                with torch.no_grad():
                    tokens, t_lens, coords, c_lens = next(iter(dataloader))
                    tokens, coords = tokens.to(device), coords.to(device)
                    preds = model(tokens, target_len=coords.size(1))
                    
                    pred_traj = tensor_to_trajectory(preds[0, :c_lens[0]].cpu())
                    target_traj = tensor_to_trajectory(coords[0, :c_lens[0]].cpu())
                    
                    pred_path = tracker.get_path("predictions", "final_prediction.svg")
                    renderer.render(pred_traj, pred_path, format="svg")
                    
                    target_path = tracker.get_path("predictions", "final_target.svg")
                    renderer.render(target_traj, target_path, format="svg")
                
    # Capture and save environment info
    if is_primary:
        from src.utils.environment import capture_environment, save_environment
        from src.utils.config import load_colab_config
        cfg = load_colab_config()
        env_info = capture_environment(cfg, train_dir, t0_total)
        save_environment(env_info, tracker.exp_dir)
                    
        print(f"Training completed. Outputs saved to {tracker.exp_dir}")
        
    cleanup_ddp()
    return tracker.exp_dir if is_primary else None
