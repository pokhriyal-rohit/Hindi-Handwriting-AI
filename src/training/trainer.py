import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

from src.config.production import ProductionConfig
from src.models.production.model import ProductionHandwritingModel
from src.models.mdn import mdn_loss

class ProductionTrainer:
    """
    Robust production training pipeline.
    Handles Checkpointing, Mixed Precision, TensorBoard Logging, and Gradient Clipping.
    """
    def __init__(self, model: ProductionHandwritingModel, config: ProductionConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        t_config = self.config.training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=t_config.learning_rate, 
            weight_decay=t_config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=t_config.scheduler_patience, factor=t_config.scheduler_factor
        )
        
        self.scaler = torch.amp.GradScaler(enabled=t_config.mixed_precision and device == "cuda")
        self.writer = SummaryWriter(log_dir=self.config.log_dir)
        
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Saves a rich checkpoint with all necessary states to resume perfectly."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "config": self.config.model_dump(),
            "representation_version": "2.0" # Hardcoded for now
        }
        torch.save(checkpoint, path)
        if is_best:
            best_path = Path(path).parent / "best_model.pt"
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, path: str):
        """Resumes training from a checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
            
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        
    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        
        # Calculate current teacher forcing ratio
        progress = min(1.0, self.current_epoch / self.config.training.teacher_forcing_decay_epochs)
        tf_ratio = self.config.training.teacher_forcing_start - progress * (self.config.training.teacher_forcing_start - self.config.training.teacher_forcing_end)
        
        for batch_idx, batch in enumerate(dataloader):
            # Assumes batch = (text_tokens, text_lengths, coordinates)
            text_tokens, text_lengths, coords = [b.to(self.device) for b in batch]
            
            x = coords[:, :-1, :] # Inputs
            y = coords[:, 1:, :]  # Targets
            
            # Autocast for Mixed Precision
            with torch.amp.autocast(device_type="cuda" if self.device == "cuda" else "cpu", enabled=self.config.training.mixed_precision):
                mdn_params, _ = self.model(text_tokens, text_lengths, x)
                pi, mu1, mu2, sigma1, sigma2, rho, eos = mdn_params
                loss = mdn_loss(pi, mu1, mu2, sigma1, sigma2, rho, eos, y)
                loss = loss / self.config.training.gradient_accumulation_steps
                
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            self.global_step += 1
            
            if self.global_step % 10 == 0:
                self.writer.add_scalar("Train/Loss", loss.item() * self.config.training.gradient_accumulation_steps, self.global_step)
                self.writer.add_scalar("Train/LR", self.optimizer.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar("Train/TeacherForcing", tf_ratio, self.global_step)
                
        return total_loss / len(dataloader)
        
    def validate(self, dataloader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                text_tokens, text_lengths, coords = [b.to(self.device) for b in batch]
                x = coords[:, :-1, :]
                y = coords[:, 1:, :]
                
                with torch.amp.autocast(device_type="cuda" if self.device == "cuda" else "cpu", enabled=self.config.training.mixed_precision):
                    mdn_params, _ = self.model(text_tokens, text_lengths, x)
                    pi, mu1, mu2, sigma1, sigma2, rho, eos = mdn_params
                    loss = mdn_loss(pi, mu1, mu2, sigma1, sigma2, rho, eos, y)
                    
                total_loss += loss.item()
                
        val_loss = total_loss / len(dataloader)
        self.writer.add_scalar("Val/Loss", val_loss, self.current_epoch)
        return val_loss
        
    def fit(self, train_loader, val_loader):
        print(f"Starting production training on {self.device}")
        
        while self.current_epoch < self.config.training.epochs:
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                
            ckpt_name = f"epoch_{self.current_epoch:03d}_loss_{val_loss:.4f}.pt"
            self.save_checkpoint(os.path.join(self.config.checkpoint_dir, ckpt_name), is_best=is_best)
            
            print(f"Epoch {self.current_epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {self.best_loss:.4f}")
            self.current_epoch += 1
            
        self.writer.close()
