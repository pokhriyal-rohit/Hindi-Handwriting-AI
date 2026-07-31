import os
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.offline_dataset import OfflineDataset, offline_collate_fn
from src.tokenizers.devanagari import DevanagariTokenizer
from src.models.ocr.registry import build_ocr_model
from src.evaluation.metrics.ocr import OCRMetrics

def train_ocr_model(
    config: Dict[str, Any],
    exp_id: str,
    resume_checkpoint: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting OCR Training on device: {device}")
    
    # Setup Experiment Directory
    exp_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "experiments", "OCR", exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 32)
    eval_every = config.get("eval_every", 5)
    model_name = config.get("model_name", "crnn_baseline")
    
    canonical_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "canonical", "offline"))
    train_dir = os.path.join(canonical_dir, "train")
    val_dir = os.path.join(canonical_dir, "validation")
    
    if not os.path.exists(train_dir):
        raise RuntimeError(f"Offline dataset missing at {train_dir}")
        
    # 1. Build & Save Tokenizer
    tokenizer = DevanagariTokenizer()
    with open(os.path.join(train_dir, "labels.json"), "r", encoding="utf-8") as f:
        train_labels = json.load(f)
        tokenizer.build_vocab(list(train_labels.values()))
    
    vocab_path = os.path.join(exp_dir, "vocab.json")
    tokenizer.save_vocab(vocab_path)
    print(f"Tokenizer built and saved. Vocab size: {tokenizer.vocab_size}")
    
    # 2. Datasets
    train_dataset = OfflineDataset(train_dir)
    val_dataset = OfflineDataset(val_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=offline_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=offline_collate_fn)
    
    # 3. Model & Loss
    model = build_ocr_model(model_name, tokenizer.vocab_size, config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    
    start_epoch = 1
    best_loss = float('inf')
    best_cer = float('inf')
    best_wer = float('inf')
    
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        best_cer = ckpt.get('best_cer', float('inf'))
        best_wer = ckpt.get('best_wer', float('inf'))
        print(f"Resumed from {resume_checkpoint} at epoch {start_epoch}")
        
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        
        for i, (images, input_lengths, texts, metadata) in enumerate(train_loader):
            images = images.to(device)
            
            # Tokenize targets on the fly
            targets = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long).to(device)
            targets_1d = torch.cat(targets).to(device)
            
            optimizer.zero_grad()
            preds = model(images) # (B, T, vocab)
            
            preds = preds.permute(1, 0, 2) 
            preds = torch.nn.functional.log_softmax(preds, dim=2)
            
            pred_lengths = model.get_output_length(input_lengths).to(device)
            pred_lengths = torch.clamp(pred_lengths, max=preds.size(0))
            
            loss = criterion(preds, targets_1d, pred_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Fast dev run for CPU verification
            if os.environ.get("FAST_DEV_RUN") == "1" and i >= 2:
                break
            
        epoch_time = time.time() - t0
        avg_loss = total_loss / len(train_loader)
        
        # Validation & Evaluation
        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            val_loss_total = 0.0
            cer_total = 0.0
            wer_total = 0.0
            
            with torch.no_grad():
                for j, (images, input_lengths, texts, metadata) in enumerate(val_loader):
                    images = images.to(device)
                    targets = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
                    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long).to(device)
                    targets_1d = torch.cat(targets).to(device)
                    
                    preds = model(images)
                    preds_ctc = torch.nn.functional.log_softmax(preds.permute(1, 0, 2), dim=2)
                    pred_lengths = torch.clamp(model.get_output_length(input_lengths), max=preds_ctc.size(0)).to(device)
                    
                    val_loss = criterion(preds_ctc, targets_1d, pred_lengths, target_lengths)
                    val_loss_total += val_loss.item()
                    
                    _, max_idx = torch.max(preds, dim=2) # (B, T)
                    
                    for i in range(images.size(0)):
                        raw_pred = max_idx[i, :pred_lengths[i]].cpu().tolist()
                        pred_str = tokenizer.decode(raw_pred, remove_repeats=True)
                        target_str = texts[i]
                        
                        cer_total += OCRMetrics.compute_cer(target_str, pred_str)
                        wer_total += OCRMetrics.compute_wer(target_str, pred_str)
                        
                    if os.environ.get("FAST_DEV_RUN") == "1" and j >= 1:
                        break
                        
            avg_val_loss = val_loss_total / len(val_loader)
            mean_cer = cer_total / len(val_dataset)
            mean_wer = wer_total / len(val_dataset)
            
            print(f"Epoch {epoch:03d} | Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f} | CER: {mean_cer:.2f} | WER: {mean_wer:.2f} | {epoch_time:.2f}s")
            
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_cer': best_cer,
                'best_wer': best_wer
            }
            
            torch.save(ckpt_data, os.path.join(exp_dir, "latest.pt"))
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(ckpt_data, os.path.join(exp_dir, "best_loss.pt"))
            if mean_cer < best_cer:
                best_cer = mean_cer
                torch.save(ckpt_data, os.path.join(exp_dir, "best_cer.pt"))
            if mean_wer < best_wer:
                best_wer = mean_wer
                torch.save(ckpt_data, os.path.join(exp_dir, "best_wer.pt"))
        else:
            print(f"Epoch {epoch:03d} | Train: {avg_loss:.4f} | {epoch_time:.2f}s")
            
    print(f"OCR Training completed. Outputs saved to {exp_dir}")
    return exp_dir
