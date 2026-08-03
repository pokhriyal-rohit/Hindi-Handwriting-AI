import os
import torch
import random
import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except ImportError:
    plt = None
    fm = None


import urllib.request

def _get_devanagari_font(size: int = 12):
    """Return a FontProperties object backed by the best available Devanagari font.
    Searches for Noto Sans Devanagari (Kaggle/Linux) then Nirmala UI (Windows),
    falling back to downloading a font if nothing suitable is found.
    """
    if fm is None:
        return None
    candidates = [
        "Noto Sans Devanagari",
        "Nirmala UI",
        "Mangal",
        "FreeSans",
        "Lohit Devanagari",
    ]
    for name in candidates:
        try:
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            if path and "DejaVu" not in path and "STIXGeneral" not in path:
                return fm.FontProperties(fname=path, size=size)
        except Exception:
            pass
            
    # Last resort system search
    for font in fm.fontManager.ttflist:
        if "devanagari" in font.name.lower():
            try:
                return fm.FontProperties(fname=font.fname, size=size)
            except Exception:
                pass
                
    # If not found locally, download to a 'fonts' directory
    font_dir = os.path.join(os.path.dirname(__file__), "..", "..", "fonts")
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, "NotoSansDevanagari-Regular.ttf")
    
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
        try:
            print("Downloading Devanagari font for Matplotlib...")
            urllib.request.urlretrieve(url, font_path)
        except Exception as e:
            print(f"Warning: Failed to download Devanagari font: {e}")
            return fm.FontProperties(size=size)
            
    return fm.FontProperties(fname=font_path, size=size)
from torch.utils.data import DataLoader, Subset
from src.datasets.offline_dataset import OfflineDataset, offline_collate_fn
from src.datasets.online_dataset import CanonicalTrajectoryDataset, synthetic_collate_fn
from src.tokenizers.devanagari import DevanagariTokenizer
from src.models.ocr.registry import build_ocr_model
from src.models.baseline_lstm import BaselineLSTM
from src.training.utils import tensor_to_trajectory
from src.evaluation.metrics.ocr import OCRMetrics
from src.evaluation.visualization.plotters import plot_trajectory_overlay

def preview_ocr(exp_id: str, num_samples: int = 5):
    if plt is None:
        print("Matplotlib not installed. Cannot preview.")
        return
        
    exp_dir = os.path.join("experiments", "OCR", exp_id)
    canonical_dir = os.path.join("data", "canonical", "offline", "validation")
    preview_dir = os.path.join(exp_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    
    if not os.path.exists(canonical_dir):
        print(f"Dataset split not found: {canonical_dir}")
        return
        
    tokenizer = DevanagariTokenizer()
    vocab_path = os.path.join(exp_dir, "vocab.json")
    if os.path.exists(vocab_path):
        tokenizer.load_vocab(vocab_path)
    
    dataset = OfflineDataset(canonical_dir)
    if len(dataset) == 0:
        return
        
    # sample random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=offline_collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ocr_model("crnn_baseline", vocab_size=tokenizer.vocab_size, config={}).to(device)
    
    ckpt_path = os.path.join(exp_dir, "best_cer.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(exp_dir, "latest.pt")
        
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    else:
        print(f"No checkpoint found at {ckpt_path}")
        return
        
    model.eval()
    
    def decode_with_confidences(indices, confidences, tokenizer):
        chars = []
        confs = []
        prev = -1
        for i, idx in enumerate(indices):
            if idx != prev and idx != 0:
                chars.append(tokenizer.idx_to_char.get(idx, ""))
                confs.append(confidences[i])
            elif idx == prev and idx != 0:
                confs[-1] = max(confs[-1], confidences[i])
            prev = idx
        return "".join(chars), confs

    with torch.no_grad():
        for i, (images, input_lengths, texts, metadata) in enumerate(loader):
            images = images.to(device)
            preds = model(images)
            
            probs = torch.nn.functional.softmax(preds, dim=-1)
            pred_confidences, pred_indices = probs.max(dim=-1)
            
            ocr_model = model.module if hasattr(model, 'module') else model
            pred_lengths = torch.clamp(ocr_model.get_output_length(input_lengths), max=preds.size(1))
            
            raw_pred = pred_indices[0, :pred_lengths[0]].cpu().tolist()
            raw_confs = pred_confidences[0, :pred_lengths[0]].cpu().tolist()
            
            pred_text, char_confs = decode_with_confidences(raw_pred, raw_confs, tokenizer)
            
            gt_text = texts[0]
            cer = OCRMetrics.compute_cer(gt_text, pred_text)
            wer = OCRMetrics.compute_wer(gt_text, pred_text)
            overall_conf = sum(char_confs) / len(char_confs) if char_confs else 0.0
            
            img_np = images[0].cpu().numpy().transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(-1)
                
            fig = plt.figure(figsize=(10, 6))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
            
            ax1 = fig.add_subplot(gs[0])
            ax1.imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
            ax1.set_title("Input Image")
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[1])
            ax2.axis('off')
            
            char_conf_lines = [f"{char} : {conf:.2f}" for char, conf in zip(pred_text, char_confs)]
            char_conf_str = "\n".join(char_conf_lines)
            
            info_text = (
                f"Ground Truth:\n{gt_text}\n\n"
                f"Prediction:\n{pred_text}\n\n"
                f"CER: {cer:.4f}  |  WER: {wer:.4f}\n"
                f"Overall Confidence: {overall_conf:.2f}\n\n"
                f"Character Confidence:\n{char_conf_str}"
            )
            
            deva_font = _get_devanagari_font(size=11)
            ax2.text(0.0, 1.0, info_text, transform=ax2.transAxes,
                     verticalalignment='top', fontproperties=deva_font)
            
            plt.tight_layout()
            out_path = os.path.join(preview_dir, f"preview_{i}.png")
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f"Saved OCR preview to {out_path}")

def preview_trajectory(exp_id: str, num_samples: int = 5):
    exp_dir = os.path.join("experiments", exp_id)
    if not os.path.exists(exp_dir):
        exp_dir = os.path.join("runs", exp_id)
        
    canonical_dir = os.path.join("data", "canonical", "online", "validation")
    preview_dir = os.path.join(exp_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)
    
    dataset = CanonicalTrajectoryDataset(canonical_dir)
    if len(dataset) == 0:
        return
        
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=synthetic_collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineLSTM(vocab_size=5000, embed_dim=64, hidden_dim=128, max_out_len=200).to(device)
    
    ckpt_path = os.path.join(exp_dir, "checkpoints", "best_dtw.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(exp_dir, "checkpoints", "latest.pt")
        
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    else:
        print(f"No checkpoint found at {ckpt_path}")
        return
        
    model.eval()
    
    with torch.no_grad():
        for i, (tokens, t_lens, coords, c_lens) in enumerate(loader):
            tokens, coords = tokens.to(device), coords.to(device)
            preds = model(tokens, target_len=coords.size(1))
            
            length = c_lens[0].item()
            pred_traj = tensor_to_trajectory(preds[0, :length])
            target_traj = tensor_to_trajectory(coords[0, :length])
            
            out_path = os.path.join(preview_dir, f"overlay_{i}.png")
            plot_trajectory_overlay(pred_traj, target_traj, out_path)
            print(f"Saved Trajectory preview to {out_path}")

def run_preview(exp_id: str, num_samples: int = 5, mode: str = "auto"):
    if mode == "ocr" or exp_id.endswith("_ocr"):
        preview_ocr(exp_id, num_samples)
    else:
        preview_trajectory(exp_id, num_samples)
