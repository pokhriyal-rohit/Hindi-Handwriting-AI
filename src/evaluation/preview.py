import os
import torch
import random
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from torch.utils.data import DataLoader
from src.datasets.offline_dataset import OfflineDataset, offline_collate_fn
from src.datasets.online_dataset import CanonicalTrajectoryDataset, synthetic_collate_fn
from src.tokenizers.devanagari import DevanagariTokenizer
from src.models.ocr.registry import build_ocr_model
from src.models.baseline_lstm import BaselineLSTM
from src.training.utils import tensor_to_trajectory
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
    
    with torch.no_grad():
        for i, (images, targets_1d, target_lengths, raw_texts) in enumerate(loader):
            images = images.to(device)
            preds = model(images)
            preds = preds.permute(1, 0, 2)
            pred_indices = preds.argmax(dim=-1)
            pred_text = tokenizer.decode(pred_indices[0].cpu().tolist())
            
            img_np = images[0].cpu().numpy().transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(-1)
                
            plt.figure(figsize=(6, 3))
            plt.imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
            plt.title(f"Target: {raw_texts[0]}\nPred: {pred_text}", fontname='Nirmala UI' if os.name == 'nt' else 'sans-serif')
            plt.axis('off')
            out_path = os.path.join(preview_dir, f"preview_{i}.png")
            plt.savefig(out_path)
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

def run_preview(exp_id: str, num_samples: int = 5):
    if exp_id.endswith("_ocr"):
        preview_ocr(exp_id, num_samples)
    else:
        preview_trajectory(exp_id, num_samples)
