import os
import json
import torch
from torch.utils.data import DataLoader
from src.evaluation.metrics.ocr import OCRMetrics
from src.datasets.offline_dataset import OfflineDataset, offline_collate_fn
from src.datasets.online_dataset import CanonicalTrajectoryDataset, synthetic_collate_fn
from src.tokenizers.devanagari import DevanagariTokenizer
from src.models.ocr.registry import build_ocr_model
from src.models.baseline_lstm import BaselineLSTM
from src.training.utils import tensor_to_trajectory
from src.evaluation.metrics.trajectory import DTWMetric, EndpointErrorMetric
from src.utils.config import load_colab_config

def evaluate_ocr(exp_id: str, split: str = "validation"):
    exp_dir = os.path.join("experiments", "OCR", exp_id)
    canonical_dir = os.path.join("data", "canonical", "offline", split)
    
    if not os.path.exists(canonical_dir):
        print(f"Dataset split not found: {canonical_dir}")
        return
        
    tokenizer = DevanagariTokenizer()
    vocab_path = os.path.join(exp_dir, "vocab.json")
    if os.path.exists(vocab_path):
        tokenizer.load_vocab(vocab_path)
    else:
        print(f"Vocab not found at {vocab_path}")
        return
        
    dataset = OfflineDataset(canonical_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=offline_collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We don't have the config, but we can assume crnn_baseline for now, or read from config if we saved it
    # For now, default to building crnn_baseline
    model = build_ocr_model("crnn_baseline", vocab_size=tokenizer.vocab_size, config={}).to(device)
    
    ckpt_path = os.path.join(exp_dir, "best_cer.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(exp_dir, "latest.pt")
        
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}")
        return
        
    model.eval()
    
    print(f"Starting OCR evaluation on {split} ({len(dataset)} samples)...")
    
    cer_scores = []
    wer_scores = []
    
    with torch.no_grad():
        for images, targets_1d, target_lengths, raw_texts in loader:
            images = images.to(device)
            preds = model(images)
            preds = preds.permute(1, 0, 2)
            preds = torch.nn.functional.log_softmax(preds, dim=2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(1), dtype=torch.long)
            
            # Use greedy decoding for metrics
            # preds shape: [batch, time, classes]
            pred_indices = preds.argmax(dim=-1)
            
            for b in range(images.size(0)):
                pred_text = tokenizer.decode(pred_indices[b].cpu().tolist(), remove_repeats=True)
                cer_scores.append(OCRMetrics.compute_cer(raw_texts[b], pred_text))
                wer_scores.append(OCRMetrics.compute_wer(raw_texts[b], pred_text))
                
    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else float('nan')
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else float('nan')
    result = {"cer": avg_cer, "wer": avg_wer}
    
    print(f"Evaluation complete.")
    print(f"CER: {result['cer']:.4f} | WER: {result['wer']:.4f}")
    
    report_path = os.path.join(exp_dir, f"evaluation_{split}.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Report saved to {report_path}")

def evaluate_trajectory(exp_id: str, split: str = "validation"):
    exp_dir = os.path.join("experiments", exp_id)
    if not os.path.exists(exp_dir):
        # try runs folder
        exp_dir = os.path.join("runs", exp_id)
        
    canonical_dir = os.path.join("data", "canonical", "online", split)
    if not os.path.exists(canonical_dir):
        print(f"Dataset split not found: {canonical_dir}")
        return
        
    dataset = CanonicalTrajectoryDataset(canonical_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=synthetic_collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineLSTM(vocab_size=5000, embed_dim=64, hidden_dim=128, max_out_len=200).to(device)
    
    ckpt_path = os.path.join(exp_dir, "checkpoints", "best_dtw.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(exp_dir, "checkpoints", "best_loss.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(exp_dir, "checkpoints", "latest.pt")
        
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}")
        return
        
    model.eval()
    
    dtw_metric = DTWMetric()
    ee_metric = EndpointErrorMetric()
    dtw_scores = []
    ee_scores = []
    
    print(f"Starting Trajectory evaluation on {split} ({len(dataset)} samples)...")
    
    with torch.no_grad():
        for tokens, t_lens, coords, c_lens in loader:
            tokens, coords = tokens.to(device), coords.to(device)
            preds = model(tokens, target_len=coords.size(1))
            batch_size = tokens.size(0)
            
            for i in range(batch_size):
                length = c_lens[i].item()
                pred_traj = tensor_to_trajectory(preds[i, :length])
                target_traj = tensor_to_trajectory(coords[i, :length])
                
                dtw_res = dtw_metric.evaluate(pred_traj, target_traj)
                ee_res = ee_metric.evaluate(pred_traj, target_traj)
                dtw = dtw_res.get("dtw_distance")
                ee = ee_res.get("endpoint_error")
                if dtw is not None and dtw != float('inf'):
                    dtw_scores.append(dtw)
                if ee is not None and ee != float('inf'):
                    ee_scores.append(ee)
                    
    avg_dtw = sum(dtw_scores)/len(dtw_scores) if dtw_scores else float('nan')
    avg_ee = sum(ee_scores)/len(ee_scores) if ee_scores else float('nan')
    
    print(f"Evaluation complete.")
    print(f"DTW: {avg_dtw:.4f} | Endpoint Error: {avg_ee:.4f}")
    
    report_path = os.path.join(exp_dir, f"evaluation_{split}.json")
    with open(report_path, "w") as f:
        json.dump({"DTW": avg_dtw, "EndpointError": avg_ee}, f, indent=2)
    print(f"Report saved to {report_path}")

def run_evaluation(exp_id: str, split: str = "validation", mode: str = "auto"):
    if mode == "ocr" or exp_id.endswith("_ocr"):
        evaluate_ocr(exp_id, split)
    else:
        evaluate_trajectory(exp_id, split)
