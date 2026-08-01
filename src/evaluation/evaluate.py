import os
import time
import json
import csv
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
    
    
    # 1. Setup output directories
    metrics_dir = os.path.join(exp_dir, "metrics")
    predictions_dir = os.path.join(exp_dir, "predictions")
    reports_dir = os.path.join(exp_dir, "reports")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    cer_scores = []
    wer_scores = []
    sample_metrics = []
    total_time = 0.0
    
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
        for images, input_lengths, texts, metadata in loader:
            images = images.to(device)
            start_time = time.time()
            preds = model(images)
            total_time += time.time() - start_time
            
            probs = torch.nn.functional.softmax(preds, dim=-1)
            pred_confidences, pred_indices = probs.max(dim=-1)
            
            ocr_model = model.module if hasattr(model, 'module') else model
            pred_lengths = torch.clamp(ocr_model.get_output_length(input_lengths), max=preds.size(1))
            
            for b in range(images.size(0)):
                raw_pred = pred_indices[b, :pred_lengths[b]].cpu().tolist()
                raw_confs = pred_confidences[b, :pred_lengths[b]].cpu().tolist()
                
                pred_text, char_confs = decode_with_confidences(raw_pred, raw_confs, tokenizer)
                
                cer = OCRMetrics.compute_cer(texts[b], pred_text)
                wer = OCRMetrics.compute_wer(texts[b], pred_text)
                overall_conf = sum(char_confs) / len(char_confs) if char_confs else 0.0
                
                cer_scores.append(cer)
                wer_scores.append(wer)
                
                # Format character confidences for output
                char_conf_dict = {char: round(conf, 4) for char, conf in zip(pred_text, char_confs)}
                
                sample_metrics.append({
                    "filename": metadata[b]["rel_path"],
                    "ground_truth": texts[b],
                    "predicted": pred_text,
                    "overall_confidence": round(overall_conf, 4),
                    "character_confidences": char_conf_dict,
                    "cer": round(cer, 4),
                    "wer": round(wer, 4)
                })
                
    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else float('nan')
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else float('nan')
    avg_conf = sum([m["overall_confidence"] for m in sample_metrics]) / len(sample_metrics) if sample_metrics else 0.0
    avg_inf_time = (total_time / len(dataset)) * 1000 # ms per sample
    
    result = {
        "exp_id": exp_id,
        "split": split,
        "num_samples": len(dataset),
        "cer": avg_cer,
        "wer": avg_wer,
        "average_confidence": avg_conf,
        "avg_inference_time_ms": avg_inf_time,
        "checkpoint": os.path.basename(ckpt_path)
    }
    
    print(f"Evaluation complete.")
    print(f"CER: {result['cer']:.4f} | WER: {result['wer']:.4f}")
    
    # Save evaluation JSON
    report_json_path = os.path.join(metrics_dir, f"evaluation_{split}.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    # Save prediction JSON
    pred_json_path = os.path.join(predictions_dir, "prediction.json")
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(sample_metrics, f, indent=2, ensure_ascii=False)
        
    # Save prediction CSV
    pred_csv_path = os.path.join(predictions_dir, "predictions.csv")
    with open(pred_csv_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "ground_truth", "predicted", "overall_confidence", "cer", "wer"])
        writer.writeheader()
        for sm in sample_metrics:
            writer.writerow({
                "filename": sm["filename"],
                "ground_truth": sm["ground_truth"],
                "predicted": sm["predicted"],
                "overall_confidence": sm["overall_confidence"],
                "cer": sm["cer"],
                "wer": sm["wer"]
            })
            
    # Save prediction TXT
    pred_txt_path = os.path.join(predictions_dir, "prediction.txt")
    with open(pred_txt_path, "w", encoding="utf-8") as f:
        for sm in sample_metrics:
            f.write(f"{sm['filename']} | GT: {sm['ground_truth']} | PR: {sm['predicted']} | CER: {sm['cer']} | CONF: {sm['overall_confidence']}\n")
            
    # Generate Markdown Report
    md_report_path = os.path.join(reports_dir, f"evaluation_report.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(f"# OCR Evaluation Report\n\n")
        f.write(f"- **Experiment ID**: {exp_id}\n")
        f.write(f"- **Dataset Split**: {split}\n")
        f.write(f"- **Number of Samples**: {len(dataset)}\n")
        f.write(f"- **Checkpoint Used**: {os.path.basename(ckpt_path)}\n\n")
        f.write(f"## Metrics\n\n")
        f.write(f"- **CER**: {avg_cer:.4f}\n")
        f.write(f"- **WER**: {avg_wer:.4f}\n")
        f.write(f"- **Average Confidence**: {avg_conf:.4f}\n")
        f.write(f"- **Average Inference Time**: {avg_inf_time:.2f} ms/sample\n")
        
    print(f"Reports saved to {metrics_dir}, {predictions_dir}, and {reports_dir}")

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
