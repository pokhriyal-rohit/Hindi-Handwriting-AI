import os
import json
import torch
import numpy as np
from collections import defaultdict
from src.datasets.online_dataset import CanonicalTrajectoryDataset, synthetic_collate_fn
from src.tokenizers.devanagari import DevanagariTokenizer
from src.models.baseline_lstm import BaselineLSTM
from src.evaluation.metrics.trajectory import DTWMetric
from src.datasets.structures import TrajectorySample
from src.training.utils import tensor_to_trajectory

def evaluate_characters():
    device = torch.device("cpu")
    print("Evaluating per-character trajectory generation difficulty...")
    
    # 1. Setup Data and Model
    # Note: CanonicalTrajectoryDataset does not take a tokenizer, it ord-encodes
    tokenizer = DevanagariTokenizer() # just used for getting vocab_size
    dataset = CanonicalTrajectoryDataset(r"data\canonical\online\train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=synthetic_collate_fn)
    
    model = BaselineLSTM(
        vocab_size=5000,
        embed_dim=64,
        hidden_dim=128,
        max_out_len=200
    ).to(device)
    
    # Load weights
    ckpt_path = r"experiments\2026-07-31_001_baseline_lstm\checkpoints\latest.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded weights from {ckpt_path}")
    else:
        print("Warning: Checkpoint not found, evaluating untrained model.")
        
    model.eval()
    
    # 2. Metric setup
    dtw = DTWMetric()
    char_errors = defaultdict(list)
    
    # 3. Evaluate Loop
    print(f"Evaluating {len(dataset)} samples...")
    with torch.no_grad():
        for i, (tokens, t_lens, coords, c_lens) in enumerate(dataloader):
            tokens, coords = tokens.to(device), coords.to(device)
            
            preds = model(tokens, target_len=coords.size(1))
            
            # Convert to TrajectorySample
            pred_traj = tensor_to_trajectory(preds[0, :c_lens[0]].cpu())
            tgt_traj = tensor_to_trajectory(coords[0, :c_lens[0]].cpu())
            
            # Compute DTW
            try:
                res = dtw.evaluate(pred_traj, tgt_traj)
                dist = res["dtw_distance"]
            except Exception as e:
                print(f"DTW Error on sample {i}: {e}")
                dist = float('inf')
                
            if dist == float('inf'):
                continue
                
            # Attribute error to all characters in the sequence
            text = "".join(chr(t) for t in tokens[0, :t_lens[0]].cpu().numpy())
            for char in text:
                char_errors[char].append(dist)
                
            if i % 50 == 0 and i > 0:
                print(f"Processed {i}/{len(dataset)} samples...")
                
    # 4. Aggregate and Rank
    ranking = []
    for char, errors in char_errors.items():
        if len(errors) > 2: # filter out noise (characters with < 3 samples)
            ranking.append({
                "character": char,
                "mean_dtw": np.mean(errors),
                "std_dtw": np.std(errors),
                "samples": len(errors)
            })
            
    # Sort by hardest first
    ranking.sort(key=lambda x: x["mean_dtw"], reverse=True)
    
    print("\n--- Character Difficulty Ranking (Hardest to Easiest) ---")
    print("(Character ranking saved to character_difficulty.json to avoid console encoding errors)")
    # 5. Save results
    out_file = "character_difficulty.json"
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(ranking, f, indent=2, ensure_ascii=False)
        
    print(f"\nDone! See {out_file} for full ranking.")

if __name__ == "__main__":
    evaluate_characters()
