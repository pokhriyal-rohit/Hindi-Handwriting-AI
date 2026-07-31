import os
import json
import time
import torch
from PIL import Image
import torchvision.transforms as T

from src.models.ocr.registry import build_ocr_model
from src.tokenizers.devanagari import DevanagariTokenizer

def run_recognition(image_path: str, exp_dir: str):
    """
    Runs inference on a single image using a trained OCR model.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    vocab_path = os.path.join(exp_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab not found in experiment directory: {vocab_path}")
        
    ckpt_path = os.path.join(exp_dir, "latest.pt") # Or best_wer.pt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = DevanagariTokenizer()
    tokenizer.load_vocab(vocab_path)
    
    # Normally we'd load config.yaml from exp_dir, but we'll use defaults for CRNN
    config = {"img_channels": 1, "hidden_size": 256}
    model = build_ocr_model("crnn_baseline", tokenizer.vocab_size, config).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    img = Image.open(image_path).convert("L")
    w, h = img.size
    img_height = 32
    new_w = int(w * (img_height / h))
    img = img.resize((new_w, img_height), Image.Resampling.LANCZOS)
    
    transform = T.Compose([
        T.Grayscale(1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    tensor_img = transform(img).unsqueeze(0).to(device) # (1, 1, 32, W)
    
    t0 = time.time()
    with torch.no_grad():
        preds = model(tensor_img) # (1, T, vocab)
        probs = torch.nn.functional.softmax(preds, dim=2)
        
        # Greedy decoding with confidences
        max_probs, max_idx = torch.max(probs, dim=2) # (1, T)
        
        input_lengths = torch.tensor([tensor_img.shape[3]], dtype=torch.long)
        pred_lengths = model.get_output_length(input_lengths)
        T_actual = pred_lengths[0].item()
        
        raw_pred = max_idx[0, :T_actual].cpu().tolist()
        raw_probs = max_probs[0, :T_actual].cpu().tolist()
        
    inference_time = time.time() - t0
    
    # Decode and compute character confidences
    decoded_chars = []
    char_confidences = []
    prev = -1
    for p_idx, p_prob in zip(raw_pred, raw_probs):
        if p_idx != prev and p_idx != 0:
            char = tokenizer.decode([p_idx], remove_repeats=False)
            decoded_chars.append(char)
            char_confidences.append(p_prob)
        elif p_idx == prev and p_idx != 0:
            # If it's a repeat, we might want to update the confidence or just skip
            # Standard CTC usually takes the max probability in a continuous run
            char_confidences[-1] = max(char_confidences[-1], p_prob)
            
        prev = p_idx
        
    predicted_text = "".join(decoded_chars)
    overall_confidence = sum(char_confidences) / len(char_confidences) if char_confidences else 0.0
    
    result = {
        "predicted_text": predicted_text,
        "overall_confidence": round(overall_confidence, 4),
        "character_confidences": [round(c, 4) for c in char_confidences],
        "inference_time_seconds": round(inference_time, 4)
    }
    
    with open("prediction.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    with open("prediction.txt", "w", encoding="utf-8") as f:
        f.write(predicted_text)
        
    print(f"Recognized: {predicted_text}")
    print(f"Confidence: {result['overall_confidence']:.2f}")
    print(f"Time: {result['inference_time_seconds']:.3f}s")
    print(f"Saved to prediction.json and prediction.txt")
    
    return result
