import os
import json
import math
import time

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "custom_hindi", "writer_mock"))

def generate_mock_samples():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    prompts = [
        "क", "ख", "ग", "घ", "च", "छ", "ज", "झ", "ट", "ठ", 
        "ड", "ढ", "त", "थ", "द", "ध", "न", "प", "फ", "ब",
        "भारत", "नमस्ते", "विद्यालय", "विज्ञान", "शांति", "प्रेम", "सत्य", "पुस्तकालय", "अध्यापक", "विद्यार्थी",
        "संस्कृति", "धर्म", "ज्ञान", "प्रकाश", "आसमान", "धरती", "समुद्र", "पर्वत", "नदी", "वन"
    ]
    samples_per_prompt = 10
    
    for prompt in prompts:
        prompt_dir = os.path.join(DATA_DIR, prompt)
        os.makedirs(prompt_dir, exist_ok=True)
        
        for i in range(samples_per_prompt):
            strokes = []
            current_stroke = []
            
            # Simulate a 2-second drawing with 60Hz sampling (~120 points)
            start_t = int(time.time() * 1000)
            
            # Generate deterministic sine wave with noise for uniqueness
            noise_x = (i * 5) % 20
            noise_y = (i * 3) % 15
            
            for step in range(120):
                x = 100 + (step * 2) + noise_x + math.sin(step * 0.1) * 10
                y = 200 + noise_y + math.cos(step * 0.1) * 20
                t = start_t + (step * 16) # ~60Hz
                
                current_stroke.append({"x": x, "y": y, "t": t})
                
                # Random pen lift
                if step == 60:
                    strokes.append(list(current_stroke))
                    current_stroke = []
                    
            if current_stroke:
                strokes.append(list(current_stroke))
                
            stats = {
                "strokes": len(strokes),
                "points": 120,
                "duration_ms": 120 * 16,
                "path_length_px": 300.0,
                "avg_speed_px_ms": 0.15,
                "max_speed_px_ms": 0.5,
                "bounding_box": {"min_x": 100, "max_x": 350, "min_y": 180, "max_y": 250}
            }
            
            data = {
                "writer_id": "writer_mock",
                "word": prompt,
                "script": "Devanagari",
                "device": "mouse",
                "canvas_width": 800,
                "canvas_height": 400,
                "timestamp": "2026-07-31T12:00:00Z",
                "sampling_rate": "browser-event-driven",
                "statistics": stats,
                "strokes": strokes
            }
            
            filepath = os.path.join(prompt_dir, f"sample_{i+1:03d}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
    print(f"Generated {len(prompts) * samples_per_prompt} pilot mock samples in {DATA_DIR}")

if __name__ == "__main__":
    generate_mock_samples()
