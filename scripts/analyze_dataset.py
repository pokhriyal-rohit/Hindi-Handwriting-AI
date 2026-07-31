import os
import json
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "custom_hindi"))

def analyze_dataset():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} does not exist.")
        return

    writers = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    total_samples = 0
    words_collected = set()
    all_strokes = []
    all_points = []
    all_durations = []
    all_speeds = []
    all_path_lengths = []
    
    issues = []
    
    for writer in writers:
        writer_dir = os.path.join(DATA_DIR, writer)
        for word in os.listdir(writer_dir):
            word_dir = os.path.join(writer_dir, word)
            if not os.path.isdir(word_dir):
                continue
            
            words_collected.add(word)
            samples = [f for f in os.listdir(word_dir) if f.endswith(".json")]
            
            for s in samples:
                filepath = os.path.join(word_dir, s)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total_samples += 1
                    
                    stats = data.get("statistics", {})
                    if not stats:
                        issues.append(f"{writer}/{word}/{s} is missing statistics.")
                        continue
                    
                    strokes = stats.get("strokes", 0)
                    points = stats.get("points", 0)
                    duration = stats.get("duration_ms", 0)
                    speed = stats.get("avg_speed_px_ms", 0)
                    path_len = stats.get("path_length_px", 0)
                    
                    all_strokes.append(strokes)
                    all_points.append(points)
                    all_durations.append(duration)
                    all_speeds.append(speed)
                    all_path_lengths.append(path_len)
                    
                    if points < 10:
                        issues.append(f"{writer}/{word}/{s} has very few points ({points}).")
                    if duration > 10000:
                        issues.append(f"{writer}/{word}/{s} has unusually long duration ({duration}ms).")
                    if duration == 0:
                        issues.append(f"{writer}/{word}/{s} has 0ms duration.")

    print("Dataset Summary")
    print("---------------")
    print(f"Samples: {total_samples}")
    print(f"Writers: {len(writers)}")
    print(f"Unique Prompts: {len(words_collected)}")
    
    if total_samples > 0:
        print(f"\nAverage strokes/sample: {np.mean(all_strokes):.1f}")
        print(f"Average points/sample: {np.mean(all_points):.1f}")
        print(f"Average duration: {np.mean(all_durations):.0f} ms")
        print(f"Average speed: {np.mean(all_speeds):.2f} px/ms")
        print(f"Min/Max path length: {np.min(all_path_lengths):.0f} / {np.max(all_path_lengths):.0f} px")
    
    print("\nPotential issues:")
    if issues:
        for iss in issues[:10]:
            print(f"- {iss}")
        if len(issues) > 10:
            print(f"... and {len(issues) - 10} more issues.")
    else:
        print("None detected.")
        
    # Generate dataset_manifest.yaml
    manifest = {
        "dataset_name": "Custom Hindi Online",
        "version": "1.0.0",
        "writers": writers,
        "samples": total_samples,
        "unique_prompts": len(words_collected),
        "collector_version": "1.1",
        "schema_version": 1,
        "last_updated": __import__('datetime').datetime.utcnow().isoformat() + "Z"
    }
    import yaml
    manifest_path = os.path.join(DATA_DIR, "dataset_manifest.yaml")
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, sort_keys=False, allow_unicode=True)
    print(f"\nManifest saved to {manifest_path}")

if __name__ == "__main__":
    analyze_dataset()
