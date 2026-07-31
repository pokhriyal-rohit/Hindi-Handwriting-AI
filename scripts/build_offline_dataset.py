"""
scripts/build_offline_dataset.py
===================================
Migrates the raw IIIT-HW-Hindi_v1 offline image dataset into the canonical structure.
Creates: data/canonical/offline/[split]/writer_[id]/[image.jpg]
Generates: data/canonical/offline/[split]/labels.json
"""

import os
import sys
import shutil
import json
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "IIIT-HW-Hindi_v1")
CANONICAL_OFFLINE_DIR = os.path.join(PROJECT_ROOT, "data", "canonical", "offline")

def migrate_split(split_name: str, map_file: str):
    map_path = os.path.join(RAW_DIR, map_file)
    if not os.path.exists(map_path):
        print(f"Skipping {split_name} (Map file not found: {map_path})")
        return

    out_split_dir = os.path.join(CANONICAL_OFFLINE_DIR, split_name)
    os.makedirs(out_split_dir, exist_ok=True)
    
    labels = {}
    
    print(f"Parsing {map_file} for split: {split_name}...")
    with open(map_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"Migrating {len(lines)} images for {split_name}...")
    
    missing = 0
    copied = 0
    
    for line in tqdm(lines, desc=f"Migrating {split_name}"):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(" ")
        if len(parts) < 2:
            continue
            
        rel_img_path = parts[0]
        text_label = " ".join(parts[1:])
        
        # e.g. rel_img_path: HindiSeg/test/9/2/22.jpg
        # Writer ID is usually the first number after the split folder.
        # Format: HindiSeg/[split]/[writer_id]/[page]/[word].jpg
        path_parts = rel_img_path.replace("\\", "/").split("/")
        if len(path_parts) >= 4:
            writer_id = path_parts[2]
        else:
            writer_id = "unknown"
            
        src_img_path = os.path.join(RAW_DIR, "HindiSeg", rel_img_path)
        if not os.path.exists(src_img_path):
            missing += 1
            continue
            
        writer_dir_name = f"writer_{writer_id}"
        out_writer_dir = os.path.join(out_split_dir, writer_dir_name)
        os.makedirs(out_writer_dir, exist_ok=True)
        
        # We need a unique basename to avoid collisions just in case
        # original is 22.jpg, we prefix with page ID
        page_id = path_parts[-2] if len(path_parts) >= 5 else "0"
        orig_basename = os.path.basename(rel_img_path)
        new_basename = f"page{page_id}_{orig_basename}"
        
        dst_img_path = os.path.join(out_writer_dir, new_basename)
        
        # Use hardlink if possible to save massive space, otherwise copy
        try:
            if not os.path.exists(dst_img_path):
                os.link(src_img_path, dst_img_path)
        except OSError:
            shutil.copy2(src_img_path, dst_img_path)
            
        rel_canonical_path = f"{writer_dir_name}/{new_basename}"
        labels[rel_canonical_path] = text_label
        copied += 1
        
    # Save labels.json
    labels_path = os.path.join(out_split_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
        
    # Save metadata.json
    metadata = {
        "dataset_name": "IIIT-HW-Hindi_v1",
        "version": "1.0",
        "source": "IIIT Hyderabad",
        "license": "Research Use Only",
        "number_of_images": copied,
        "number_of_writers": len(set([k.split("/")[0] for k in labels.keys()])),
        "number_of_samples": copied,
        "creation_timestamp": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    }
    metadata_path = os.path.join(out_split_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f"Finished {split_name}: Copied/Linked {copied} images. Missing: {missing}.")
    print(f"Labels saved to: {labels_path}")
    print(f"Metadata saved to: {metadata_path}\n")

def main():
    print("Building Canonical Offline Dataset...")
    
    # IIIT dataset map files
    splits = [
        ("train", "train.txt"),
        ("validation", "val.txt"),
        ("test", "test.txt")
    ]
    
    for split_name, map_file in splits:
        migrate_split(split_name, map_file)
        
    print("Offline dataset migration complete.")

if __name__ == "__main__":
    main()
