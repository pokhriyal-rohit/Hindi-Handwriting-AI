import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import List, Dict, Tuple

class OfflineDataset(Dataset):
    """
    Loads offline images and labels from data/canonical/offline/[split]/
    """
    def __init__(self, data_dir: str, img_height: int = 32, augment: bool = False):
        self.data_dir = data_dir
        self.img_height = img_height
        self.augment = augment
        
        labels_path = os.path.join(data_dir, "labels.json")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels not found at {labels_path}")
            
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels: Dict[str, str] = json.load(f)
            
        self.samples = list(self.labels.items()) # list of (rel_path, text)
        
        # We resize height to img_height and keep aspect ratio for width
        transforms_list = [T.Grayscale(1)]
        
        if self.augment:
            transforms_list.extend([
                T.RandomRotation(degrees=3, fill=255),
                T.RandomAffine(degrees=0, translate=(0.02, 0.05), scale=(0.95, 1.05), fill=255),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                T.ColorJitter(brightness=0.2, contrast=0.2)
            ])
            
        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.transform = T.Compose(transforms_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, text = self.samples[idx]
        img_path = os.path.join(self.data_dir, rel_path)
        
        img = Image.open(img_path).convert("L")
        w, h = img.size
        new_w = int(w * (self.img_height / h))
        img = img.resize((new_w, self.img_height), Image.Resampling.LANCZOS)
        
        tensor_img = self.transform(img) # (1, H, W)
        
        metadata = {
            "rel_path": rel_path,
            "writer_id": rel_path.split("/")[0].replace("writer_", "")
        }
        
        return tensor_img, text, metadata

def offline_collate_fn(batch: List[Tuple[torch.Tensor, str, dict]]):
    """
    Collates batches of offline images. Images have different widths.
    We must pad the widths to the maximum width in the batch.
    Texts are returned as lists of strings (tokenization happens in training loop).
    """
    images, texts, metadata = zip(*batch)
    
    # 1. Pad images to max width in this batch
    max_w = max(img.shape[2] for img in images)
    batch_size = len(images)
    
    padded_images = torch.zeros(batch_size, 1, images[0].shape[1], max_w)
    
    input_lengths = []
    for i, img in enumerate(images):
        w = img.shape[2]
        padded_images[i, :, :, :w] = img
        input_lengths.append(w)
        
    return padded_images, torch.tensor(input_lengths, dtype=torch.long), list(texts), list(metadata)
