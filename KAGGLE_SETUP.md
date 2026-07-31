# Google Kaggle Setup Guide

This guide ensures a seamless, reproducible deployment of the Hindi Handwriting AI Platform to Google Kaggle environments.

## 1. Environment Initialization
Open a new Google Kaggle notebook with a GPU runtime (T4 or A100).
Run the following commands to clone and install:

```bash
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/Hindi-Handwriting-AI.git
%cd Hindi-Handwriting-AI

# Install specific Kaggle requirements
!pip install -r requirements_Kaggle.txt
!pip install torchvision>=0.15.0
```

## 2. Dataset Upload
Upload your zipped `canonical` dataset directly to the Kaggle environment.

```bash
# Assuming you uploaded canonical.zip to /content/
!unzip /content/canonical.zip -d data/
```

Ensure your directory structure matches:
```
Hindi-Handwriting-AI/
└── data/
    └── canonical/
        ├── online/
        └── offline/
```

## 3. Training Commands

### Train Trajectory Generator
Train the online sequence-to-sequence model:
```bash
!python main.py train
```

### Train OCR Pipeline
Train the image-to-text offline CRNN model:
```bash
!python main.py train-ocr
```

## 4. Evaluation and Inference

### Evaluate Trajectory Models
```bash
!python main.py evaluate --exp_dir experiments/exp_001
```

### Recognize Text (OCR Inference)
Run inference on a single handwritten image:
```bash
!python main.py recognize --image data/canonical/offline/test/writer_x/img.png --exp_dir experiments/OCR/2026-07-31_001_ocr
```

## 5. Resuming Training
To resume training from a checkpoint (to protect against Kaggle disconnects):
*(Ensure you modify `train_ocr.py` or pass resume args if implemented in CLI)*
Checkpoints are automatically saved in your experiment directory (e.g., `experiments/OCR/.../latest.pt`).

## 6. Troubleshooting
- **No Module Named Torchvision:** Ensure you ran `pip install torchvision>=0.15.0` as Kaggle occasionally desyncs vision dependencies.
- **Out of Memory (OOM):** Modify `configs/ocr.yaml` or `configs/training.yaml` to reduce `batch_size` from 64 to 32 or 16.
- **Drive Backup:** Mount your Google Drive and symlink the `experiments/` folder to prevent data loss when the instance shuts down.
  ```python
  from google.Kaggle import drive
  drive.mount('/content/drive')
  !ln -s /content/drive/MyDrive/Hindi_AI_Experiments experiments
  ```
