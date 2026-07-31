# Hindi Handwriting AI Platform

A complete platform for both Online Handwriting Generation and Offline Image Understanding for the Devanagari script.

## Architecture
This repository implements a modular, PyTorch-based architecture supporting:
1. **Online Trajectory Generation:** Converts text into synthetic handwriting trajectories using LSTMs/Transformers.
2. **Offline Image Understanding (OCR):** Predicts text from grayscale handwritten images using a CRNN (VGG + BiLSTM + CTC).
3. **Rendering:** Engine to convert trajectories into SVG/PNG.

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

## Documentation
- [Colab Setup Guide](COLAB_SETUP.md)
- [Project Technical Reference](PROJECT_TECHNICAL_REFERENCE.md)
- [Training Strategy](TRAINING_STRATEGY.md)
- [Style Dataset Analysis](STYLE_DATASET_ANALYSIS.md)
- [Colab Release Report](COLAB_RELEASE_REPORT.md)
- `python main.py train-ocr`: Train the OCR module
- `python main.py recognize --image ... --exp_dir ...`: Run inference on a handwritten image
