# Project Technical Reference

This document provides a comprehensive technical overview of the Hindi Handwriting AI Platform. It is designed to rapidly onboard AI assistants, contributors, and researchers.

## 1. Overall Project Goal
To build a state-of-the-art multimodal AI platform that bridges online (trajectory-based) and offline (image-based) Devanagari handwriting. The ultimate goal is conditional trajectory generation (text + style image → trajectory) and robust offline OCR.

## 2. Current Architecture
The system employs a dual-pipeline PyTorch architecture:
1. **Online Generator:** An LSTM/Transformer-based sequence-to-sequence model generating `(x, y, pen_state)` trajectories from text embeddings.
2. **Offline OCR:** A Convolutional Recurrent Neural Network (CRNN: VGG Encoder + BiLSTM + CTC) that maps grayscale images to text.

## 3. Repository Structure
```
Hindi-Handwriting-AI/
├── configs/            # Modular YAML configurations (dataset, model, training, ocr)
├── data/
│   ├── canonical/      # The ONLY data format the models interact with (online/offline)
│   ├── manifests/      # Dataset tracking
│   └── reports/        # Diagnostic reports
├── docs/               # Technical documentation
├── experiments/        # All training runs (logs, checkpoints, outputs)
├── scripts/            # Data ingestion, building, and setup scripts
├── src/                # Core Python package
│   ├── datasets/       # PyTorch Dataset definitions (OnlineDataset, OfflineDataset)
│   ├── evaluation/     # Metrics (DTW, CER, WER, Frechet, Performance)
│   ├── inference/      # Inference pipelines (recognize.py)
│   ├── models/         # Neural network definitions (lstm, transformer, ocr/)
│   ├── renderer/       # SVG/PNG conversion engine
│   ├── tokenizers/     # Character-level text tokenization
│   ├── training/       # PyTorch training loops
│   └── utils/          # Config loaders, environment checkers
├── tests/              # Unit tests
└── main.py             # Unified CLI entry point
```

## 4. Pipeline Data Flow
- **Offline Pipeline:** `Raw Images` → `build_offline_dataset.py` → `data/canonical/offline` → `OfflineDataset` → `train_ocr.py` (CTCLoss) → `latest.pt`.
- **Online Pipeline:** `Raw JSON` → `build_canonical_dataset.py` → `data/canonical/online` → `OnlineDataset` → `train.py` (MSE/MDN Loss) → `latest.pt`.
- **Renderer Flow:** `Trajectory Tensor` → `Structures` → `Layout Engine` → `SVG/PNG File`.

## 5. Core Systems
- **Configuration:** Fully YAML-based (`configs/`). Configurations are isolated by concern (`ocr.yaml`, `training.yaml`) and loaded via `src.utils.config`.
- **Experiment System:** Automatically creates uniquely timestamped directories under `experiments/` (e.g., `experiments/OCR/2026-07-31_001_ocr`).
- **Canonical Schema:**
  - *Online:* `writer_x/file.json` containing `{text, strokes: [{points: [{x, y, time}]}]}`.
  - *Offline:* `writer_x/file.png` alongside a central `labels.json` and `metadata.json`.
- **Tokenizer:** `src.tokenizers.devanagari` handles dynamic character mapping. Index `0` is strictly reserved for the CTC `<blank>` token.
- **Model Registry:** Located in `src.models.ocr.registry`. Models like `crnn_baseline` are registered via decorators for dynamic instantiation.

## 6. Project Status
- **Phase 1 (Infrastructure):** ✅ Completed. Data pipelines, metrics, and renderer established.
- **Phase 2 (OCR Image Understanding):** ✅ Completed. CRNN baseline trained, CTC loss active, and inference engine deployed.
- **Phase 3 (Style Encoding):** ❌ Suspended pending data collection.
- **Phase 4 (Stroke Recovery):** ❌ Not Started.

## 7. Known Limitations
- **Data Scarcity:** The repository currently has only 12 offline writers and 1 mock online writer. This is statistically insufficient for generalized style learning.
- **Modality Gap:** There are no shared writers between the online and offline domains, preventing direct latent-space pairing.

## 8. Future Roadmap
1. **Data Collection Campaign:** Procure 200–500 unique Devanagari writers providing paired digital (online) and physical (offline) samples.
2. **Stage A Training (Geometry):** Pretrain the trajectory generator on a massive, identity-agnostic character dataset.
3. **Stage B Training (Style):** Introduce the Style Encoder (Contrastive Learning) and fine-tune the generator with concatenated style embeddings to clone handwriting.
