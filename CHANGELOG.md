# Changelog

## [Unreleased]
### Added
- **Phase 2: Image Understanding (OCR)**
  - `scripts/build_offline_dataset.py` for massive offline image ingestion with metadata generation.
  - `src/datasets/offline_dataset.py` for decoupled offline PyTorch data loading.
  - `src/tokenizers/devanagari.py` to handle dynamic CTC tokenization.
  - `src/models/ocr/registry.py` and `base.py` for modular OCR architecture mapping.
  - `src/models/ocr/crnn.py` (VGG+BiLSTM) as the baseline OCR model.
  - `src/training/train_ocr.py` for CTC optimization, validation decoding, and robust checkpointing.
  - `src/inference/recognize.py` for single-image text prediction with confidences.
  - CLI integration: `ingest-offline`, `train-ocr`, `recognize`.
  - Comprehensive unit testing in `tests/test_ocr.py`.
- **Documentation & Colab Readiness**
  - Generated `PROJECT_TECHNICAL_REFERENCE.md`, `COLAB_SETUP.md`, `TRAINING_STRATEGY.md`.
  - Generated full Phase 3A research report `STYLE_DATASET_ANALYSIS.md`.
