# GitHub Repository Architecture

This document outlines the exact directory and file structure that will be uploaded when this repository is pushed to GitHub. 

> [!NOTE]
> **Data and Experiments Excluded**
> To keep the repository lightweight and clean, raw data, processed canonical datasets, experiment checkpoints (`experiments/`), and generated release zips (`releases/`) are intentionally excluded via `.gitignore` and will **not** be uploaded to GitHub.

## Repository Structure (Tracked by Git)

```text
Hindi-Handwriting-AI/
├── .github/
│   └── workflows/
│       └── python-app.yml            # CI/CD pipeline configuration
├── configs/                          # Modular YAML configurations
│   ├── dataset.yaml
│   ├── default.yaml
│   ├── evaluation.yaml
│   ├── model.yaml
│   ├── model_transformer.yaml
│   ├── ocr.yaml
│   └── training.yaml
├── docs/                             # Technical documentation and guides
│   ├── BENCHMARKS.md
│   ├── BENCHMARK_FORMAT.md
│   ├── CANONICAL_SCHEMA.md
│   ├── CHANGELOG.md
│   ├── DATASET_COMPARISON.md
│   ├── DATASET_COMPATIBILITY_REPORT.md
│   ├── DECISIONS.md
│   ├── EVALUATION_ARCHITECTURE.md
│   ├── INFERENCE_ARCHITECTURE.md
│   ├── KNOWN_ISSUES.md
│   ├── METRIC_REFERENCE.md
│   ├── MODEL_COMPARISON_PLAN.md
│   ├── MODEL_PROFILE.md
│   ├── NEXT_TASK.md
│   ├── OATH.md
│   ├── PERFORMANCE_REPORT.md
│   ├── PRODUCTION_ARCHITECTURE.md
│   ├── PROJECT_STATUS.md
│   ├── RENDERER_PROFILE.md
│   └── RESEARCH_LOG.md
├── scripts/                          # Data ingestion, processing, and evaluation scripts
│   ├── analyze_dataset.py
│   ├── benchmark_architectures.py
│   ├── benchmark_models.py
│   ├── build_canonical_dataset.py
│   ├── build_offline_dataset.py
│   ├── build_release.py
│   ├── debug_gradients.py
│   ├── evaluate_characters.py
│   ├── experiment_data_scaling.py
│   ├── profile_production_model.py
│   ├── run_analysis.py
│   ├── run_evaluation_tests.py
│   ├── run_inference_tests.py
│   ├── run_renderer_tests.py
│   ├── simulate_collection.py
│   ├── test_conversion.py
│   ├── test_synthetic.py
│   └── validate_dataset.py
├── src/                              # Core Python package
│   ├── datasets/                     # Dataset loading, parsing, and PyTorch datasets
│   ├── evaluation/                   # Metrics (DTW, CER, WER, Frechet, Performance)
│   ├── inference/                    # Inference pipelines and logic
│   ├── interfaces/                   # Abstract base classes and typing interfaces
│   ├── metrics/                      # Additional reconstruction metrics
│   ├── models/                       # Neural network definitions (LSTM, Transformer, OCR CRNN)
│   ├── renderer/                     # SVG/PNG conversion and layout engine
│   ├── tokenizers/                   # Character-level text tokenizers (Devanagari)
│   ├── tools/                        # Web tools (e.g., dataset collection tool)
│   ├── training/                     # PyTorch training loops, loss functions, experiments
│   └── utils/                        # Config loading, visualizers, environments
├── tests/                            # Comprehensive Unit and Integration Tests
│   ├── fixtures/
│   ├── integration/
│   ├── test_continuous.py
│   ├── test_dataset.py
│   ├── test_evaluation*.py
│   ├── test_inference*.py
│   ├── test_ocr.py
│   ├── test_production.py
│   ├── test_renderer.py
│   └── test_visual_regression.py
├── .gitignore                        # Defines which files to exclude (e.g., data/, experiments/)
├── CHANGELOG.md                      # History of changes
├── Experiment_001_Report.md          # Output metrics for the first run
├── KAGGLE_RELEASE_REPORT.md          # Details of the Kaggle zip bundles
├── KAGGLE_SETUP.md                   # Instructions for running on Kaggle
├── Kaggle_Training.ipynb             # Jupyter Notebook for Kaggle execution
├── LICENSE                           # Project License
├── NEXT_TASK.md                      # Immediate next tasks planning
├── PROJECT_STATUS.md                 # High-level project state
├── PROJECT_TECHNICAL_REFERENCE.md    # Master documentation reference
├── README.md                         # Main repository entrypoint
├── STYLE_DATASET_ANALYSIS.md         # Analysis of available handwriting styles
├── TRAINING_STRATEGY.md              # Long-term ML strategy
├── VERSION                           # Current semantic version (v1.1.0)
├── character_difficulty.json         # Computed geometry difficulty per character
├── data_scaling_results.json         # Results of scaling experiments
├── main.py                           # Unified CLI entrypoint
├── requirements.txt                  # Standard Python dependencies
└── requirements_kaggle.txt           # Kaggle-specific dependencies
```
