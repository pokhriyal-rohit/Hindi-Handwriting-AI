# Hindi Handwriting AI

A modular, reproducible research platform for generating online Hindi (Devanagari) handwriting as time-series trajectories — sequences of (x, y, pen_state) coordinates.

> **Engineering quality: 9/10. Research is just beginning.**  
> The platform is frozen at `v1.0.0-research-platform`. Future work evolves hypotheses, not infrastructure.

---

## What This Is

Most handwriting generation projects produce *images*. This project generates *trajectories* — the actual sequence of pen movements in order, with stroke timing and pen-lift events. This enables animation, style transfer, and proper generative modelling of the drawing process itself.

The repository is designed around a single principle from [`docs/OATH.md`](docs/OATH.md):

> **The repository exists to discover truth about handwriting generation, not to confirm assumptions.**

---

## Architecture Overview

```
Canvas Collector → Canonical Schema → Training Loop → Renderer
                        │                   │
                        ▼                   ▼
                    Converters          BaselineLSTM
                    Scalers             MDN / Transformer
                    Tokenizer           Production Model
                        │
                        ▼
                 Evaluation Framework
                 (DTW, Fréchet, Endpoint Error)
```

See [`PROJECT_AUDIT_REPORT.md`](PROJECT_AUDIT_REPORT.md) for a full module-by-module analysis.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# For PNG/PDF rendering also install:
# pip install cairosvg Pillow
```

### 2. Collect real handwriting

```bash
python src/tools/dataset_collector/server.py
# Open http://localhost:8080 in your browser
# Follow the prompts: 20 characters, 10 ligatures, 20 words, 10 sentences
```

Data is saved to `data/raw/custom_hindi/<writer_id>/<word>/sample_xxx.json`.

### 3. Validate the dataset (run before every training job)

```bash
python scripts/validate_dataset.py
```

This checks all 9 quality criteria per sample and produces a `validation_report.json`.

### 4. Analyse the dataset

```bash
python scripts/analyze_dataset.py
```

Prints statistics (strokes, points, duration, speed, path length) and generates `dataset_manifest.yaml`.

### 5. Train the baseline

```bash
python -m src.training.cli train --epochs 100
```

With options:
```bash
python -m src.training.cli train \
    --epochs 100 \
    --batch-size 32 \
    --val-split 0.1 \
    --exp-id exp_010
```

Checkpoints, SVG predictions, and metrics land in `experiments/exp_XXX/`.

### 6. Resume a run

```bash
python -m src.training.cli train --resume experiments/exp_010/checkpoint_50.pt
```

---

## Project Layout

```
Hindi-Handwriting-AI/
├── src/
│   ├── datasets/          # Canonical schema + converters + scalers
│   ├── models/            # BaselineLSTM, MDN, Transformer, Production
│   ├── training/          # Training loop, loss, experiment tracker, CLI
│   ├── inference/         # Session, pipeline, hooks, cache, result
│   ├── renderer/          # SVG/PNG/PDF/GIF/MP4 rendering engine
│   ├── evaluation/        # DTW, Fréchet, geometry, performance metrics
│   └── tools/
│       └── dataset_collector/  # HTML5 Canvas web collector + server
├── scripts/
│   ├── analyze_dataset.py      # Dataset statistics + manifest generation
│   ├── validate_dataset.py     # Pre-training quality gate (9 checks)
│   └── simulate_collection.py  # Generates augmented mock data for smoke tests
├── experiments/           # One directory per training run (auto-created)
├── data/
│   └── raw/
│       └── custom_hindi/  # Collected trajectory JSON files (hierarchical)
├── docs/                  # Architecture docs, benchmark history, research log
├── tests/                 # 21 test files covering all framework components
├── requirements.txt
├── PROJECT_AUDIT_REPORT.md
└── README.md
```

---

## Research Workflow

Every experiment follows a strict four-field template defined in [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md):

```markdown
## Question
What specific question are we trying to answer?

## Hypothesis
What do we expect to happen?

## Evidence Required
What specific metrics or observations must change to support the hypothesis?

## Success Criterion
What is the strict mathematical threshold for success?
```

Results are appended to [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and never overwritten.

---

## Current Experiments

| ID | Dataset | Model | Train Loss | Val Loss | Status |
|---|---|---|---|---|---|
| 000 | Synthetic-5 | BaselineLSTM | 4.314 | N/A | ✅ Complete |
| 001 | Augmented-410 | BaselineLSTM | 1.156 | N/A | ⚠️ Smoke test only |

The **Real Human Benchmarks** section of `docs/BENCHMARKS.md` is pending data collection.

---

## Data Types

The project uses strict terminology to prevent mixing data types in analysis:

| Type | Definition |
|---|---|
| **Synthetic** | Generated entirely from fonts or procedural math |
| **Augmented** | Synthetic trajectories with added noise/transforms |
| **Human** | Recorded from a real person via the collector |

---

## Canonical Schema

Every sample — regardless of source — is represented as a `TrajectorySample`:

```python
TrajectorySample(
    sample_id="...",
    writer_id="...",       # supports style encoders
    script="devanagari",
    language="hi",
    text="नमस्ते",
    strokes=[
        Stroke(stroke_id=0, points=[
            Point(x=10.0, y=5.0, pen_state=1, timestamp=0.0),
            ...
        ])
    ],
    metadata=DatasetMetadata(
        dataset_name="custom_collector",
        is_synthetic=False,         # CRITICAL flag
        sampling_rate_hz=60.0
    )
)
```

See [`docs/CANONICAL_SCHEMA.md`](docs/CANONICAL_SCHEMA.md) for full specification.

---

## Key Documents

| Document | Purpose |
|---|---|
| [`docs/OATH.md`](docs/OATH.md) | Research philosophy |
| [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) | Per-experiment scientific record |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Append-only benchmark history |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architectural decisions record |
| [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) | Open bugs and blockers |
| [`PROJECT_AUDIT_REPORT.md`](PROJECT_AUDIT_REPORT.md) | Full technical audit at v1.0.0 |

---

## Next Milestone

**Milestone C: First Genuine Human Handwriting Baseline**

```
□ Collect 100+ real samples via the web collector
□ Run validate_dataset.py — fix any rejections
□ Run analyze_dataset.py — verify statistics
□ Train BaselineLSTM for 100 epochs (batch=32, val-split=0.1)
□ Log both Train Loss and Val Loss curves
□ Run BenchmarkOrchestrator on the validation set
□ Log Experiment 001 in RESEARCH_LOG.md (all four fields)
□ Append to BENCHMARKS.md under Real Human Benchmarks
```

---

## License

Research use only. See individual dataset licenses in [`docs/DATASET_COMPARISON.md`](docs/DATASET_COMPARISON.md).
