# Inference Framework Architecture

## Overview
The Inference Framework acts as the canonical serving layer for the entire project. It unifies the model, tokenizer, and rendering engines behind a single, abstracted `InferencePipeline`. It is strictly decoupled from UI logic (REST APIs, GUIs, CLI), acting purely as a Python API entrypoint.

## Core Pipeline Architecture
The generation pipeline executes strictly linearly:

1. **Text Input**: Receives raw Unicode string.
2. **Preprocessing**: Normalizes spaces, characters, and standardizes scripts.
3. **Tokenizer**: Converts text into numerical indices.
4. **Predictor (Inference)**: Passes indices through neural network to output raw `dx, dy, pen_state`.
5. **Coordinate Reconstruction**: Upgrades differential tensors into the canonical `TrajectorySample` (absolute coords).
6. **Post Processing**: Optional smoothing, noise removal, and pressure estimation on the Trajectory.
7. **Layout Engine**: Wraps or spaces text (Paragraph, Notebook, Page).
8. **Rendering Engine**: Simulates ink flow and outputs absolute geometric paths.
9. **Exporter**: Flushes coordinates to disk (SVG, PNG, PDF, MP4).
10. **InferenceResult**: Wraps all generated artifacts, cache statuses, and execution latencies into a canonical payload.

## Core Entities
- **InferenceSession**: An object that initializes and retains the expensive components (Model weights in GPU VRAM, Renderer instances, Tokenizer Vocab). It is long-lived and handles multiple inference requests.
- **InferenceConfig**: The definition payload storing hardware specifications (`device: 'cuda'`), precision (`float16`), random seeds, and specific rendering versions.
- **InferenceResult**: The ultimate payload. Contains timing metadata, warning logs, the raw `TrajectorySample`, and final export file paths.

## Lifecycle Hooks
The pipeline triggers external callback hooks at specific phases. This ensures profiling metrics, debugging logs, and external visualization systems can attach dynamically without hardcoding them into the pipeline execution path:
- `on_inference_start()`
- `on_prediction_complete()`
- `on_rendering_start()`
- `on_rendering_complete()`
- `on_inference_end()`

## Future Deployment Strategy
Because the Inference Framework is self-contained and heavily cached, deploying to a production REST API is trivial (using FastAPI/Flask to simply instantiate a global `InferenceSession` and route `/generate` payloads directly to the pipeline).
