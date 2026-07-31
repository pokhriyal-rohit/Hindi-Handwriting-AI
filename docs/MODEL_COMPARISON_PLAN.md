# Model Comparison Plan (Phase 4)

## Purpose of Benchmarking
The purpose of this benchmark is to **scientifically determine the optimal generative architecture** for trajectory-based Hindi handwriting synthesis. We will NOT default to production-scale models or hype-driven architectures (like Diffusion). Instead, we will evaluate lightweight baselines based on empirical evidence (training speed, memory footprint, reconstruction fidelity, and generalization).

## Candidate Architectures
To keep the benchmark fair and computationally comparable, all models will be designed as "Tiny" baselines with roughly equivalent parameter counts (~100k - 500k).

1. **Tiny GRU (Gated Recurrent Unit) + MDN**
   - *Reason:* Classic, highly efficient sequence model. Less prone to vanishing gradients than Vanilla RNNs, and requires fewer parameters than LSTM. Traditionally strong at online handwriting (e.g., Alex Graves' work).
2. **Tiny LSTM (Long Short-Term Memory) + MDN**
   - *Reason:* The industry standard for temporal sequences before Transformers. Better at capturing long-term dependencies (crucial for long words or full sentences) compared to GRU, at the cost of slightly more parameters.
3. **Tiny Causal Transformer Decoder + MDN**
   - *Reason:* State-of-the-art for sequence modeling due to parallelizable training and exact self-attention over long contexts. However, Transformers can struggle with continuous coordinate regressions compared to discrete tokens. We must benchmark if it outperforms recurrence.

*Note: Diffusion models are explicitly excluded from this initial benchmark phase and will only be considered if all autoregressive baselines fail to meet the success criteria.*

## Training Protocol
- **Dataset:** Stage 1 `SyntheticTrajectoryGenerator` (Bootstrap data).
- **Representation:** Modular Continuous Coordinate Representation (pluggable features, scaled).
- **Hyperparameters:** Kept strictly comparable across candidates (e.g., hidden size ~128, layers ~2).
- **Optimizer:** AdamW.
- **Batch Size:** 32.
- **Epochs:** 10 (lightweight validation loop).

## Evaluation Metrics
1. **System Metrics:**
   - Training Time (Total and Samples/sec)
   - GPU Memory Usage (Peak MB)
   - Checkpoint Size (MB)
   - Inference Speed (ms per sequence)
2. **Numerical Reconstruction Metrics:**
   - Validation Loss (Negative Log-Likelihood for MDN / MSE)
   - RMSE (Root Mean Square Error) between generated and ground truth trajectories
   - MAE (Mean Absolute Error)
   - Maximum Coordinate Error
   - Endpoint Error
3. **Qualitative Generation Metrics:**
   - Visual smoothness of rendered SVG.
   - Continuity of strokes.

## Benchmark Methodology
1. Train each model on the identical frozen `TrajectorySample` pipeline.
2. At the end of training, the framework will automatically generate a machine-readable `benchmark.json`.
3. The JSON will comprehensively log: Model Name, Git Commit, Dataset/Representation Versions, Hyperparameters, Seed, Hardware, and all Evaluation Metrics.

## Decision Criteria
An architecture will be selected for Phase 5 (Production Scale) based on:
1. Lowest Validation NLL/RMSE.
2. Best visual smoothness (no jitter, correct pen lifts).
3. Most efficient inference speed (critical for real-time generation).
4. Stable convergence without massive memory spikes.

## Expected Risks
- **Transformer Collapse:** Continuous coordinates (vs discrete tokens) often cause mode collapse in Transformers without extensive positional encoding or patching.
- **GRU/LSTM Bottleneck:** Recurrent models might struggle to compress extremely long coordinate sequences into a single hidden state vector.

## Success Conditions
The benchmarking phase will be considered successful when:
1. All three candidate architectures have successfully trained and generated reproducible `benchmark.json` logs.
2. At least one architecture produces visually distinguishable handwriting trajectories that do not devolve into random noise.
3. A clear, data-backed architecture recommendation is made and documented in `docs/DECISIONS.md`.
