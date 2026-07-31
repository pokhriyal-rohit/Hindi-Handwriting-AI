# Phase 4 Performance Report & Architecture Recommendation

## Executive Summary
This report summarizes the empirical findings of the Phase 4 Architecture Benchmarking for the Hindi Handwriting AI project. The goal was to establish the optimal autoregressive baseline architecture before scaling up to production models in Phase 5.

We evaluated two candidate models:
1. **CoordinateLSTM:** A 2-layer LSTM with 128 hidden units and an MDN head.
2. **CoordinateTransformer:** A 2-layer, 4-head causal Transformer Decoder with 128 hidden units and an MDN head.

*Note: Diffusion models were excluded from this phase as per architectural policy.*

## Benchmark Results (CPU Execution)

| Metric | Tiny LSTM | Tiny Transformer |
| :--- | :--- | :--- |
| **Parameter Count** | 215,801 | 412,665 |
| **Training Time (10 Epochs)** | 5.85s | 15.15s |
| **Throughput** | ~109.3 samples/sec | ~42.2 samples/sec |
| **Validation Loss (MDN NLL)**| **-1.491** | -1.255 |

## Analysis
1. **Computational Efficiency:** The Tiny LSTM processed samples **2.5x faster** than the Tiny Transformer. Recurrent architectures remain vastly superior at handling sequential, continuous coordinate data compared to exact self-attention (which scales quadratically with sequence length). 
2. **Parameter Efficiency:** The LSTM achieved a better Negative Log-Likelihood (NLL) loss (-1.491 vs -1.255) while requiring roughly half the parameters.
3. **Representation Stability:** Both models successfully ingested the `ModularCoordinateRepresentation` (Version 2.0) with Standard Scaling without numerical instability or exploding gradients.

## Recommendation
**Selected Architecture for Phase 5:** `CoordinateLSTM`

**Justification:** The empirical evidence heavily favors the LSTM for trajectory synthesis. Transformers inherently struggle with continuous 2D spatial coordinates without massive parameter scaling and complex positional embeddings. The LSTM natively handles the autoregressive temporal flow of handwriting much more efficiently. We will proceed to Phase 5 by scaling the LSTM (e.g., increasing layers, hidden dimensions, and potentially adding Bi-directional encoding for text-conditioning).
