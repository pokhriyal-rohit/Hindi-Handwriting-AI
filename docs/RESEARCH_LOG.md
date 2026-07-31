# Research Log

This document records the scientific narrative of the Hindi Handwriting AI project. 
Every experiment follows the scientific method: **Observation → Hypothesis → Experiment → Benchmark → Conclusion → Next Experiment**.

---

## Experiment 000 (Milestone A)

**Question**
Can the baseline LSTM learn a deterministic end-to-end mapping from text to canonical trajectories?

**Hypothesis**
Yes, a standard Encoder-Decoder LSTM using MSE and BCE loss should overfit a synthetic, noise-free dataset.

**Dataset**
Synthetic-5 (Deterministic sine waves)

**Result**
Training converged rapidly. Rendered SVGs matched the mathematical ground truth.

**Decision**
The pipeline is functionally sound. Proceed to collect genuine human data.

---

## Experiment 001 (Planned)

**Question**
Can the baseline LSTM overfit real human handwriting?

**Hypothesis**
Yes, but training loss will plateau higher than synthetic data due to biological variance and sensor noise.

**Dataset**
Custom Hindi Online v1.0 (Pilot 100 samples)

**Result**
[Pending collection...]

**Decision**
[Pending...]
