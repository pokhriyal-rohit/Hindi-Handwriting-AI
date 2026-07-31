# Architectural Decisions Record

This document records all significant architectural decisions made during the project.

## Decision 1: Strict Evidence-Based Architecture Selection
**Date:** Current Phase
**Context:** Premature optimization and architecture selection (e.g., choosing Discrete-Token Transformers immediately) violate industrial long-term engineering principles.
**Decision:** The project will NOT commit to any architecture (LSTM, Transformer, Diffusion) or coordinate representation (Continuous vs Discrete) until comprehensive Dataset Analysis and Prototyping/Benchmarking are complete.
**Consequences:** 
- Requires building a robust Dataset Analysis module.
- Requires building a decoupled Coordinate Representation interface that supports multiple downstream models.

## Decision 2: Stroke Order Preservation
**Date:** Current Phase
**Context:** Should we algorithmically normalize the order of strokes in Devanagari (e.g., always draw the shirorekha last)?
**Decision:** Never normalize handwriting stroke order unless the dataset itself contains inconsistent ordering/corruption. 
**Consequences:** The model will learn natural human handwriting behavior rather than artificially reordered trajectories.

## Decision 3: Synthetic Data Policy
**Date:** Current Phase
**Context:** Using fonts to generate synthetic data for handwriting synthesis.
**Decision:** Synthetic font-generated trajectories will NOT be used as primary training data, only for optional pre-training if real datasets are insufficient.
