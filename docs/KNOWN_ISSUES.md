# Known Issues

This document tracks unresolved bugs, architectural tech-debt, and potential risks identified during development.

## Open Issues
- **CRITICAL DATASET BLOCKER:** The Dataset Analysis module confirmed that ALL downloaded datasets (including `IIIT-HW-Hindi_v1` and `Dataset_hindi_character`) are **strictly offline (image-based) datasets**. They consist of `.jpg` images and text annotations. There is **zero online coordinate (trajectory) data** available. 
  - *Impact:* Online trajectory synthesis (SVG generation, pen state, animated drawing) requires online training data (X, Y, Pen_State). 
  - *Constraint Conflict:* The user policy states "Synthetic font-generated trajectories may only be used for optional pretraining... Human handwriting should always be the primary source". Without an online Devanagari dataset (like LipiTK), we are blocked from training the final production model.
- **Tesla T4 Memory Constraints:** Long sequence lengths (full pages) remain a severe risk for potential Transformer prototypes. This will be carefully evaluated during Phase 4 Benchmarking.
