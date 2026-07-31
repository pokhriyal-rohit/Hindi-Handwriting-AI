# Next Task

**Target Phase:** Phase 2 (Dataset Analysis)

**Immediate Task:**
Build the **Dataset Analysis Module**.

**Requirements for this task:**
- Write a Python script/module that iterates over the parsed `Trajectory` objects from the dataset.
- Extract and calculate:
  - Number of samples and writers.
  - Average strokes per sample, points per stroke, and sequence lengths.
  - Coordinate ranges and delta ($\Delta x, \Delta y$) movement distributions.
  - Pen state distributions.
- Generate visualizations (histograms, scatter plots) for coordinate distributions, stroke lengths, sequence lengths, and point density.
- Do NOT implement any tokenizers or models during this step.
