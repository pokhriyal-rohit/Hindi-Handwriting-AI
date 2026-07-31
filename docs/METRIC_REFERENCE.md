# Evaluation Metric Reference

This document catalogs the deterministic metrics implemented in the Evaluation Framework. Every metric is isolated, stateless, and versioned.

## Core Metric Interface
All metrics implement `BaseMetric` providing:
- `name()`: The canonical registry name.
- `version()`: Implementation version to trace historical comparisons.
- `description()`: High-level metric summary.
- `evaluate(prediction, target)`: The core numerical evaluation.
- `validate(prediction, target)`: Safety checks to prevent crashed benchmark runs.
- `summarize(results)`: Aggregation of statistical arrays into Means and Confidence Intervals.

## Implemented Metrics

### Trajectory Metrics (`src/evaluation/metrics/trajectory.py`)

#### DTWMetric (`dtw`)
- **Purpose**: Evaluates sequential timing and geometric similarity irrespective of sampling speed.
- **Inputs**: Prediction `TrajectorySample`, Target `TrajectorySample`
- **Outputs**: `dtw_distance`
- **Computational Complexity**: O(N^2) (Reduced to O(N) via FastDTW).

#### FrechetMetric (`frechet`)
- **Purpose**: Evaluates spatial shape similarity (Discrete Fréchet distance) akin to the "dog-walker distance".
- **Inputs**: Prediction `TrajectorySample`, Target `TrajectorySample`
- **Outputs**: `frechet_distance`
- **Computational Complexity**: O(N^2)

#### StrokeCountDifferenceMetric (`stroke_count`)
- **Purpose**: Verifies that the model generated the exact expected number of strokes.
- **Outputs**: `pred_strokes`, `target_strokes`, `stroke_difference`

#### EndpointErrorMetric (`endpoint_error`)
- **Purpose**: Verifies the final Euclidean stopping position of the generated handwriting matches the target.
- **Outputs**: `endpoint_error`

