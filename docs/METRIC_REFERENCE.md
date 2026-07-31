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

### Geometry Metrics (`src/evaluation/metrics/geometry.py`)

#### PathLengthDifferenceMetric (`path_length`)
- **Purpose**: Evaluates the absolute difference in the total Euclidean arc length of the generated curves.
- **Outputs**: `pred_length`, `target_length`, `length_difference`

#### BoundingBoxDifferenceMetric (`bounding_box`)
- **Purpose**: Computes geometric scaling differences by comparing bounding box width, height, and total area.
- **Outputs**: `width_difference`, `height_difference`, `area_difference`

#### SmoothnessScoreMetric (`smoothness`)
- **Purpose**: Computes angular variance between sequential coordinate line segments. A higher variance implies jittery, unsmooth handwriting.
- **Outputs**: `pred_smoothness`, `target_smoothness`, `smoothness_difference`

