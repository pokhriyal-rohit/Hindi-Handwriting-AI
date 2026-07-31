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
*(More metrics will be added here as Subsystem 2 and Subsystem 3 are implemented).*
