# Evaluation Architecture

## Overview
The Evaluation Framework is a standalone subsystem responsible for objectively measuring handwriting quality, rendering efficiency, and model performance. It is completely decoupled from any specific neural network or renderer implementation, operating strictly on the canonical `TrajectorySample` and standard performance hooks.

## Project Structure
```text
src/
    evaluation/
        metrics/         # Individual metric implementations
        reports/         # Markdown/JSON/CSV report generators
        benchmarks/      # Batch execution orchestrators
        visualization/   # Trajectory overlays, DTW paths, heatmaps
        statistics/      # Math aggregators (Mean, Std, Confidence Intervals)
        comparison/      # Model A vs Model B comparators
```

## The Metric Interface
Every metric must adhere to a strict 4-step contract, ensuring consistent lifecycle management and reporting:

```python
class BaseMetric:
    def name(self) -> str:
        """Returns the canonical name of the metric."""
        pass
        
    def validate(self, prediction: TrajectorySample, target: TrajectorySample) -> bool:
        """Verifies the inputs are valid for this specific metric."""
        pass

    def evaluate(self, prediction: TrajectorySample, target: TrajectorySample) -> dict:
        """Executes the core calculation and returns a dictionary of scores."""
        pass

    def summarize(self, results: List[dict]) -> dict:
        """Aggregates a batch of evaluation results into statistical summaries."""
        pass
```

## Benchmark Workflow
1. **Initialize Suite**: Load configuration, freeze random seeds, capture hardware/git metadata.
2. **Batch Execution**: Orchestrate predictions across the dataset.
3. **Metric Evaluation**: Pass `<Prediction, Target>` pairs through all registered metric plugins.
4. **Statistical Aggregation**: Reduce arrays of scores into Mean/Median/Variance using the `statistics` subsystem.
5. **Report Generation**: Export `evaluation_report.md`, `evaluation_report.json`, and `benchmark_summary.csv`.

## Future Extension Strategy
Because all metrics are decoupled and loaded dynamically via `@Registry.register_metric("name")`, new state-of-the-art metrics (e.g., Learned Perceptual Trajectory Patch Similarity) can be added instantly without altering the Benchmark Orchestrator.
