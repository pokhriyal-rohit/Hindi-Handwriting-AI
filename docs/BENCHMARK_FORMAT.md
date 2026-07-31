# Benchmark Report Format

Every time the `BenchmarkOrchestrator` executes a suite, it dumps deterministic output files tracking exactly what was run, what code version was used, and the resulting metrics.

## `evaluation_report.json`
A machine-readable dump containing the full configuration and statistical output.
```json
{
  "metadata": {
    "evaluation_version": "1.0.0",
    "dataset_version": "unknown",
    "renderer_version": "unknown",
    "representation_version": "unknown",
    "model_version": "unknown",
    "report_version": "1.0.0",
    "metric_versions": {
      "dtw": "1.0.0",
      "frechet": "1.0.0"
    }
  },
  "results": {
    "dtw_mean": 1.5,
    "svg_time_mean": 0.04
  }
}
```

## `evaluation_report.md`
A human-readable markdown version summarizing the `results` for quick visual inspection, usually attached to GitHub Pull Requests or model tracking dashboards.

## `benchmark_summary.csv`
A flattened CSV table where each execution appends a new row containing both metadata and metric scores, allowing for long-term historical tracking and Python Pandas/Matplotlib trend plotting.
