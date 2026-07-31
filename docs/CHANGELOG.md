# Changelog

## [Unreleased] - Phase 6 Framework Upgrade
### Added
- **Subsystem 1: Configuration & Error Handling**
  - Implemented `src/renderer/exceptions.py` with custom exception hierarchy (`RendererError`, `InvalidTrajectoryError`, `ExporterError`, `LayoutError`, `CacheError`, `PluginRegistrationError`).
  - Added strict versioning to `RenderingConfig` (`renderer_version`, `layout_version`, `exporter_version`).
  - Wrapped `RenderingEngine.render()` pipeline in robust try/except blocks to gracefully catch and raise custom exceptions.
  - Added unit tests for invalid trajectory handling.
- **Subsystem 2: Exporter Contract Upgrade**
  - Designed and enforced a robust 4-step lifecycle contract in `src/renderer/exporters/base.py`: `initialize()`, `export()`, `validate()`, and `cleanup()`.
  - Refactored `SVGExporter`, `PNGExporter`, `PDFExporter`, `GIFExporter`, and `MP4Exporter` to inherit from `BaseExporter`.
  - Upgraded `RenderingEngine` to invoke `validate()` and ensure `cleanup()` is called via `finally` blocks, preventing temporary file leaks (e.g. from CairoSVG).
- **Subsystem 3: Multi-Level Caching System**
  - Upgraded `src/renderer/cache.py` to create a deterministic hash key incorporating trajectory metadata, geometrical paths, and plugin/renderer versioning.
  - Implemented `serve_from_cache` to bypass rasterization and file IO for exact rendering matches.
  - Added unit test validation to ensure cache hits skip exporter execution and return identical results.
- **Subsystem 4: Advanced Layout Engine**
  - Created `src/renderer/layout/advanced.py` housing `ParagraphLayout` and `NotebookLayout`.
  - Implemented dynamic geometric wrapping algorithm for paragraph structures.
  - Integrated `NotebookLayout` to forcefully align multiline text against consistent baseline heights.
- **Subsystem 5: Profiling & Visual Regression**
  - Integrated `time.perf_counter()` hooks directly into `RenderingEngine.render()` pipeline, logging microsecond-precision benchmarks to `docs/RENDERER_PROFILE.md`.
  - Implemented deterministic `test_visual_regression_svg` generating hardcoded baseline hashes in `tests/fixtures/svg/baseline_test_001.svg` to strictly catch unintended geometric scaling or configuration drifts.

## [Unreleased] - Phase 7 Evaluation Framework
### Added
- **Subsystem 1: Metric Interfaces & Registry**
  - Implemented `BaseMetric` contract defining `name`, `version`, `description`, `evaluate`, `validate`, and `summarize`.
  - Expanded `@Registry.register_metric` dynamic plugin system.
  - Implemented `EvaluationConfig` to freeze benchmark state and metadata (git commit, metric versions).
  - Generated initial `docs/METRIC_REFERENCE.md` catalog.
- **Subsystem 2: Trajectory Metrics**
  - Created `src/evaluation/metrics/trajectory.py`.
  - Implemented geometric trajectory metrics: `DTWMetric`, `FrechetMetric`, `StrokeCountDifferenceMetric`, and `EndpointErrorMetric`.
  - Used `fastdtw` and `similaritymeasures` wrapped in safe import checks for robust, optimized time-series evaluation.
- **Subsystem 3: Geometry Metrics**
  - Created `src/evaluation/metrics/geometry.py`.
  - Implemented static geometry metrics: `PathLengthDifferenceMetric`, `BoundingBoxDifferenceMetric`, and `SmoothnessScoreMetric` (computing angular variance).
- **Subsystem 4: Performance & Rendering Metrics**
  - Created `src/evaluation/metrics/performance.py`.
  - Implemented `SVGGenerationTimeMetric` integrating `psutil` memory tracking and generating temporary renders inside the metric scope.
  - Implemented `InferenceLatencyMetric` and `SystemMemoryUsageMetric` tracking model metadata cleanly attached via the `TrajectorySample.extensions` schema.
- **Subsystem 5: Visualization & Reports**
  - Created `src/evaluation/reports/generators.py` supporting deterministic JSON, Markdown, and CSV summary generations.
  - Created `src/evaluation/visualization/plotters.py` providing Matplotlib-based trajectory overlays for visual evaluations.
- **Subsystem 6: Batch Benchmarking**
  - Implemented `src/evaluation/benchmarks/orchestrator.py` `BenchmarkOrchestrator` to automatically execute all registered metrics over a batch of predictions and targets.
  - Hardened execution loop with graceful degradation (ignoring missing module `ImportError` dynamically while preserving other metrics).
  - Designed automated report generation hook bridging `EvaluationConfig` tracking to output file dumps.
