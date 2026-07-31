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
