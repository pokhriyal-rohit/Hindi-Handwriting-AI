# Changelog

## [Unreleased] - Phase 6 Framework Upgrade
### Added
- **Subsystem 1: Configuration & Error Handling**
  - Implemented `src/renderer/exceptions.py` with custom exception hierarchy (`RendererError`, `InvalidTrajectoryError`, `ExporterError`, `LayoutError`, `CacheError`, `PluginRegistrationError`).
  - Added strict versioning to `RenderingConfig` (`renderer_version`, `layout_version`, `exporter_version`).
  - Wrapped `RenderingEngine.render()` pipeline in robust try/except blocks to gracefully catch and raise custom exceptions.
  - Added unit tests for invalid trajectory handling.
