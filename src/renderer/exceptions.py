class RendererError(Exception):
    """Base exception for the Rendering Engine."""
    pass

class InvalidTrajectoryError(RendererError):
    """Raised when the canonical TrajectorySample is invalid or empty."""
    pass

class PluginRegistrationError(RendererError):
    """Raised when a requested plugin is missing or invalid."""
    pass

class ExporterError(RendererError):
    """Raised when an exporter fails (e.g., missing dependencies)."""
    pass

class LayoutError(RendererError):
    """Raised when layout calculation fails (e.g., invalid margins)."""
    pass

class CacheError(RendererError):
    """Raised when cache corruption or I/O failure occurs."""
    pass
