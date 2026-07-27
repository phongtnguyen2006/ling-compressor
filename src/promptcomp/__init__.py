"""Local, deterministic, non-generative linguistic prompt compressor."""

from .pipeline import compress
from .types import CompressionResult

__version__ = "0.1.0"

__all__ = ["compress", "CompressionResult", "__version__"]
