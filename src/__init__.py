"""
HW5 NLP Pipeline - Source Package

Modules:
    - config: Configuration constants and parameters
    - utils_io: File I/O utilities
    - load_articles: Article loading and preprocessing
    - coref_contexts: Coreference resolution and context extraction
    - extract_descriptions: Entity description extraction
    - embed_cluster: Embedding and clustering pipeline
    - manual_eval_helpers: Helpers for manual evaluation
    - freq_tables: Frequency table generation
    - stats_tests: Statistical hypothesis testing
"""

__version__ = "0.1.0"

# Only import config by default (lightweight)
from . import config

# Other modules are imported on-demand to avoid heavy dependency loading
__all__ = [
    "config",
    "utils_io",
    "load_articles",
    "coref_contexts",
    "extract_descriptions",
    "embed_cluster",
    "manual_eval_helpers",
    "freq_tables",
    "stats_tests",
]


def __getattr__(name):
    """Lazy import of modules to avoid loading heavy dependencies upfront."""
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
