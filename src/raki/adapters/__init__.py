from raki.adapters.alcove import AlcoveAdapter
from raki.adapters.alcove_pipeline import AlcovePipelineAdapter
from raki.adapters.loader import DatasetLoader, LoadError
from raki.adapters.protocol import SessionAdapter
from raki.adapters.redact import redact_sensitive
from raki.adapters.registry import AdapterRegistry
from raki.adapters.session_schema import SessionSchemaAdapter


def default_registry() -> AdapterRegistry:
    """Build the default adapter registry with all built-in adapters.

    Returns a fresh ``AdapterRegistry`` each call so that callers can
    customise their copy without affecting others.

    Registration order matters for format detection: more-specific adapters
    must be registered *before* generic ones so that ``detect()`` is tried in
    the right sequence.  ``AlcovePipelineAdapter`` (directory-based) is
    registered before ``AlcoveAdapter`` (single-file JSON) to prevent the
    generic adapter from consuming pipeline export directories.
    """
    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    registry.register(AlcovePipelineAdapter())
    registry.register(AlcoveAdapter())
    return registry


__all__ = [
    "AdapterRegistry",
    "AlcoveAdapter",
    "AlcovePipelineAdapter",
    "DatasetLoader",
    "LoadError",
    "SessionAdapter",
    "SessionSchemaAdapter",
    "default_registry",
    "redact_sensitive",
]
