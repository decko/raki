from raki.adapters.alcove import AlcoveAdapter
from raki.adapters.loader import DatasetLoader, LoadError
from raki.adapters.protocol import SessionAdapter
from raki.adapters.redact import redact_sensitive
from raki.adapters.registry import AdapterRegistry
from raki.adapters.session_schema import SessionSchemaAdapter


def default_registry() -> AdapterRegistry:
    """Build the default adapter registry with all built-in adapters.

    Returns a fresh ``AdapterRegistry`` each call so that callers can
    customise their copy without affecting others.
    """
    registry = AdapterRegistry()
    registry.register(SessionSchemaAdapter())
    registry.register(AlcoveAdapter())
    return registry


__all__ = [
    "AdapterRegistry",
    "AlcoveAdapter",
    "DatasetLoader",
    "LoadError",
    "SessionAdapter",
    "SessionSchemaAdapter",
    "default_registry",
    "redact_sensitive",
]
