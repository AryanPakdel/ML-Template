"""The generic string-keyed Registry: register, lookup, and error behaviour."""

from __future__ import annotations

import pytest

from ml_pipeline.core.registry import Registry


def _fresh_registry() -> Registry[type]:
    """A throwaway registry with two entries registered."""
    registry: Registry[type] = Registry("widget")

    @registry.register("alpha")
    class Alpha:
        pass

    @registry.register("beta")
    class Beta:
        pass

    return registry


def test_register_and_get() -> None:
    """get() returns exactly the object that was registered."""
    registry = _fresh_registry()
    assert registry.get("alpha").__name__ == "Alpha"
    assert "alpha" in registry
    assert len(registry) == 2


def test_available_is_sorted() -> None:
    """available() lists all keys in sorted order."""
    registry = _fresh_registry()
    assert registry.available() == ["alpha", "beta"]


def test_duplicate_key_raises() -> None:
    """Re-registering an existing key is a KeyError (copy-paste guard)."""
    registry = _fresh_registry()
    with pytest.raises(KeyError, match="already registered"):
        registry.register("alpha")(object)


def test_unknown_key_error_lists_available() -> None:
    """Unknown-key errors name the registry and enumerate valid keys."""
    registry = _fresh_registry()
    with pytest.raises(KeyError) as excinfo:
        registry.get("gamma")
    message = str(excinfo.value)
    assert "widget" in message
    assert "alpha" in message
    assert "beta" in message
