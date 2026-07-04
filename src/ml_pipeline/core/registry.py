"""Generic string-keyed registry used by every swappable component in the pipeline.

Each stage (loaders, splitters, imputers, encoders, models, samplers, ...) owns a
module-level ``Registry`` instance and populates it with the ``register`` decorator.
Adding a new implementation therefore never requires touching orchestration code:
create one file, register the class/factory, and import it from the subpackage's
``__init__``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named mapping from string keys to implementations (classes or factories)."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: dict[str, T] = {}

    @property
    def name(self) -> str:
        """Human-readable registry name, used in error messages."""
        return self._name

    def register(self, key: str) -> Callable[[T], T]:
        """Return a decorator that registers the decorated object under ``key``.

        Raises:
            KeyError: if ``key`` is already registered (duplicate registrations
                almost always indicate a copy-paste mistake).
        """

        def decorator(item: T) -> T:
            if key in self._items:
                raise KeyError(
                    f"'{key}' is already registered in the '{self._name}' registry."
                )
            self._items[key] = item
            return item

        return decorator

    def get(self, key: str) -> T:
        """Look up an implementation by key.

        Raises:
            KeyError: with the list of available keys, so config typos are
                self-explanatory.
        """
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(
                f"Unknown {self._name} '{key}'. Available: {self.available()}"
            ) from None

    def available(self) -> list[str]:
        """Sorted list of registered keys."""
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, keys={self.available()})"
