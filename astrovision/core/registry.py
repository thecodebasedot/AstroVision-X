"""A tiny name -> factory registry used to make components swappable."""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, List, TypeVar

from .exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a string key to a factory, so configs can name components.

    >>> detectors = Registry("detector")
    >>> @detectors.register("threshold")
    ... def _build():
    ...     return "threshold-detector"
    >>> detectors.create("threshold")
    'threshold-detector'
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T] = None):
        """Register ``factory`` under ``name``; usable as a decorator."""
        def _decorate(fn: Callable[..., T]) -> Callable[..., T]:
            key = name.lower()
            if key in self._items:
                raise RegistryError(f"{self.kind} '{name}' is already registered")
            self._items[key] = fn
            return fn

        if factory is not None:
            return _decorate(factory)
        return _decorate

    def get(self, name: str) -> Callable[..., T]:
        key = str(name).lower()
        if key not in self._items:
            raise RegistryError(
                f"unknown {self.kind} '{name}'; available: {', '.join(self.names()) or '<none>'}"
            )
        return self._items[key]

    def create(self, name: str, *args, **kwargs) -> T:
        """Instantiate the component registered under ``name``."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return str(name).lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Registry {self.kind}: {', '.join(self.names())}>"
