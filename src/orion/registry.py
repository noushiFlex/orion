"""Name -> class registries. Sources, features and models plug in by decorating their class."""

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, cls: type[T]) -> type[T]:
        name = cls.name  # type: ignore[attr-defined]
        if name in self._items:
            raise ValueError(f"{self.kind} {name!r} registered twice")
        self._items[name] = cls
        return cls

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {sorted(self._items)}") from None

    def names(self) -> list[str]:
        return sorted(self._items)
