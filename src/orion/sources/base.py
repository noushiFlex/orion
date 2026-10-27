"""Source contract: fetch() writes raw files (each recorded in the manifest); load() returns a validated frame."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import pandas as pd

from orion.config import Config
from orion.paths import RAW
from orion.registry import Registry


class Source(ABC):
    name: ClassVar[str]
    folder: ClassVar[str]  # directory under data/raw/

    @property
    def raw_dir(self) -> Path:
        return RAW / self.folder

    @abstractmethod
    def fetch(self, cfg: Config, refresh: bool = False) -> list[Path]:
        """Download missing files. Past seasons are final; the current season is always re-fetched."""

    @abstractmethod
    def load(self, cfg: Config) -> pd.DataFrame:
        """Read raw files for cfg's leagues and seasons into a schema-validated frame."""


SOURCES: Registry[Source] = Registry("source")


def require_csv_header(marker: bytes):
    def check(body: bytes) -> None:
        if marker not in body[:64]:
            raise ValueError(f"unexpected content (no {marker!r} header): {body[:80]!r}")
    return check
