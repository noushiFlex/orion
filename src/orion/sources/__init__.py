"""Data sources. Importing this package registers every source in SOURCES."""

from orion.sources.base import SOURCES, Source
from orion.sources import clubelo, football_data, transfermarkt, understat  # noqa: F401  (registration)

__all__ = ["SOURCES", "Source"]
