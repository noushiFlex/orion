"""Features. Importing this package registers every feature in FEATURES."""

from orion.features.base import FEATURES, Context, Feature, MatchLevel, TeamSeries
from orion.features import (clubelo, elo, form, goalrate, injuries, lineup, manager, market,  # noqa: F401
                            odds_shape, referee, rest, suspensions)  # (registration)

__all__ = ["FEATURES", "Context", "Feature", "MatchLevel", "TeamSeries"]
