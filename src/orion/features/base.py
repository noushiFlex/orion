"""Feature contract and point-in-time types.

A feature returns either
  - TeamSeries: values about a team that become known at `available_at` (e.g. a rating after a match), or
  - MatchLevel: values known before kickoff of one specific match (e.g. closing odds).
orion.pit joins them to matches strictly before kickoff day, so a feature cannot see its own match or the future.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import pandas as pd

from orion.config import Config
from orion.registry import Registry


@dataclass(frozen=True)
class TeamSeries:
    frame: pd.DataFrame  # team, available_at, <value columns>
    max_age: pd.Timedelta | None = None  # values older than this at kickoff are treated as missing


@dataclass(frozen=True)
class MatchLevel:
    frame: pd.DataFrame  # match_id, <value columns>


@dataclass
class Context:
    """What features may read. `matches` holds played and scheduled matches (goals null when scheduled)."""
    cfg: Config | None
    matches: pd.DataFrame

    @cached_property
    def played(self) -> pd.DataFrame:
        return self.matches.dropna(subset=["home_goals", "away_goals"]).sort_values(["date", "match_id"])


class Feature(ABC):
    name: ClassVar[str]

    @abstractmethod
    def compute(self, ctx: Context) -> TeamSeries | MatchLevel:
        ...

    def derive(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional post-join step (differences, helper-column cleanup). Columns must keep the feature's prefix."""
        return df


FEATURES: Registry[Feature] = Registry("feature")


def team_long(played: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, played match) with for/against stats, sorted by team then date."""
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({"team": played[f"{side}_id"].values, "date": played["date"].values,
                          "match_id": played["match_id"].values})
        for stat in ("goals", "shots", "sot", "corners", "xg"):
            d[f"{stat}_for"] = played[f"{side}_{stat}"].values
            d[f"{stat}_against"] = played[f"{other}_{stat}"].values
        diff = d["goals_for"] - d["goals_against"]
        d["points"] = (diff > 0) * 3 + (diff == 0) * 1
        parts.append(d)
    return pd.concat(parts, ignore_index=True).sort_values(["team", "date", "match_id"], ignore_index=True)
