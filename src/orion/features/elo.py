"""Elo computed from the results in the dataset (one shared pool; teams outside the configured leagues are unseen)."""

import math

import pandas as pd

from orion.features.base import FEATURES, Context, Feature, TeamSeries


@FEATURES.register
class Elo(Feature):
    name = "elo"

    def __init__(self, k: float = 20, home_advantage: float = 65, start: float = 1500,
                 promoted_start: float = 1400, goal_diff: bool = True):
        self.k, self.hfa, self.start, self.promoted_start, self.goal_diff = k, home_advantage, start, promoted_start, goal_diff

    def compute(self, ctx: Context) -> TeamSeries:
        played = ctx.played
        first_season = played["season"].iloc[0] if len(played) else None
        ratings: dict[str, float] = {}
        rows = []
        for season, date, h, a, hg, ag in zip(played["season"], played["date"], played["home_id"], played["away_id"],
                                              played["home_goals"], played["away_goals"]):
            # Teams first seen after the first season are newcomers (usually promoted): start them lower.
            init = self.start if season == first_season else self.promoted_start
            rh, ra = ratings.get(h, init), ratings.get(a, init)
            expected = 1 / (1 + 10 ** ((ra - rh - self.hfa) / 400))
            result = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            mult = math.log(abs(hg - ag) + 1) + 1 if self.goal_diff else 1.0
            delta = self.k * mult * (result - expected)
            ratings[h], ratings[a] = rh + delta, ra - delta
            rows += [(h, date, rh + delta), (a, date, ra - delta)]
        return TeamSeries(pd.DataFrame(rows, columns=["team", "available_at", "rating"]))

    def derive(self, df: pd.DataFrame) -> pd.DataFrame:
        df["elo_diff"] = df["home_elo_rating"] - df["away_elo_rating"]
        return df
