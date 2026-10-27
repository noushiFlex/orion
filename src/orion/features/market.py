"""Bookmaker-implied probabilities from Pinnacle closing odds, margin removed by normalisation.

Closing odds are only known at kickoff. Enabling this feature makes a model lean on the market; keep it off to
measure a model's own signal against the `market` baseline.
"""

from orion.features.base import FEATURES, Context, Feature, MatchLevel


@FEATURES.register
class Market(Feature):
    name = "market"

    def compute(self, ctx: Context) -> MatchLevel:
        m = ctx.matches
        inv = 1 / m[["odds_close_home", "odds_close_draw", "odds_close_away"]].to_numpy()
        total = inv.sum(axis=1)
        frame = m[["match_id"]].copy()
        for i, outcome in enumerate(("home", "draw", "away")):
            frame[f"p_{outcome}"] = inv[:, i] / total
        frame["overround"] = total - 1
        return MatchLevel(frame)
