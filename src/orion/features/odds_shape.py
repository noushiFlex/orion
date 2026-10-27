"""Shape of the pre-closing betting market: how much bookmakers disagree, per outcome.

    spread_<o> = log(best price / average price)   wide = disagreement, often news not yet priced in everywhere
    b365_<o>   = log(Bet365 price / average price) a large recreational book against the market
All inputs are collected before closing (same time as the average odds), so they are known before kickoff.
"""

import numpy as np

from orion.features.base import FEATURES, Context, Feature, MatchLevel

OUTCOMES = ("home", "draw", "away")


@FEATURES.register
class OddsShape(Feature):
    name = "odds_shape"

    def compute(self, ctx: Context) -> MatchLevel:
        m = ctx.matches
        frame = m[["match_id"]].copy()
        for o in OUTCOMES:
            avg = m[f"odds_pre_{o}"].to_numpy(dtype=float)
            frame[f"spread_{o}"] = np.log(m[f"odds_max_{o}"].to_numpy(dtype=float) / avg)
            frame[f"b365_{o}"] = np.log(m[f"odds_b365_{o}"].to_numpy(dtype=float) / avg)
        return MatchLevel(frame)
