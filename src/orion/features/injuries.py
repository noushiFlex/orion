"""Injured regulars on match day (salimt Transfermarkt injury history, up to 2026-09-29).

Per side: n_regulars_out, out_m (their market value, EUR m), out_share (share of the regulars' value missing).
An injury counts if it started at least 2 days before the match (so it was public) and had not ended.
"""

import numpy as np

from orion.features.base import FEATURES, Context, Feature, MatchLevel
from orion.features.tm import linked_games, per_side
from orion.sources.transfermarkt import Transfermarkt


@FEATURES.register
class Injuries(Feature):
    name = "injuries"

    def __init__(self, window: int = 10, min_starts: int = 5):
        self.window, self.min_starts = window, min_starts

    def compute(self, ctx: Context) -> MatchLevel:
        inj = Transfermarkt().injured_regulars(ctx.cfg, self.window, self.min_starts)
        frame = per_side(linked_games(ctx), inj, ["n_out", "out_m", "regulars_m"])
        for side in ("home", "away"):
            frame[f"{side}_out_share"] = frame[f"{side}_out_m"] / frame[f"{side}_regulars_m"].replace(0, np.nan)
            frame = frame.drop(columns=f"{side}_regulars_m")
        frame["out_share_diff"] = frame["home_out_share"] - frame["away_out_share"]
        return MatchLevel(frame)
