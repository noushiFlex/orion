"""Bookmaker-implied probabilities, margin removed by normalisation.

kind="pre": market-average odds collected before closing (Friday for weekend games, Tuesday midweek). Available
in history and in fixtures.csv, so usable for upcoming matches.
kind="close": Pinnacle closing odds, only known at kickoff (and absent from 2026/27 files).
Enabling this feature makes a model lean on the market; v1-v1.2 keep it off to measure their own signal.
"""

from orion.features.base import FEATURES, Context, Feature, MatchLevel


@FEATURES.register
class Market(Feature):
    name = "market"

    def __init__(self, kind: str = "pre"):
        if kind not in ("pre", "close"):
            raise ValueError(f"market kind must be 'pre' or 'close', got {kind!r}")
        self.kind = kind

    def compute(self, ctx: Context) -> MatchLevel:
        m = ctx.matches
        inv = 1 / m[[f"odds_{self.kind}_home", f"odds_{self.kind}_draw", f"odds_{self.kind}_away"]].to_numpy(dtype=float)
        total = inv.sum(axis=1)
        frame = m[["match_id"]].copy()
        for i, outcome in enumerate(("home", "draw", "away")):
            frame[f"p_{outcome}"] = inv[:, i] / total
        frame["overround"] = total - 1
        return MatchLevel(frame)
