"""Bookmaker odds as a model, margin removed. Rows without odds get training base rates.

kind="close" (default): Pinnacle closing odds, the bar to beat. Known only at kickoff, so never an ensemble member.
kind="pre": market-average odds before closing; a legitimate ensemble member.
"""

import numpy as np
import pandas as pd

from orion.models.base import MODELS, Model

ODDS = ["odds_close_home", "odds_close_draw", "odds_close_away"]


@MODELS.register
class Market(Model):
    name = "market"

    def __init__(self, feature_cols, seed=0, kind: str = "close"):
        super().__init__(feature_cols, seed)
        if kind not in ("pre", "close"):
            raise ValueError(f"market kind must be 'pre' or 'close', got {kind!r}")
        self.cols = [f"odds_{kind}_{o}" for o in ("home", "draw", "away")]
        self.uses_closing_odds = kind == "close"

    def fit(self, df: pd.DataFrame) -> "Market":
        counts = np.bincount(df["outcome"].astype(int), minlength=3)
        self.prior = counts / counts.sum()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        inv = 1 / df[self.cols].to_numpy(dtype=float)
        p = inv / inv.sum(axis=1, keepdims=True)
        p[np.isnan(p).any(axis=1)] = self.prior
        return p
