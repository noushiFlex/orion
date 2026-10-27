"""Market + correction: multinomial logit whose main inputs are the pre-closing market log-odds, plus all features.

The market log-odds are multiplied by `market_scale` after standardisation, which makes their L2 penalty
market_scale**2 times weaker: the model starts from the market and only learns small corrections from the other
features, instead of re-learning team strength from scratch. The correction strength (C) is chosen inside the
training data, never on test seasons.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from orion.models.base import MODELS, Model

PRE = ["odds_pre_home", "odds_pre_draw", "odds_pre_away"]


def market_logits(df: pd.DataFrame) -> np.ndarray:
    inv = 1 / df[PRE].to_numpy(dtype=float)
    p = inv / inv.sum(axis=1, keepdims=True)
    return np.column_stack([np.log(p[:, 0] / p[:, 1]), np.log(p[:, 2] / p[:, 1])])


@MODELS.register
class MarketResidual(Model):
    name = "market_residual"

    def __init__(self, feature_cols, seed=0, C: float | str = "auto", market_scale: float = 10.0,
                 C_grid: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2), holdout: float = 0.2):
        super().__init__(feature_cols, seed)
        self.C, self.C_grid, self.holdout, self.market_scale = C, C_grid, holdout, market_scale
        self.imp_m, self.sc_m = SimpleImputer(strategy="median"), StandardScaler()
        self.imp_f = SimpleImputer(strategy="median", keep_empty_features=True)
        self.sc_f = StandardScaler()

    def _matrix(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        m, f = market_logits(df), df[self.feature_cols]
        if fit:
            m = self.sc_m.fit_transform(self.imp_m.fit_transform(m))
            f = self.sc_f.fit_transform(self.imp_f.fit_transform(f))
        else:
            m = self.sc_m.transform(self.imp_m.transform(m))
            f = self.sc_f.transform(self.imp_f.transform(f))
        return np.hstack([m * self.market_scale, f])

    def fit(self, df: pd.DataFrame) -> "MarketResidual":
        """C="auto": pick C on the most recent `holdout` share of the training rows (chronological), then refit."""
        df = df.dropna(subset=PRE).sort_values("date", kind="stable")  # rows without pre-closing odds teach nothing
        X, y = self._matrix(df, fit=True), df["outcome"].astype(int).to_numpy()
        C = self.C
        if C == "auto":
            cut = int(len(df) * (1 - self.holdout))
            losses = {c: log_loss(y[cut:], LogisticRegression(C=c, max_iter=3000).fit(X[:cut], y[:cut])
                                  .predict_proba(X[cut:]), labels=[0, 1, 2]) for c in self.C_grid}
            C = min(losses, key=losses.get)
        self.C_ = C
        self.clf = LogisticRegression(C=C, max_iter=3000).fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = np.zeros((len(df), 3))
        p[:, self.clf.classes_] = self.clf.predict_proba(self._matrix(df, fit=False))
        return p
