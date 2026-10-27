"""Goals model: one Poisson regression per side, Dixon-Coles low-score correction, outcome probabilities from the
score matrix. Also exposes the full score distribution (predict_scores) for exact-score questions."""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from orion.models.base import MODELS, Model


@MODELS.register
class Poisson(Model):
    name = "poisson"

    def __init__(self, feature_cols, seed=0, alpha: float = 1.0, max_goals: int = 10,
                 rho_grid: tuple[float, float, int] = (-0.2, 0.1, 31)):
        super().__init__(feature_cols, seed)
        self.alpha, self.max_goals, self.rho_grid = alpha, max_goals, rho_grid
        self.rho = 0.0

    def _pipe(self):
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             PoissonRegressor(alpha=self.alpha, max_iter=1000))

    def fit(self, df: pd.DataFrame) -> "Poisson":
        X = df[self.feature_cols]
        self.home_, self.away_ = self._pipe().fit(X, df["home_goals"]), self._pipe().fit(X, df["away_goals"])
        mu_h, mu_a = self._rates(df)
        hg = df["home_goals"].clip(upper=self.max_goals).astype(int).to_numpy()
        ag = df["away_goals"].clip(upper=self.max_goals).astype(int).to_numpy()
        rows = np.arange(len(df))
        best = -np.inf
        for rho in np.linspace(*self.rho_grid):
            ll = np.log(np.clip(_score_matrix(mu_h, mu_a, rho, self.max_goals)[rows, hg, ag], 1e-12, None)).sum()
            if ll > best:
                best, self.rho = ll, float(rho)
        return self

    def _rates(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[self.feature_cols]
        return self.home_.predict(X), self.away_.predict(X)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """(n, G+1, G+1) probabilities, [i, h, a] = P(home scores h, away scores a)."""
        return _score_matrix(*self._rates(df), self.rho, self.max_goals)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        m = self.predict_scores(df)
        n = m.shape[1]
        return np.stack([(m * np.tril(np.ones((n, n)), -1)).sum((1, 2)),
                         np.trace(m, axis1=1, axis2=2),
                         (m * np.triu(np.ones((n, n)), 1)).sum((1, 2))], axis=1)


def _score_matrix(mu_h: np.ndarray, mu_a: np.ndarray, rho: float, max_goals: int) -> np.ndarray:
    k = np.arange(max_goals + 1)
    m = poisson.pmf(k[None, :], mu_h[:, None])[:, :, None] * poisson.pmf(k[None, :], mu_a[:, None])[:, None, :]
    m[:, 0, 0] *= 1 - mu_h * mu_a * rho
    m[:, 0, 1] *= 1 + mu_h * rho
    m[:, 1, 0] *= 1 + mu_a * rho
    m[:, 1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum(axis=(1, 2), keepdims=True)
