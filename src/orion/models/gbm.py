"""Gradient-boosted trees on the outcome directly. Handles missing features natively."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from orion.models.base import MODELS, Model


@MODELS.register
class GBM(Model):
    name = "gbm"

    def __init__(self, feature_cols, seed=0, **params):
        super().__init__(feature_cols, seed)
        # Fixed iteration count: sklearn's early stopping would hold out a random, not chronological, split.
        self.clf = HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **params)

    def fit(self, df: pd.DataFrame) -> "GBM":
        self.clf.fit(df[self.feature_cols], df["outcome"].astype(int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = np.zeros((len(df), 3))
        p[:, self.clf.classes_] = self.clf.predict_proba(df[self.feature_cols])
        return p
