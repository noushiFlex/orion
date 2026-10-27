"""Multinomial logistic regression on standardised features: a strong, hard-to-overfit baseline for 1/X/2."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from orion.models.base import MODELS, Model


@MODELS.register
class Logit(Model):
    name = "logit"

    def __init__(self, feature_cols, seed=0, C: float = 0.05):
        super().__init__(feature_cols, seed)
        self.pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                  LogisticRegression(C=C, max_iter=2000))

    def fit(self, df: pd.DataFrame) -> "Logit":
        self.pipe.fit(df[self.feature_cols], df["outcome"].astype(int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = np.zeros((len(df), 3))
        p[:, self.pipe.classes_] = self.pipe.predict_proba(df[self.feature_cols])
        return p
