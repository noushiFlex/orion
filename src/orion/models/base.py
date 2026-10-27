"""Model contract: fit on a dataset frame, return (n, 3) probabilities ordered home, draw, away."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from orion.registry import Registry


class Model(ABC):
    name: ClassVar[str]
    uses_closing_odds: bool = False  # True if predictions need odds only known at kickoff (never an ensemble member)

    def __init__(self, feature_cols: list[str], seed: int = 0):
        self.feature_cols = list(feature_cols)
        self.seed = seed

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "Model":
        """df holds feature_cols plus targets: outcome (0/1/2), home_goals, away_goals."""

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        ...


MODELS: Registry[Model] = Registry("model")


def build(name: str, params: dict, feature_cols: list[str], seed: int) -> Model:
    """Config entries are `<instance name>: {model: <registry name>, ...params}`; `model` defaults to the instance name."""
    params = dict(params)
    return MODELS.get(params.pop("model", name))(feature_cols, seed=seed, **params)


def combine(probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of member probabilities, renormalised per row."""
    total = sum(weights.values())
    p = sum(probs[name] * (w / total) for name, w in weights.items())
    return p / p.sum(axis=1, keepdims=True)


def fit_weights(probs: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    """Non-negative weights summing to 1 that minimise the log-loss of `combine` on held-out predictions."""
    names = list(probs)
    stacked = np.stack([probs[n][np.arange(len(y)), y] for n in names], axis=1)  # P(observed outcome) per member

    def loss(z):
        w = np.exp(z - z.max())
        w /= w.sum()
        return -np.mean(np.log(np.clip(stacked @ w, 1e-15, None)))

    z = minimize(loss, np.zeros(len(names)), method="L-BFGS-B").x
    w = np.exp(z - z.max())
    return dict(zip(names, (w / w.sum()).round(4).tolist()))
