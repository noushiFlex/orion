"""Fitting shared by the backtest and the final training, so both follow exactly the same procedure.

Ensemble weights are either fixed in the config (`ensemble: {poisson: 0.5, gbm: 0.5}`) or learned
(`ensemble: {stack: [market_pre, poisson, ...]}`): members are fitted on every training season but the last,
weights are chosen to minimise log-loss on that last season, then members are refitted on all training seasons.
"""

import logging

import numpy as np
import pandas as pd

from orion import progress, seasons
from orion.config import Config
from orion.models import Model, build, combine, fit_weights

log = logging.getLogger(__name__)


def steps(cfg: Config) -> int:
    """Number of model fits per training call (for progress bars)."""
    return len(cfg.models) + len(cfg.stack)


def fit(cfg: Config, feature_cols: list[str], train: pd.DataFrame, bar: progress.Task | None = None,
        label: str = "") -> tuple[dict[str, Model], dict[str, float]]:
    bar = bar or progress.Task()
    weights = _ensemble_weights(cfg, feature_cols, train, bar, label)
    models = {}
    for name, params in cfg.models.items():
        bar.advance(0, label=f"{label}{name} · {len(train):,} matches")
        models[name] = build(name, params, feature_cols, cfg.seed).fit(train)
        bar.advance()
    _check_members(models, weights)
    return models, weights


def predict(models: dict[str, Model], weights: dict[str, float], df: pd.DataFrame) -> dict[str, np.ndarray]:
    probs = {name: m.predict_proba(df) for name, m in models.items()}
    if weights:
        probs["ensemble"] = combine(probs, weights)
    return probs


def _ensemble_weights(cfg, feature_cols, train, bar, label) -> dict[str, float]:
    if not cfg.stack:
        return cfg.ensemble
    year = train["season"].map(seasons.start_year)
    inner, holdout = train[year < year.max()], train[year == year.max()]
    probs = {}
    for name in cfg.stack:
        bar.advance(0, label=f"{label}stacking {name} · holdout {holdout['season'].iloc[0]}")
        probs[name] = build(name, cfg.models[name], feature_cols, cfg.seed).fit(inner).predict_proba(holdout)
        bar.advance()
    weights = fit_weights(probs, holdout["outcome"].astype(int).to_numpy())
    log.debug("stack weights learned on %s: %s", holdout["season"].iloc[0], weights)
    return weights


def _check_members(models: dict[str, Model], weights: dict[str, float]) -> None:
    leaking = [n for n in weights if models[n].uses_closing_odds]
    if leaking:
        raise ValueError(f"ensemble members use closing odds (only known at kickoff): {leaking}")
