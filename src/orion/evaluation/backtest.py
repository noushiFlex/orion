"""Walk-forward backtest: for each test season, train on every earlier post-warm-up season, predict that season.

Scores are reported on all test matches and on the subset with closing odds, where every model is compared with
the market on exactly the same matches.
"""

import logging

import numpy as np
import pandas as pd

from orion import progress, seasons
from orion.config import Config
from orion.dataset import Dataset
from orion.evaluation.metrics import evaluate
from orion.models import MODELS, combine
from orion.models.market import ODDS

log = logging.getLogger(__name__)


def run(cfg: Config, ds: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = ds.played
    year = df["season"].map(seasons.start_year)
    first = seasons.start_year(cfg.train_from)
    preds = []
    with progress.task("Backtest", total=len(cfg.test_seasons) * len(cfg.models)) as bar:
        for season in cfg.test_seasons:
            test_year = seasons.start_year(season)
            train, test = df[(year >= first) & (year < test_year)], df[year == test_year]
            probs = {}
            for name, params in cfg.models.items():
                bar.advance(0, label=f"{season} · {name} · {len(train):,} train / {len(test):,} test")
                probs[name] = MODELS.get(name)(ds.feature_cols, seed=cfg.seed, **params).fit(train).predict_proba(test)
                bar.advance()
            if cfg.ensemble:
                probs["ensemble"] = combine(probs, cfg.ensemble)
            preds += [_frame(season, test, name, p) for name, p in probs.items()]
    predictions = pd.concat(preds, ignore_index=True)
    return predictions, score(predictions)


def _frame(season: str, test: pd.DataFrame, model: str, p: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": test["match_id"].to_numpy(), "season": season, "div": test["div"].to_numpy(), "model": model,
        "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2],
        "outcome": test["outcome"].astype(int).to_numpy(), "has_odds": test[ODDS].notna().all(axis=1).to_numpy(),
    })


def score(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per (subset, model, season), plus season='all' per model."""
    rows = []
    for subset, frame in (("all", predictions), ("with_odds", predictions[predictions["has_odds"]])):
        groups = [((m, s), g) for (m, s), g in frame.groupby(["model", "season"])]
        groups += [((m, "all"), g) for m, g in frame.groupby("model")]
        for (model, season), g in groups:
            p = g[["p_home", "p_draw", "p_away"]].to_numpy()
            rows.append({"subset": subset, "model": model, "season": season, **evaluate(g["outcome"].to_numpy(), p)})
    return pd.DataFrame(rows)
