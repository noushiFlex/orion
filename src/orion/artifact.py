"""Trained model artifact: models/<config name>/ holds the fitted models, config copy and metadata."""

import json
import shutil
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from orion import manifest, progress, seasons, tracking, training
from orion.config import Config
from orion.dataset import Dataset
from orion.models import Model
from orion.paths import MODELS as MODELS_DIR


@dataclass
class Artifact:
    models: dict[str, Model]
    ensemble: dict[str, float]  # final weights (fixed or learned)
    feature_cols: list[str]

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return training.predict(self.models, self.ensemble, df)


def train(cfg: Config, ds: Dataset) -> Artifact:
    """Fit every configured model on all played matches from the first post-warm-up season onward."""
    df = ds.played
    df = df[df["season"].map(seasons.start_year) >= seasons.start_year(cfg.train_from)]
    with progress.task("Train", total=training.steps(cfg)) as bar:
        models, weights = training.fit(cfg, ds.feature_cols, df, bar)
    art = Artifact(models, weights, ds.feature_cols)
    folder = MODELS_DIR / cfg.name
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, folder / "model.joblib")
    if cfg.path:
        shutil.copy(cfg.path, folder / "config.yaml")
    meta = {"config": cfg.name, "trained_at": manifest.now(), "dataset_fingerprint": ds.fingerprint(),
            "train_rows": len(df), "train_seasons": sorted(df["season"].unique().tolist()),
            "last_match_date": str(df["date"].max().date()), "feature_cols": ds.feature_cols,
            "ensemble_weights": weights, "environment": tracking.environment()}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return art


def load(cfg: Config) -> Artifact:
    return joblib.load(MODELS_DIR / cfg.name / "model.joblib")
