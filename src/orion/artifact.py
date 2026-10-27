"""Trained model artifact: models/<config name>/ holds the fitted models, config copy and metadata."""

import json
import shutil
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from orion import manifest, progress, seasons, tracking
from orion.config import Config
from orion.dataset import Dataset
from orion.models import MODELS, Model, combine
from orion.paths import MODELS as MODELS_DIR


@dataclass
class Artifact:
    models: dict[str, Model]
    ensemble: dict[str, float]
    feature_cols: list[str]

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        probs = {name: m.predict_proba(df) for name, m in self.models.items()}
        if self.ensemble:
            probs["ensemble"] = combine(probs, self.ensemble)
        return probs


def train(cfg: Config, ds: Dataset) -> Artifact:
    """Fit every configured model on all played matches from the first post-warm-up season onward."""
    df = ds.played
    df = df[df["season"].map(seasons.start_year) >= seasons.start_year(cfg.train_from)]
    models = {}
    with progress.task("Train", total=len(cfg.models)) as bar:
        for name, params in cfg.models.items():
            bar.advance(0, label=f"{name} · {len(df):,} matches")
            models[name] = MODELS.get(name)(ds.feature_cols, seed=cfg.seed, **params).fit(df)
            bar.advance()
    art = Artifact(models, cfg.ensemble, ds.feature_cols)
    folder = MODELS_DIR / cfg.name
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, folder / "model.joblib")
    if cfg.path:
        shutil.copy(cfg.path, folder / "config.yaml")
    meta = {"config": cfg.name, "trained_at": manifest.now(), "dataset_fingerprint": ds.fingerprint(),
            "train_rows": len(df), "train_seasons": sorted(df["season"].unique().tolist()),
            "last_match_date": str(df["date"].max().date()), "feature_cols": ds.feature_cols,
            "environment": tracking.environment()}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return art


def load(cfg: Config) -> Artifact:
    return joblib.load(MODELS_DIR / cfg.name / "model.joblib")
