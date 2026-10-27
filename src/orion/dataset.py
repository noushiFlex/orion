"""Model-ready dataset: canonical matches + per-match xG + every configured feature, joined point-in-time.

Raw per-match columns (goals, shots, xG, odds of the match itself) stay in the frame for targets, baselines and
rolling features, but only `feature_cols` are ever given to a model.
"""

import hashlib
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from orion import leagues, manifest, mapping, pit, progress
from orion.config import Config
from orion.features import FEATURES, Context
from orion.paths import PROCESSED
from orion.sources.football_data import FootballData
from orion.sources.understat import Understat

log = logging.getLogger(__name__)

OUTCOMES = ("home", "draw", "away")  # outcome codes 0, 1, 2


@dataclass
class Dataset:
    frame: pd.DataFrame
    feature_cols: list[str]

    @property
    def played(self) -> pd.DataFrame:
        return self.frame[self.frame["outcome"].notna()]

    def fingerprint(self) -> str:
        return hashlib.sha256(pd.util.hash_pandas_object(self.frame, index=False).values.tobytes()).hexdigest()[:16]


def build(cfg: Config) -> Dataset:
    with progress.task("Load matches", total=2) as bar:
        bar.advance(0, label="football-data")
        matches = FootballData().load(cfg)
        bar.advance(label="understat xG")
        matches = attach_xg(cfg, matches)
        bar.advance()
    return build_features(Context(cfg, matches), cfg.features)


def build_features(ctx: Context, features: dict[str, dict]) -> Dataset:
    df = ctx.matches.copy()
    prefixes: list[str] = []
    with progress.task("Features", total=len(features)) as bar:
        for name, params in features.items():
            bar.advance(0, label=name)
            feature = FEATURES.get(name)(**params)
            df = feature.derive(pit.join(df, feature.compute(ctx), name))
            prefixes += [f"home_{name}_", f"away_{name}_", f"{name}_"]
            bar.advance()
    feature_cols = [c for c in df.columns if c.startswith(tuple(prefixes))]
    hg, ag = df["home_goals"], df["away_goals"]
    df["outcome"] = np.select([hg > ag, hg == ag, hg < ag], [0.0, 1.0, 2.0], default=np.nan)
    return Dataset(df.reset_index(drop=True), feature_cols)


def attach_xg(cfg: Config, matches: pd.DataFrame) -> pd.DataFrame:
    """Fill home_xg/away_xg from Understat for covered leagues (name-mapped, date within one day)."""
    if not any(leagues.get(d).understat for d in cfg.leagues):
        return matches
    xg = Understat().load(cfg)
    names = mapping.load_or_build("understat", xg, matches.dropna(subset=["home_goals"]))
    xg = mapping.apply(xg, names, "understat").rename(columns={"date": "xg_date"})
    pairs = matches[["match_id", "div", "date", "home", "away"]].merge(
        xg[["div", "xg_date", "home", "away", "home_xg", "away_xg"]], on=["div", "home", "away"])
    pairs["gap"] = (pairs["date"] - pairs["xg_date"]).abs()
    pairs = pairs[pairs["gap"] <= pd.Timedelta(days=1)].sort_values("gap").drop_duplicates("match_id")
    out = matches.drop(columns=["home_xg", "away_xg"]).merge(pairs[["match_id", "home_xg", "away_xg"]],
                                                             on="match_id", how="left")
    covered = out["div"].map(lambda d: leagues.get(d).understat is not None) & out["home_goals"].notna()
    log.info("xG attached to %.1f%% of played matches in covered leagues", 100 * out.loc[covered, "home_xg"].notna().mean())
    return out


def save(cfg: Config, ds: Dataset) -> dict:
    folder = PROCESSED / cfg.name
    folder.mkdir(parents=True, exist_ok=True)
    ds.frame.to_parquet(folder / "dataset.parquet", index=False)
    meta = {
        "config": cfg.name, "built_at": manifest.now(), "fingerprint": ds.fingerprint(), "rows": len(ds.frame),
        "played": int(ds.frame["outcome"].notna().sum()), "feature_cols": ds.feature_cols,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load(cfg: Config) -> Dataset:
    folder = PROCESSED / cfg.name
    meta = json.loads((folder / "meta.json").read_text())
    return Dataset(pd.read_parquet(folder / "dataset.parquet"), meta["feature_cols"])
