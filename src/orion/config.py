"""Experiment config: one YAML file per model version (configs/<name>.yaml), validated on load."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from orion import leagues, seasons


@dataclass(frozen=True)
class Config:
    name: str
    seed: int
    leagues: tuple[str, ...]
    seasons: tuple[str, ...]
    warmup_seasons: int  # seasons that only feed rolling features / ratings, never training rows
    features: dict[str, dict]
    models: dict[str, dict]
    ensemble: dict[str, float]  # fixed weights, model name -> weight
    stack: tuple[str, ...]  # or: members whose weights are learned (see orion.training)
    test_seasons: tuple[str, ...]
    eval_leagues: tuple[str, ...] = ()  # leagues scored in the backtest (default: all); training uses every league
    path: Path | None = None

    @property
    def train_from(self) -> str:
        return self.seasons[self.warmup_seasons]

    @property
    def current_season(self) -> str:
        return self.seasons[-1]


def load(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    data = raw["data"]
    ensemble = dict(raw.get("ensemble") or {})
    stack = tuple(ensemble.pop("stack", ()))
    cfg = Config(
        name=raw["name"],
        seed=int(raw.get("seed", 0)),
        leagues=tuple(data["leagues"]),
        seasons=tuple(seasons.span(str(data["seasons"]["first"]), str(data["seasons"]["last"]))),
        warmup_seasons=int(data.get("warmup_seasons", 0)),
        features={k: (v or {}) for k, v in raw["features"].items()},
        models={k: (v or {}) for k, v in raw["models"].items()},
        ensemble={k: float(v) for k, v in ensemble.items()},
        stack=stack,
        test_seasons=tuple(str(s) for s in raw["backtest"]["test_seasons"]),
        eval_leagues=tuple(raw["backtest"].get("eval_leagues") or data["leagues"]),
        path=path,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    for code in cfg.leagues:
        leagues.get(code)
    if not set(cfg.eval_leagues) <= set(cfg.leagues):
        raise ValueError(f"eval_leagues not in data.leagues: {sorted(set(cfg.eval_leagues) - set(cfg.leagues))}")
    if cfg.warmup_seasons >= len(cfg.seasons):
        raise ValueError("warmup_seasons leaves no season to train on")
    unknown = set(cfg.test_seasons) - set(cfg.seasons)
    if unknown:
        raise ValueError(f"test seasons outside data.seasons: {sorted(unknown)}")
    if min(seasons.start_year(s) for s in cfg.test_seasons) <= seasons.start_year(cfg.train_from):
        raise ValueError("the first test season must come after the first training season")
    if cfg.ensemble and cfg.stack:
        raise ValueError("ensemble: use either fixed weights or `stack`, not both")
    missing = (set(cfg.ensemble) | set(cfg.stack)) - set(cfg.models)
    if missing:
        raise ValueError(f"ensemble members not listed in models: {sorted(missing)}")
    if cfg.stack and seasons.start_year(min(cfg.test_seasons)) - seasons.start_year(cfg.train_from) < 2:
        raise ValueError("stacking needs at least two training seasons before the first test season")
    if cfg.ensemble and sum(cfg.ensemble.values()) <= 0:
        raise ValueError("ensemble weights must sum to a positive number")
