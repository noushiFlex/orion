"""Run tracking: every backtest/predict writes runs/<UTC time>-<config>-<kind>/ with the config, environment,
dataset fingerprint and outputs, so any result can be traced back to the exact code inputs and data."""

import json
import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from orion.config import Config
from orion.paths import RUNS

_PACKAGES = ("orion", "numpy", "pandas", "scipy", "scikit-learn")


def environment() -> dict:
    return {"python": platform.python_version(), **{p: version(p) for p in _PACKAGES}}


def new_run(cfg: Config, kind: str, fingerprint: str, extra: dict | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = RUNS / f"{stamp}-{cfg.name}-{kind}"
    folder.mkdir(parents=True)
    if cfg.path:
        shutil.copy(cfg.path, folder / "config.yaml")
    meta = {"kind": kind, "config": cfg.name, "seed": cfg.seed, "dataset_fingerprint": fingerprint,
            "environment": environment(), **(extra or {})}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return folder
