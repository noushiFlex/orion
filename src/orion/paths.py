"""Filesystem layout. Every path in the project derives from ROOT (override with ORION_ROOT)."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("ORION_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"  # downloads, never modified
INTERIM = DATA / "interim"  # derived lookup tables (team-name mappings)
PROCESSED = DATA / "processed"  # model-ready datasets, one folder per config
MANIFEST = DATA / "manifest.jsonl"  # provenance of every raw file
CONFIGS = ROOT / "configs"
RUNS = ROOT / "runs"  # one folder per backtest / predict run
MODELS = ROOT / "models"  # trained artifacts, one folder per config name
