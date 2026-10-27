"""Command line: orion [-c configs/orion-v1.yaml] <command>.

  fetch [SOURCE ...] [--refresh]   download raw data (default: every registered source)
  manifest                         record pre-existing raw files in data/manifest.jsonl
  map [--rebuild]                  learn team-name mappings and report coverage
  build                            build the point-in-time dataset into data/processed/<config>/
  backtest [--rebuild]             walk-forward backtest, results in runs/
  train [--rebuild]                fit on all data, artifact in models/<config>/
  predict [--days N]               probabilities for scheduled matches in the next N days
"""

import argparse
import json
import logging
import time

import pandas as pd

from orion import artifact, config, dataset, manifest, mapping, tracking, ui
from orion.evaluation import backtest
from orion.paths import CONFIGS, MODELS as MODELS_DIR
from orion.sources import SOURCES
from orion.sources.clubelo import ClubElo
from orion.sources.football_data import FootballData
from orion.sources.transfermarkt import Transfermarkt
from orion.sources.understat import Understat

log = logging.getLogger("orion")


def _dataset(cfg, rebuild: bool) -> dataset.Dataset:
    if not rebuild:
        try:
            return dataset.load(cfg)
        except FileNotFoundError:
            pass
    ds = dataset.build(cfg)
    dataset.save(cfg, ds)
    return ds


def cmd_fetch(cfg, args):
    with ui.live():
        for name in args.sources or SOURCES.names():
            SOURCES.get(name)().fetch(cfg, refresh=args.refresh)


def cmd_manifest(cfg, args):
    log.info("backfilled %d raw files", manifest.backfill())


def cmd_map(cfg, args):
    played = FootballData().load(cfg).dropna(subset=["home_goals"])
    for name, src in (("understat", Understat().load(cfg)), ("clubelo", ClubElo().load_results(cfg)),
                      ("transfermarkt", Transfermarkt().load_results(cfg))):
        m = mapping.load_or_build(name, src, played, rebuild=args.rebuild)
        rejected = m[~m["accepted"]]
        log.info("%s: %d names accepted, %d rejected%s", name, int(m["accepted"].sum()), len(rejected),
                 f": {rejected[['div', 'source_name', 'votes', 'share']].to_dict('records')}" if len(rejected) else "")


def cmd_build(cfg, args):
    with ui.live():
        meta = dataset.save(cfg, dataset.build(cfg))
    log.info("dataset %s: %d rows (%d played), %d features, fingerprint %s",
             cfg.name, meta["rows"], meta["played"], len(meta["feature_cols"]), meta["fingerprint"])


def cmd_backtest(cfg, args):
    start = time.perf_counter()
    with ui.live():
        ds = _dataset(cfg, args.rebuild)
        predictions, scores, weights = backtest.run(cfg, ds)
    run = tracking.new_run(cfg, "backtest", ds.fingerprint())
    predictions.to_parquet(run / "predictions.parquet", index=False)
    scores.to_csv(run / "scores.csv", index=False)
    (run / "ensemble_weights.json").write_text(json.dumps(weights, indent=2))
    ui.backtest_dashboard(predictions, scores, weights, len(ds.feature_cols), time.perf_counter() - start, run)


def cmd_train(cfg, args):
    start = time.perf_counter()
    with ui.live():
        artifact.train(cfg, _dataset(cfg, args.rebuild))
    folder = MODELS_DIR / cfg.name
    ui.train_dashboard(json.loads((folder / "meta.json").read_text()), list(cfg.models), time.perf_counter() - start, folder)


def cmd_predict(cfg, args):
    start = time.perf_counter()
    with ui.live():
        ds = _dataset(cfg, rebuild=True)  # always rebuild: scheduled fixtures change daily
    art = artifact.load(cfg)
    today = pd.Timestamp.today().normalize()
    todo = ds.frame[ds.frame["outcome"].isna() & ds.frame["date"].between(today, today + pd.Timedelta(days=args.days))]
    if todo.empty:
        log.info("no scheduled matches in the next %d days (run `orion fetch football_data` to refresh fixtures)", args.days)
        return
    probs = art.predict(todo)
    best = "ensemble" if "ensemble" in probs else next(iter(probs))
    out = todo[["date", "div", "home", "away"]].copy()
    for name, p in probs.items():
        out[[f"{name}_home", f"{name}_draw", f"{name}_away"]] = p
    run = tracking.new_run(cfg, "predict", ds.fingerprint(), {"matches": len(out)})
    out.to_csv(run / "predictions.csv", index=False)
    ui.predict_dashboard(out, best, time.perf_counter() - start, run)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="orion", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default=str(CONFIGS / "orion-v1.yaml"))
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("sources", nargs="*", help=f"any of {SOURCES.names()}")
    p.add_argument("--refresh", action="store_true")
    sub.add_parser("manifest")
    sub.add_parser("map").add_argument("--rebuild", action="store_true")
    sub.add_parser("build")
    for name in ("backtest", "train"):
        sub.add_parser(name).add_argument("--rebuild", action="store_true", help="rebuild the dataset first")
    sub.add_parser("predict").add_argument("--days", type=int, default=7)
    args = parser.parse_args(argv)

    ui.setup_logging(args.verbose)
    cfg = config.load(args.config)
    ui.header(cfg.name, args.command)
    globals()[f"cmd_{args.command}"](cfg, args)


if __name__ == "__main__":
    main()
