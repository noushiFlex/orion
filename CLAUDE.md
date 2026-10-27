# foot-predict / Orion

Personal football prediction project (private use only). Orion outputs 1/X/2 probabilities and score distributions.
Goal: well-calibrated probabilities, measured against the Pinnacle closing line. Not guaranteed wins.

## Layout

```
configs/<name>.yaml     one file per model version (leagues, seasons, features, models, backtest)
src/orion/
  sources/              one module per data source: fetch() raw files, load() validated frame
  features/             one module per feature, returns TeamSeries or MatchLevel
  models/               one module per model, fit(df) / predict_proba(df) -> (n, 3) home/draw/away
  evaluation/           metrics (log-loss, Brier, RPS) and walk-forward backtest
  pit.py                the only place features meet matches (point-in-time join)
  mapping.py            learned team-name maps between sources (vote on date + score)
  schema.py             column contracts, sources validate on load
  manifest.py, http.py  provenance (data/manifest.jsonl) and verified downloads
  dataset.py, artifact.py, tracking.py, cli.py
data/raw/<source>/      downloads, never modified by code
data/interim/mappings/  learned team-name maps
data/processed/<cfg>/   dataset.parquet + meta.json
models/<cfg>/           trained artifact + config copy + meta
runs/<time>-<cfg>-<kind>/  every backtest / predict run: config, environment, fingerprint, outputs
tests/                  leakage, point-in-time, mapping, schema, metrics, models
```

## Commands

Run from the project root with the venv: `./.venv/bin/orion <cmd>`. Default config is `configs/orion-v1.yaml`;
pass `-c configs/<name>.yaml` (before the command) for another version.

| Command | What it does |
|---|---|
| `fetch [source...] [--refresh]` | Download raw data (all sources by default). Past seasons are skipped; the current season and fixtures are re-fetched. |
| `manifest` | Record raw files that are not yet in `data/manifest.jsonl`. |
| `map [--rebuild]` | Learn team-name mappings and print accepted/rejected names. |
| `build` | Build `data/processed/<cfg>/dataset.parquet`. |
| `backtest [--rebuild]` | Walk-forward backtest, scores printed and saved in `runs/`. `--rebuild` rebuilds the dataset first. |
| `train [--rebuild]` | Fit all models on every post-warm-up season, save to `models/<cfg>/`. |
| `predict [--days N]` | Probabilities for scheduled matches in the next N days (default 7). Always rebuilds the dataset. |

Full refresh, in this order:

```bash
./.venv/bin/orion fetch
./.venv/bin/orion backtest --rebuild
./.venv/bin/orion train
./.venv/bin/orion predict --days 7
```

Every command shows live progress bars with elapsed time and ETA, then a dashboard (`src/orion/ui.py`, rich +
plotext): backtest prints scores vs market, a log-loss curve per season, a per-league table and a verdict; train
and predict print summary panels. Core modules report progress through `orion.progress` and never import the UI.
plotext is pinned below 6 (6.x rewrote its API).

Tests: `./.venv/bin/pytest -q` (31 tests: leakage, point-in-time, mapping, schema, metrics, models, Transfermarkt
SQL on a synthetic DuckDB). CI: `.github/workflows/tests.yml` runs them on Python 3.12 and 3.13.

`README.md` is the public face of the repo; its figures come from `docs/make_figures.py` (`pip install -e ".[docs]"`),
which reads the latest backtest runs. Regenerate them and update the README tables after a new version.
Dependencies: `pyproject.toml`, exact versions in `requirements.lock`. Install: `./.venv/bin/pip install -e ".[dev]"`.

## Metrics

- Log-loss: main metric. Punishes confident wrong predictions. Lower is better; always guessing 1/3 gives about 1.10.
- RPS (ranked probability score): like log-loss but respects the order home < draw < away. Lower is better.
- Accuracy: share of matches where the most likely outcome happened. Ignores confidence, so secondary only.
- Compare models on the `with_odds` subset, where every model and `market` are scored on the same matches.

## Status (2026-10-27)

Orion v1 (`configs/orion-v1.yaml`): Big Five, 2015/16 to 2026/27, features form (5 and 10 matches), own Elo,
ClubElo, rest days; xG enters through form. Models: market baseline, Poisson (Dixon-Coles), GBM, ensemble
Poisson+GBM 50/50. Final model trained on 18,331 matches, 2016/17 to 2026-10-26 (2015/16 is warm-up only).

Backtest, test seasons 2022/23 to 2025/26, `with_odds` subset:

| Model | Log-loss | RPS | Accuracy |
|---|---|---|---|
| market (bar to beat) | 0.963 | 0.193 | 54.5% |
| poisson | 0.980 | 0.198 | 53.5% |
| ensemble | 0.982 | 0.199 | 53.5% |
| gbm | 0.995 | 0.201 | 52.6% |

v1 does not beat the market. Do not use it for betting.

Versions (each one a config in `configs/`, same test seasons and `with_odds` subset, best model's log-loss):

| Version | Change | Log-loss | vs closing |
|---|---|---|---|
| v1 | form, Elo, ClubElo, rest; poisson + gbm | 0.9801 | +0.0171 |
| v1.1 | + goalrate (online attack/defence from goals and xG) | 0.9780 | +0.0150 |
| v1.2 | + logit, regularised gbm, ensemble poisson/logit/gbm | 0.9769 | +0.0138 |
| v1.3 | + pre-closing market-average odds as a feature | 0.9687 | +0.0057 |
| v1.4 | stacking with `market_pre` as a member, learned weights | 0.9653 | +0.0023 |
| v1.5 | + odds_shape (bookmaker disagreement), market_residual (market + correction, C chosen in-sample) | 0.9647 | +0.0017 |
| v1.6 | trained on all 22 leagues, scored on the Big Five (`eval_leagues`) | 0.9644 | +0.0014 |
| v2 | + lineup (starting-XI Transfermarkt value vs usual XI); predicts at line-up time | 0.9639 | +0.0009 |
| **v2.1** | **+ injuries (injured regulars on match day, salimt history to 2026-09-29)** | **0.9635** | **+0.0004** |
| v2.2 | + suspensions (bans after the previous league game) | 0.9636 | +0.0006 |
| v2.3 | + manager tenure | 0.9638 | +0.0008 |
| v2.4 | + discipline form (fouls, cards) and referee tendencies | 0.9636 | +0.0005 |
| v2.5 | v2.4 + 16 football-data "new" leagues as training rows | 0.9664 | +0.0034 |

Best version: v2.1 (99.7% of the closing market's skill). Paired tests vs pre-closing odds: v2.1 -0.0020 ±
0.0016 (significant); vs closing +0.0004 ± 0.0021 and vs v2 -0.0004 ± 0.0008 (not significant). v2.2-v2.4 add
nothing measurable. v2.5 is worse: stacking weights are learned on a holdout season that mixes in leagues without
pre-closing odds, and logit/gbm learn from leagues unlike the Big Five. Keep extra leagues out of Big-Five configs.

v2 paired tests (95%): vs pre-closing market -0.0016 ± 0.0015 (significant, first one), vs closing +0.0009 ±
0.0020 and vs v1.6 -0.0005 ± 0.0006 (not significant). v2 is backtest-only: Transfermarkt stops at 2026-07-06, so
for upcoming matches the lineup columns are empty and v2 behaves like v1.6. Live use needs a current line-up source.

Out-of-sample check on 2026/27 (124 matches, after all training data), against The Open Model's pre-kickoff
ledger: pre-closing market 0.956, Orion v1.6 0.962, The Open Model 0.986 (Orion - Open Model -0.024 ± 0.028).

Reference points: pre-closing market average alone 0.9655, Pinnacle closing 0.9630. Paired tests (95%) on v1.6:
ensemble vs pre-closing market -0.0011 ± 0.0013, vs closing +0.0014 ± 0.0020, both not significant. Orion is at
market level, with no proven edge. A real gain needs information the market prices late or poorly (lineups,
absences). Never tune hyperparameters on test seasons: choose them inside the training window (see
`market_residual`, C="auto").

Comparison with The Open Model's public record (https://theopenmodel.com/record/, Big Five backtest 2023-2025,
3,362 matches): accuracy 51.0% vs their market 54.0%, calibration gap 0.68%. Orion v2 on 6,228 matches: accuracy
54.3% vs market 54.5%, calibration gap 0.57% (market 1.01%), Brier 0.573. Overall rating: v2 15/20 (market level,
beats pre-closing odds, no proven edge on closing); live-usable v1.6 14/20.

## Known limitations

- Odds for a match appear in `fixtures.csv` only a few days before kickoff (usually Tuesday/Friday). Before
  that, configs with odds features (v1.3+) have nothing to work with: use `configs/orion-v1.2.yaml` (no odds).
  The `market` model then returns base rates, not market prices; ignore it.
- A match missing from `fixtures.csv` cannot go through `orion predict` yet. Workaround: append a MATCHES row
  (no goals) to the loaded matches, then `dataset.build_features` + `training.fit` / `training.predict`.
- ClubElo mirror snapshots stop at 2026-09-01; with `max_age_days: 45` the feature is empty for later matches.
- Transfermarkt stops at 2026-07-06: v2 line-up features are empty for current matches (v2 behaves like v1.6).
- Pinnacle columns are absent from 2026/27 files: the closing-odds baseline exists only for past seasons.

Data quirk: football-data writes 0 for a missing price; `FootballData.load` turns exact zeros into nulls (7 rows
in lower divisions), any other odds below 1 still fail schema validation. Pinnacle columns are absent from 2026/27 files, so the closing
baseline only exists for past seasons; `AvgH/BbAvH` (pre-closing average) exists everywhere, fixtures included.

## Rules

- No leakage. Features only reach matches through `pit.join`, which uses values dated strictly before the match day.
  Never feed a match's own columns (goals, shots, xG, odds) to a model; only `Dataset.feature_cols`.
- New source / feature / model = new module + `@REGISTRY.register`, then list it in a config. No edits elsewhere.
- New model version = new config file (`configs/orion-v2.yaml`). Never change a config that already has results in `runs/`.
- Betting odds are not model inputs in v1. The `market` feature exists for later versions; the `market` model is the baseline.
- Judge models by log-loss / RPS on the `with_odds` subset against `market`, not by accuracy.
- Raw files are recorded in `data/manifest.jsonl`. Do not edit raw files by hand.
- Data sources and their limits are described in `data.md`.
