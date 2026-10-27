"""Build the README figures from the latest backtest runs in runs/.

    ./.venv/bin/pip install -e ".[docs]"
    ./.venv/bin/python docs/make_figures.py

Writes docs/img/{versions,calibration}-{light,dark}.png and docs/img/dashboard.svg.
"""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402

from orion import ui  # noqa: E402
from orion.paths import PROCESSED, RUNS  # noqa: E402

OUT = Path(__file__).parent / "img"
VERSIONS = ["v1", "v1.1", "v1.2", "v1.3", "v1.4", "v1.5", "v1.6", "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5"]
THEMES = {  # reference palette (dataviz skill), validated for two series in both modes
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "orion": "#2a78d6", "market": "#eb6834"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#33322f",
             "orion": "#3987e5", "market": "#d95926"},
}


def latest_run(name: str) -> Path:
    return Path(sorted(glob.glob(str(RUNS / f"*-orion-{name}-backtest")))[-1])


def best_log_loss(name: str) -> tuple[float, float, float]:
    """Best non-market model, closing market and pre-closing market log-loss on the with-odds subset."""
    s = pd.read_csv(latest_run(name) / "scores.csv")
    s = s[(s["subset"] == "with_odds") & (s["season"] == "all")].set_index("model")["log_loss"]
    models = s.drop(index=[m for m in ("market", "market_pre") if m in s.index])
    return float(models.min()), float(s["market"]), float(s.get("market_pre", np.nan))


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def versions_chart(theme: str):
    t = THEMES[theme]
    rows = [best_log_loss(v) for v in VERSIONS]
    best, closing = [r[0] for r in rows], rows[0][1]
    pre = next(r[2] for r in rows if not np.isnan(r[2]))
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=t["surface"])
    style(ax, t)
    x = np.arange(len(VERSIONS))
    ax.axhline(closing, color=t["market"], linewidth=2)
    ax.axhline(pre, color=t["muted"], linewidth=1.5, linestyle=(0, (4, 3)))
    ax.plot(x, best, color=t["orion"], linewidth=2, marker="o", markersize=8,
            markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
    ax.text(0, closing + 0.0003, "Pinnacle closing odds", color=t["text"], fontsize=9, ha="left", va="bottom")
    ax.text(0, pre + 0.0003, "pre-closing market average", color=t["text"], fontsize=9, ha="left", va="bottom")
    ax.set_ylim(closing - 0.0012, max(best) + 0.0015)
    for i in (0, VERSIONS.index("v2.1")):
        ax.annotate(f"{best[i]:.4f}", (x[i], best[i]), textcoords="offset points", xytext=(0, 10),
                    ha="center", color=t["text"], fontsize=9)
    ax.set_xticks(x, VERSIONS)
    ax.set_ylabel("log-loss (lower is better)", color=t["muted"], fontsize=9)
    ax.set_title("Orion: best model per version, 6,228 Big-Five matches (2022/23 to 2025/26)",
                 color=t["text"], fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / f"versions-{theme}.png", dpi=160, facecolor=t["surface"])
    plt.close(fig)


def calibration_chart(theme: str):
    t = THEMES[theme]
    p = pd.read_parquet(latest_run("v2.1") / "predictions.parquet")
    p = p[p["has_odds"]]
    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor=t["surface"])
    style(ax, t)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.plot([0, 0.9], [0, 0.9], color=t["muted"], linewidth=1, linestyle=(0, (4, 3)), label="perfect calibration")
    for model, label, color in (("ensemble", "Orion v2.1", t["orion"]), ("market", "Pinnacle closing", t["market"])):
        g = p[p["model"] == model]
        probs = g[["p_home", "p_draw", "p_away"]].to_numpy().ravel()
        hits = np.eye(3)[g["outcome"].to_numpy()].ravel()
        bins = np.clip((probs * 10).astype(int), 0, 9)
        df = pd.DataFrame({"b": bins, "p": probs, "o": hits}).groupby("b").agg(p=("p", "mean"), o=("o", "mean"), n=("p", "size"))
        df = df[df["n"] >= 30]
        ax.plot(df["p"], df["o"], color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=t["surface"], markeredgewidth=2, label=label, zorder=3)
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("predicted probability", color=t["muted"], fontsize=9)
    ax.set_ylabel("observed frequency", color=t["muted"], fontsize=9)
    ax.set_title("Calibration, all three outcomes", color=t["text"], fontsize=11, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in leg.get_texts():
        text.set_color(t["text"])
    fig.tight_layout()
    fig.savefig(OUT / f"calibration-{theme}.png", dpi=160, facecolor=t["surface"])
    plt.close(fig)


def dashboard_svg():
    run = latest_run("v2.1")
    ui.console = Console(record=True, width=118, force_terminal=True)
    predictions = pd.read_parquet(run / "predictions.parquet")
    weights = json.loads((run / "ensemble_weights.json").read_text())
    ui.backtest_dashboard(predictions, pd.read_csv(run / "scores.csv", dtype={"season": str}), weights,
                          n_features=len(json.loads((PROCESSED / "orion-v2.1" / "meta.json").read_text())["feature_cols"]),
                          elapsed=0.0, run=Path("runs") / run.name)
    ui.console.save_svg(str(OUT / "dashboard.svg"), title="orion -c configs/orion-v2.1.yaml backtest")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        versions_chart(theme)
        calibration_chart(theme)
    dashboard_svg()
    print("written:", sorted(p.name for p in OUT.iterdir()))
