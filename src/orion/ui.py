"""Terminal UI: live progress bars with ETA, and end-of-command dashboards. Only the CLI imports this module."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotext as plt
from rich.columns import Columns
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from rich.text import Text

from orion import progress
from orion.evaluation.metrics import evaluate

console = Console()

COLORS = {"market": "bright_black", "poisson": "cyan", "gbm": "magenta", "ensemble": "green"}
PLOT_COLORS = {"market": "gray", "poisson": "cyan", "gbm": "magenta", "ensemble": "green"}


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s", datefmt="%H:%M:%S",
                        handlers=[RichHandler(console=console, show_path=False, markup=False, rich_tracebacks=True)])


def header(name: str, command: str) -> None:
    console.print()
    console.rule(f"[bold]ORION[/]  [dim]·[/]  {name}  [dim]·[/]  [bold cyan]{command}[/]", style="cyan")


class _RichTask(progress.Task):
    def __init__(self, bar: Progress, task_id, label: str):
        self.bar, self.id, self.label = bar, task_id, label

    def advance(self, n: int = 1, label: str | None = None) -> None:
        if label:
            self.bar.update(self.id, description=f"{self.label} [dim]· {label}[/]")
        if n:
            self.bar.advance(self.id, n)


class _RichReporter(progress.Reporter):
    def __init__(self, bar: Progress):
        self.bar = bar

    def start(self, label: str, total: int | None) -> progress.Task:
        return _RichTask(self.bar, self.bar.add_task(label, total=total), label)

    def finish(self, task: _RichTask) -> None:
        t = self.bar.tasks[task.id]
        self.bar.update(task.id, completed=t.total if t.total is not None else 1, total=t.total or 1,
                        description=f"[green]✓[/] {task.label}")


@contextmanager
def live() -> Iterator[None]:
    bar = Progress(SpinnerColumn(), TextColumn("{task.description}", justify="left"), BarColumn(bar_width=32),
                   TaskProgressColumn(), MofNCompleteColumn(), TextColumn("[dim]elapsed[/]"), TimeElapsedColumn(),
                   TextColumn("[dim]eta[/]"), TimeRemainingColumn(), console=console)
    with bar, progress.using(_RichReporter(bar)):
        yield


def _panel_stats(rows: dict[str, str], title: str) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="bold")
    for k, v in rows.items():
        t.add_row(k, v)
    return Panel(t, title=title, border_style="cyan", expand=False)


def skill(log_loss: float, market: float) -> float:
    """Share of the closing market's improvement over random guessing that a model achieves."""
    return (np.log(3) - log_loss) / (np.log(3) - market)


def backtest_dashboard(predictions: pd.DataFrame, scores: pd.DataFrame, weights: dict[str, dict[str, float]],
                       n_features: int, elapsed: float, run: Path) -> None:
    odds = scores[(scores["subset"] == "with_odds") & (scores["season"] == "all")].set_index("model")
    allm = scores[(scores["subset"] == "all") & (scores["season"] == "all")].set_index("model")
    market = odds.loc["market", "log_loss"] if "market" in odds.index else np.nan
    best = odds.drop(index="market", errors="ignore")["log_loss"].idxmin()

    table = Table(title="Scores on matches with closing odds (lower is better)", title_justify="left",
                  border_style="cyan", header_style="bold")
    for col in ("model", "log-loss", "skill", "RPS", "Brier", "accuracy", "vs market", "matches"):
        table.add_column(col, justify="left" if col == "model" else "right")
    for model, r in odds.sort_values("log_loss").iterrows():
        gap = r["log_loss"] - market
        gap_txt = "[dim]baseline[/]" if model == "market" else f"[{'green' if gap < 0 else 'red'}]{gap:+.4f}[/]"
        name = f"[{COLORS.get(model, 'white')}]{model}[/]" + (" [yellow]★[/]" if model == best else "")
        table.add_row(name, f"{r['log_loss']:.4f}", f"{skill(r['log_loss'], market):.1%}", f"{r['rps']:.4f}",
                      f"{r['brier']:.4f}", f"{r['accuracy']:.1%}", gap_txt, f"{int(r['n']):,}")
    table.caption = "skill: 0% = random guessing (log-loss ln 3), 100% = closing market, above 100% beats it"

    console.print()
    console.print(table)
    console.print(_curve(scores))
    console.print(_by_league(predictions, best))

    beats = odds.loc[best, "log_loss"] < market
    verdict = ("[green]beats the market[/]" if beats else "[red]does not beat the market[/]")
    console.print(Columns([
        _panel_stats({
            "test seasons": ", ".join(sorted(predictions["season"].unique())),
            "test matches": f"{predictions['match_id'].nunique():,}",
            "with closing odds": f"{int(odds.loc[best, 'n']):,}",
            "features": str(n_features),
            "duration": f"{elapsed:.1f} s",
        }, "Run"),
        _panel_stats({
            "best model": f"{best}  ({odds.loc[best, 'log_loss']:.4f})",
            "skill": f"{skill(odds.loc[best, 'log_loss'], market):.1%} of the market",
            "market": f"{market:.4f}",
            "gap": f"{odds.loc[best, 'log_loss'] - market:+.4f}",
            "accuracy (all matches)": f"{allm.loc[best, 'accuracy']:.1%}",
            "verdict": verdict,
        }, "Verdict"),
        *([_weights_table(weights)] if any(weights.values()) else []),
    ]))
    console.print(f"[dim]saved to {run}[/]\n")


def _weights_table(weights: dict[str, dict[str, float]]) -> Table:
    members = sorted({m for w in weights.values() for m in w})
    table = Table(title="Ensemble weights per test season", title_justify="left", border_style="cyan", header_style="bold")
    table.add_column("season")
    for m in members:
        table.add_column(m, justify="right")
    for season, w in weights.items():
        table.add_row(season, *[f"{w.get(m, 0):.0%}" for m in members])
    return table


def _curve(scores: pd.DataFrame) -> Text:
    s = scores[(scores["subset"] == "with_odds") & (scores["season"] != "all")]
    seasons = sorted(s["season"].unique())
    labels = [f"{x[:2]}/{x[2:]}" for x in seasons]
    plt.clf()
    plt.theme("clear")
    plt.plot_size(min(console.width, 100), 16)
    for model, g in s.groupby("model"):
        g = g.set_index("season").reindex(seasons)
        plt.plot(list(range(len(seasons))), g["log_loss"].tolist(), label=model, marker="braille",
                 color=PLOT_COLORS.get(model, "white"))
    plt.xticks(list(range(len(seasons))), labels)
    plt.title("Log-loss per test season (lower is better)")
    return Text.from_ansi(plt.build())


def _by_league(predictions: pd.DataFrame, best: str) -> Table:
    p = predictions[predictions["has_odds"]]
    table = Table(title=f"By league: {best} vs market (log-loss)", title_justify="left", border_style="cyan",
                  header_style="bold")
    for col in ("league", best, "market", "gap", "matches"):
        table.add_column(col, justify="left" if col == "league" else "right")
    for div, g in p.groupby("div"):
        score = {m: evaluate(x["outcome"], x[["p_home", "p_draw", "p_away"]].to_numpy())
                 for m, x in g.groupby("model") if m in (best, "market")}
        if len(score) < 2:
            continue
        gap = score[best]["log_loss"] - score["market"]["log_loss"]
        table.add_row(div, f"{score[best]['log_loss']:.4f}", f"{score['market']['log_loss']:.4f}",
                      f"[{'green' if gap < 0 else 'red'}]{gap:+.4f}[/]", f"{score[best]['n']:,}")
    return table


def train_dashboard(meta: dict, models: list[str], elapsed: float, folder: Path) -> None:
    console.print(_panel_stats({
        "matches": f"{meta['train_rows']:,}",
        "seasons": f"{meta['train_seasons'][0]} → {meta['train_seasons'][-1]}",
        "last match": meta["last_match_date"],
        "features": str(len(meta["feature_cols"])),
        "models": ", ".join(models),
        "duration": f"{elapsed:.1f} s",
        "artifact": str(folder),
    }, "Trained"))


def predict_dashboard(out: pd.DataFrame, best: str, elapsed: float, run: Path) -> None:
    table = Table(title=f"Next matches · {best} probabilities", title_justify="left", border_style="cyan",
                  header_style="bold")
    for col in ("date", "league", "home", "away", "1", "X", "2", "pick"):
        table.add_column(col, justify="right" if col in ("1", "X", "2") else "left")
    for r in out.itertuples():
        p = [getattr(r, f"{best}_home"), getattr(r, f"{best}_draw"), getattr(r, f"{best}_away")]
        top = int(np.argmax(p))
        cells = [f"[bold green]{v:.0%}[/]" if i == top else f"{v:.0%}" for i, v in enumerate(p)]
        pick = [r.home, "draw", r.away][top]
        bar = "█" * round(p[top] * 10)
        table.add_row(r.date.strftime("%a %d %b"), r.div, r.home, r.away, *cells, f"{pick} [green]{bar}[/]")
    console.print(table)
    console.print(f"[dim]{len(out)} matches · {elapsed:.1f} s · saved to {run}[/]\n")
