"""Shared plumbing for Transfermarkt-based features: link Transfermarkt games to dataset matches."""

import logging

import pandas as pd

from orion import leagues, mapping
from orion.features.base import Context
from orion.sources.transfermarkt import Transfermarkt

log = logging.getLogger(__name__)


_CACHE: dict[tuple, pd.DataFrame] = {}


def _games(cfg) -> pd.DataFrame:
    key = (cfg.leagues, cfg.seasons[0])  # several TM features share one load per dataset build
    if key not in _CACHE:
        _CACHE[key] = Transfermarkt().load(cfg)
    return _CACHE[key]


def linked_games(ctx: Context) -> pd.DataFrame:
    """match_id -> game_id, home/away club ids and team ids (football-data names), date within one day."""
    games = _games(ctx.cfg)
    names = mapping.load_or_build("transfermarkt", games.dropna(subset=["home_goals", "away_goals"]), ctx.played)
    games = mapping.apply(games, names, "transfermarkt")
    pairs = ctx.matches[["match_id", "div", "date", "home", "away"]].merge(
        games.drop(columns=["home_goals", "away_goals"]).rename(columns={"date": "tm_date"}), on=["div", "home", "away"])
    pairs["gap"] = (pairs["date"] - pairs["tm_date"]).abs()
    pairs = pairs[pairs["gap"] <= pd.Timedelta(days=1)].sort_values("gap").drop_duplicates("match_id")
    country = pairs["div"].map(lambda d: leagues.get(d).country)
    pairs["home_id"], pairs["away_id"] = country + "/" + pairs["home"], country + "/" + pairs["away"]
    return pairs.drop(columns=["gap"]).reset_index(drop=True)


def per_side(pairs: pd.DataFrame, values: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Attach per-(game_id, club_id) values to both sides: <side>_<col> columns keyed by match_id."""
    out = pairs[["match_id"]].copy()
    v = values.set_index(["game_id", "club_id"])[cols]
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([pairs["game_id"], pairs[f"{side}_club_id"]])
        out[[f"{side}_{c}" for c in cols]] = v.reindex(idx).to_numpy()
    return out
