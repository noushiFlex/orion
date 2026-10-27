"""Team-name mapping between a secondary source and football-data.co.uk (the reference names).

Names are learned, not hand-written: matches are paired when division, score and date (+-1 day) agree and the pair
is unambiguous; each source name maps to the reference name that wins the vote. A mapping is accepted only with
enough votes and a clear majority. Several source spellings may map to one club (aliases). configs/team_overrides.csv (div,source,source_name,ref_name) wins over votes.
"""

import logging

import pandas as pd

from orion.paths import CONFIGS, INTERIM

log = logging.getLogger(__name__)

OVERRIDES = CONFIGS / "team_overrides.csv"


def build_name_map(source: pd.DataFrame, reference: pd.DataFrame, tolerance_days: int = 1,
                   min_votes: int = 3, min_share: float = 0.9) -> pd.DataFrame:
    """Both frames need div, date, home, away, home_goals, away_goals. Returns one row per (div, source_name)."""
    key = ["div", "home_goals", "away_goals"]
    s = source.dropna(subset=key[1:]).reset_index(drop=True)
    s["sid"] = s.index
    r = reference.dropna(subset=key[1:])
    pairs = s.merge(r[key + ["date", "home", "away"]], on=key, suffixes=("_s", "_r"))
    pairs = pairs[(pairs["date_s"] - pairs["date_r"]).abs() <= pd.Timedelta(days=tolerance_days)]
    pairs = pairs[pairs.groupby("sid")["sid"].transform("size") == 1]  # unambiguous pairs only

    votes = pd.concat([
        pairs[["div", f"{side}_s", f"{side}_r"]].set_axis(["div", "source_name", "ref_name"], axis=1)
        for side in ("home", "away")
    ])
    counts = votes.value_counts().rename("votes").reset_index()
    counts["share"] = counts["votes"] / counts.groupby(["div", "source_name"])["votes"].transform("sum")
    best = counts.sort_values("votes", ascending=False).drop_duplicates(["div", "source_name"])
    best["accepted"] = (best["votes"] >= min_votes) & (best["share"] >= min_share)

    # Several source names for one club are spelling aliases (e.g. "MGladbach" / "M'gladbach"): kept, but logged.
    accepted = best[best["accepted"]]
    aliases = accepted[accepted.duplicated(["div", "ref_name"], keep=False)]
    if len(aliases):
        log.info("aliases: %s", aliases.groupby(["div", "ref_name"])["source_name"].agg(list).to_dict())
    return best.sort_values(["div", "source_name"]).reset_index(drop=True)


def load_or_build(source_name: str, source: pd.DataFrame, reference: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    """Cached in data/interim/mappings/. Rebuilt when asked or when the source covers divisions the cache lacks."""
    path = INTERIM / "mappings" / f"{source_name}_to_football_data.csv"
    if path.exists() and not rebuild:
        cached = pd.read_csv(path)
        if set(source["div"]) <= set(cached["div"]):
            return _with_overrides(source_name, cached)
    built = build_name_map(source, reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    built.to_csv(path, index=False)
    names = source[["div", "home"]].drop_duplicates().shape[0]
    log.info("%s: %d/%d names mapped -> %s", source_name, int(built["accepted"].sum()), names, path.name)
    return _with_overrides(source_name, built)


def _with_overrides(source_name: str, mapping: pd.DataFrame) -> pd.DataFrame:
    if not OVERRIDES.exists():
        return mapping
    o = pd.read_csv(OVERRIDES)
    o = o[o["source"] == source_name].assign(votes=0, share=1.0, accepted=True)[mapping.columns]
    return pd.concat([o, mapping]).drop_duplicates(["div", "source_name"], keep="first")


def apply(df: pd.DataFrame, mapping: pd.DataFrame, label: str) -> pd.DataFrame:
    """Translate home/away into reference names; rows with an unmapped team are dropped and counted."""
    ok = mapping[mapping["accepted"]].set_index(["div", "source_name"])["ref_name"]
    out = df.copy()
    for side in ("home", "away"):
        out[side] = pd.MultiIndex.from_arrays([out["div"], out[side]]).map(ok)
    dropped = out[["home", "away"]].isna().any(axis=1)
    if dropped.any():
        log.info("%s: %d rows dropped, team not mapped", label, int(dropped.sum()))
    return out[~dropped]
