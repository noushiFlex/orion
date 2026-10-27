"""Column contracts. Sources validate on load so an upstream format change fails loudly, never silently."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Col:
    name: str
    kind: str  # "str" | "float" | "date"
    required: bool = True  # must exist upstream; optional columns are added as nulls
    nullable: bool = True
    min: float | None = None


@dataclass(frozen=True)
class Schema:
    name: str
    cols: tuple[Col, ...]

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with exactly the declared columns, coerced types, and checked constraints."""
        errors = []
        out = pd.DataFrame(index=df.index)
        for c in self.cols:
            if c.name not in df.columns:
                if c.required:
                    errors.append(f"missing column {c.name!r}")
                out[c.name] = pd.NaT if c.kind == "date" else np.nan
                continue
            s = df[c.name]
            if c.kind == "float":
                coerced = pd.to_numeric(s, errors="coerce").astype("float64")
            elif c.kind == "date":
                coerced = pd.to_datetime(s, errors="coerce").astype("datetime64[ns]")
            else:
                coerced = s.where(s.isna(), s.astype(str).str.strip()).astype(object)
            garbage = s.notna() & coerced.isna()
            if garbage.any():
                errors.append(f"{c.name}: {int(garbage.sum())} unparseable values, e.g. {s[garbage].iloc[0]!r}")
            if not c.nullable and coerced.isna().any():
                errors.append(f"{c.name}: {int(coerced.isna().sum())} nulls")
            if c.min is not None and (coerced < c.min).any():
                errors.append(f"{c.name}: values below {c.min}")
            out[c.name] = coerced
        if errors:
            raise SchemaError(f"{self.name}: " + "; ".join(errors))
        return out


def _stats(*names: str) -> tuple[Col, ...]:
    return tuple(Col(f"{side}_{n}", "float", required=False, min=0) for n in names for side in ("home", "away"))


# One row per match, played or scheduled. Goals are null for scheduled fixtures.
MATCHES = Schema("matches", (
    Col("match_id", "str", nullable=False),
    Col("div", "str", nullable=False),
    Col("season", "str", nullable=False),
    Col("date", "date", nullable=False),
    Col("time", "str", required=False),
    Col("home", "str", nullable=False),
    Col("away", "str", nullable=False),
    Col("home_id", "str", nullable=False),
    Col("away_id", "str", nullable=False),
    Col("referee", "str", required=False),
    *_stats("goals", "shots", "sot", "corners", "fouls", "yellow", "red", "xg"),
    Col("odds_close_home", "float", required=False, min=1),
    Col("odds_close_draw", "float", required=False, min=1),
    Col("odds_close_away", "float", required=False, min=1),
    Col("odds_pre_home", "float", required=False, min=1),
    Col("odds_pre_draw", "float", required=False, min=1),
    Col("odds_pre_away", "float", required=False, min=1),
    Col("odds_max_home", "float", required=False, min=1),
    Col("odds_max_draw", "float", required=False, min=1),
    Col("odds_max_away", "float", required=False, min=1),
    Col("odds_b365_home", "float", required=False, min=1),
    Col("odds_b365_draw", "float", required=False, min=1),
    Col("odds_b365_away", "float", required=False, min=1),
))

# Per-match xG from a secondary source, in that source's team names.
XG = Schema("xg", (
    Col("div", "str", nullable=False),
    Col("date", "date", nullable=False),
    Col("home", "str", nullable=False),
    Col("away", "str", nullable=False),
    Col("home_goals", "float", nullable=False, min=0),
    Col("away_goals", "float", nullable=False, min=0),
    Col("home_xg", "float", min=0),
    Col("away_xg", "float", min=0),
))

# Dated club ratings snapshots.
RATINGS = Schema("ratings", (
    Col("club", "str", nullable=False),
    Col("country", "str"),
    Col("date", "date", nullable=False),
    Col("elo", "float", nullable=False),
))

# Results from a secondary source, used only to learn team-name mappings.
RESULTS = Schema("results", (
    Col("div", "str", nullable=False),
    Col("date", "date", nullable=False),
    Col("home", "str", nullable=False),
    Col("away", "str", nullable=False),
    Col("home_goals", "float", nullable=False, min=0),
    Col("away_goals", "float", nullable=False, min=0),
))
