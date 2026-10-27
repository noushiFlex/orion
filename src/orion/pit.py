"""Point-in-time joins: the single place where features meet matches.

TeamSeries values are taken from the latest row dated strictly before the match day (allow_exact_matches=False),
so a match never sees its own result or anything later. Values older than max_age become missing.
"""

import pandas as pd

from orion.features.base import MatchLevel, TeamSeries


def join(df: pd.DataFrame, values: TeamSeries | MatchLevel, name: str) -> pd.DataFrame:
    if isinstance(values, MatchLevel):
        return df.join(values.frame.set_index("match_id").add_prefix(f"{name}_"), on="match_id")

    right = values.frame.copy()
    right["team"] = right["team"].astype(str)  # merge_asof needs identical key dtypes on both sides
    right["available_at"] = right["available_at"].astype("datetime64[ns]")
    right = right.sort_values("available_at", kind="stable")
    value_cols = [c for c in right.columns if c not in ("team", "available_at")]
    for side in ("home", "away"):
        left = df[["match_id", "date", f"{side}_id"]].copy()
        left["date"] = left["date"].astype("datetime64[ns]")
        left[f"{side}_id"] = left[f"{side}_id"].astype(str)
        left = left.sort_values("date", kind="stable")
        joined = pd.merge_asof(left, right, left_on="date", right_on="available_at", left_by=f"{side}_id",
                               right_by="team", direction="backward", allow_exact_matches=False,
                               tolerance=values.max_age)
        joined = joined.set_index("match_id")[value_cols].add_prefix(f"{side}_{name}_")
        df = df.join(joined, on="match_id")
    return df
