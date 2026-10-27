"""Days since each team's previous match (capped), from matches in the dataset."""

import pandas as pd

from orion.features.base import FEATURES, Context, Feature, TeamSeries, team_long


@FEATURES.register
class Rest(Feature):
    name = "rest"

    def __init__(self, cap_days: int = 30):
        self.cap_days = cap_days

    def compute(self, ctx: Context) -> TeamSeries:
        long = team_long(ctx.played)
        return TeamSeries(pd.DataFrame({"team": long["team"], "available_at": long["date"], "last_played": long["date"]}))

    def derive(self, df: pd.DataFrame) -> pd.DataFrame:
        for side in ("home", "away"):
            days = (df["date"] - df[f"{side}_rest_last_played"]).dt.days
            df[f"{side}_rest_days"] = days.clip(upper=self.cap_days).astype("float64")
            df = df.drop(columns=f"{side}_rest_last_played")
        return df
