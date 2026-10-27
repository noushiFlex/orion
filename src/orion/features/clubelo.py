"""ClubElo rating snapshots (cross-country scale, history back to 2000), joined to teams via a learned name map."""

import logging

import pandas as pd

from orion import leagues, mapping
from orion.features.base import FEATURES, Context, Feature, TeamSeries
from orion.sources.clubelo import ClubElo as ClubEloSource

log = logging.getLogger(__name__)


@FEATURES.register
class ClubElo(Feature):
    name = "clubelo"

    def __init__(self, max_age_days: int = 45):
        self.max_age = pd.Timedelta(days=max_age_days)

    def compute(self, ctx: Context) -> TeamSeries:
        src = ClubEloSource()
        names = mapping.load_or_build("clubelo", src.load_results(ctx.cfg), ctx.played)
        names = names[names["accepted"]]
        names["team"] = names["div"].map(lambda d: leagues.get(d).country) + "/" + names["ref_name"]
        team_of = names.drop_duplicates("source_name").set_index("source_name")["team"]

        ratings = src.load(ctx.cfg)
        ratings["team"] = ratings["club"].map(team_of)
        ratings = ratings.dropna(subset=["team"])
        log.info("clubelo: %d clubs matched to dataset teams", ratings["team"].nunique())
        frame = ratings.rename(columns={"date": "available_at", "elo": "rating"})[["team", "available_at", "rating"]]
        return TeamSeries(frame, max_age=self.max_age)

    def derive(self, df: pd.DataFrame) -> pd.DataFrame:
        df["clubelo_diff"] = df["home_clubelo_rating"] - df["away_clubelo_rating"]
        return df
