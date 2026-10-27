"""Referee tendencies from his previous matches in the dataset (football-data `Referee`, not every league).

Per match: n_prev, home_win_rate, cards_per_match, fouls_per_match, each shrunk towards the league-wide mean
with `prior` pseudo-matches. Computed with an expanding mean shifted by one match: never the match itself.
"""

from orion import leagues
from orion.features.base import FEATURES, Context, Feature, MatchLevel


@FEATURES.register
class Referee(Feature):
    name = "referee"

    def __init__(self, prior: float = 20):
        self.prior = prior

    def compute(self, ctx: Context) -> MatchLevel:
        m = ctx.matches.sort_values(["date", "match_id"], kind="stable").copy()
        country = m["div"].map(lambda d: leagues.get(d).country)
        m["ref"] = country + "/" + m["referee"].astype(str)  # same name in two countries: two referees
        played = m["home_goals"].notna()
        m["home_win"] = (m["home_goals"] > m["away_goals"]).astype(float).where(played)
        m["cards"] = (m[["home_yellow", "away_yellow", "home_red", "away_red"]].sum(axis=1, min_count=1)).where(played)
        m["fouls"] = (m[["home_fouls", "away_fouls"]].sum(axis=1, min_count=1)).where(played)

        frame = m[["match_id"]].copy()
        known = m["referee"].notna()
        g = m[known].groupby("ref")
        frame.loc[known, "n_prev"] = g["home_win"].transform(lambda s: s.shift().expanding().count()).fillna(0)
        for col in ("home_win", "cards", "fouls"):
            # League-wide mean of all matches played before the match day (never the day itself).
            daily = m.groupby("date")[col].agg(["sum", "count"]).cumsum().shift()
            prior_mean = (daily["sum"] / daily["count"]).reindex(m["date"]).fillna(0).to_numpy()[known.to_numpy()]
            total = g[col].transform(lambda s: s.shift().expanding().sum()).fillna(0)
            count = g[col].transform(lambda s: s.shift().expanding().count()).fillna(0)
            name = {"home_win": "home_win_rate", "cards": "cards_per_match", "fouls": "fouls_per_match"}[col]
            frame.loc[known, name] = (total + self.prior * prior_mean) / (count + self.prior)
        return MatchLevel(frame)
