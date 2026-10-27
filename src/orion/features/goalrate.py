"""Online attack/defence ratings: a streaming Dixon-Coles.

Each team has an attack and a defence rating on the log scale. Expected goals for a match are
    home: exp(base + home_adv + att[home] - def[away]),   away: exp(base + att[away] - def[home]).
After every match, one gradient step on the Poisson log-likelihood of the observed value moves the four ratings
(and the global base / home advantage). Ratings can be learned from goals and from xG (separate rating sets).
Between seasons ratings shrink towards 0; teams first seen after the first season start below average.
"""

import math

import pandas as pd

from orion.features.base import FEATURES, Context, Feature, TeamSeries


@FEATURES.register
class GoalRate(Feature):
    name = "goalrate"

    def __init__(self, targets: list[str] = ("goals", "xg"), lr: float = 0.04, lr_global: float = 0.002,
                 season_shrink: float = 0.75, newcomer: float = -0.15):
        unknown = set(targets) - {"goals", "xg"}
        if unknown:
            raise ValueError(f"goalrate targets must be goals or xg, got {sorted(unknown)}")
        self.targets, self.lr, self.lr_global = list(targets), lr, lr_global
        self.season_shrink, self.newcomer = season_shrink, newcomer

    def compute(self, ctx: Context) -> TeamSeries:
        played = ctx.played
        first_season = played["season"].iloc[0] if len(played) else None
        state = {t: {"att": {}, "def": {}, "base": math.log(1.3), "hfa": 0.25} for t in self.targets}
        last_season: dict[str, str] = {}
        rows = []
        cols = ["season", "date", "home_id", "away_id", "home_goals", "away_goals", "home_xg", "away_xg"]
        for season, date, h, a, hg, ag, hx, ax in played[cols].itertuples(index=False):
            for team in (h, a):
                if team not in last_season:
                    init = 0.0 if season == first_season else self.newcomer
                    for s in state.values():
                        s["att"][team], s["def"][team] = init, init
                elif last_season[team] != season:
                    for s in state.values():
                        s["att"][team] *= self.season_shrink
                        s["def"][team] *= self.season_shrink
                last_season[team] = season
            values = {"goals": (hg, ag), "xg": (hx, ax)}
            for target, s in state.items():
                yh, ya = values[target]
                if pd.isna(yh) or pd.isna(ya):  # no xG for this match: ratings stay as they are
                    continue
                att, dfn = s["att"], s["def"]
                lam_h = math.exp(s["base"] + s["hfa"] + att[h] - dfn[a])
                lam_a = math.exp(s["base"] + att[a] - dfn[h])
                gh, ga = yh - lam_h, ya - lam_a
                att[h] += self.lr * gh
                dfn[a] -= self.lr * gh
                att[a] += self.lr * ga
                dfn[h] -= self.lr * ga
                s["base"] += self.lr_global * (gh + ga)
                s["hfa"] += self.lr_global * gh
            for team in (h, a):
                row = {"team": team, "available_at": date}
                for target, s in state.items():
                    row[f"att_{target}"], row[f"def_{target}"] = s["att"][team], s["def"][team]
                rows.append(row)
        return TeamSeries(pd.DataFrame(rows))

    def derive(self, df: pd.DataFrame) -> pd.DataFrame:
        # Log-scale expected-goal edges: attack of one side against the defence of the other.
        for t in self.targets:
            df[f"goalrate_edge_home_{t}"] = df[f"home_goalrate_att_{t}"] - df[f"away_goalrate_def_{t}"]
            df[f"goalrate_edge_away_{t}"] = df[f"away_goalrate_att_{t}"] - df[f"home_goalrate_def_{t}"]
        return df
