"""Manager tenure: league games each side's current manager had already coached before this match.

The manager on the bench is known before kickoff; the count uses previous games only. Short tenures capture the
"new manager" effect. Capped at `cap` games.
"""

import pandas as pd

from orion.features.base import FEATURES, Context, Feature, MatchLevel
from orion.features.tm import linked_games


@FEATURES.register
class Manager(Feature):
    name = "manager"

    def __init__(self, cap: int = 20, new_within: int = 5):
        self.cap, self.new_within = cap, new_within

    def compute(self, ctx: Context) -> MatchLevel:
        pairs = linked_games(ctx)
        long = pd.concat([pd.DataFrame({"row": pairs.index, "side": side, "team": pairs[f"{side}_id"],
                                        "date": pairs["tm_date"], "manager": pairs[f"{side}_manager"]})
                          for side in ("home", "away")]).sort_values(["team", "date"], kind="stable")
        long = long.reset_index(drop=True)
        run =(long["manager"] != long.groupby("team")["manager"].shift()).cumsum()  # new run at each change
        long["tenure"] = long.groupby(run).cumcount().clip(upper=self.cap).astype(float)
        long.loc[long["manager"].isna(), "tenure"] = float("nan")
        frame = pairs[["match_id"]].copy()
        for side in ("home", "away"):
            tenure = long[long["side"] == side].set_index("row")["tenure"].reindex(pairs.index)
            frame[f"{side}_tenure"] = tenure.to_numpy()
            frame[f"{side}_new"] = (tenure < self.new_within).astype(float).where(tenure.notna()).to_numpy()
        return MatchLevel(frame)
