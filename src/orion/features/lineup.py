"""Starting-XI strength from Transfermarkt market values.

For each match and side:
    log_value     mean log market value of the starters (each valued strictly before the match day)
    vs_usual      log_value minus the team's mean over its previous `window` matches: rotation, absences
    value_m       total XI value in EUR millions
plus lineup_log_diff (home - away). The line-up itself is public about an hour before kickoff, so a model
using this feature predicts at line-up time, later than the pre-closing odds it is compared with.
"""

import logging

import numpy as np
import pandas as pd

from orion.features.base import FEATURES, Context, Feature, MatchLevel
from orion.features.tm import linked_games, per_side
from orion.sources.transfermarkt import Transfermarkt

log = logging.getLogger(__name__)


@FEATURES.register
class Lineup(Feature):
    name = "lineup"

    def __init__(self, window: int = 10, min_valued: int = 8):
        self.window, self.min_valued = window, min_valued

    def compute(self, ctx: Context) -> MatchLevel:
        xi = Transfermarkt().xi_values(ctx.cfg)
        xi.loc[xi["n_valued"] < self.min_valued, ["value_m", "log_value"]] = np.nan  # too few valued starters
        pairs = linked_games(ctx)
        frame = per_side(pairs, xi, ["log_value", "value_m"])
        usual = self._usual(pairs, frame)
        for side in ("home", "away"):
            frame[f"{side}_vs_usual"] = frame[f"{side}_log_value"] - usual[side]
        frame["log_diff"] = frame["home_log_value"] - frame["away_log_value"]
        log.info("lineup: XI values for %d matches", int(frame["home_log_value"].notna().sum()))
        return MatchLevel(frame)

    def _usual(self, pairs: pd.DataFrame, frame: pd.DataFrame) -> dict[str, pd.Series]:
        """Each side's usual XI value: mean over the team's previous `window` games (shift(1): never this game)."""
        long = pd.concat([pd.DataFrame({"row": pairs.index, "side": side, "team": pairs[f"{side}_id"],
                                        "date": pairs["tm_date"], "v": frame[f"{side}_log_value"]})
                          for side in ("home", "away")]).sort_values(["team", "date"], kind="stable")
        long = long.reset_index(drop=True)
        long["usual"] = long.groupby("team")["v"].transform(lambda s: s.shift(1).rolling(self.window, min_periods=3).mean())
        return {side: long[long["side"] == side].set_index("row")["usual"].reindex(pairs.index) for side in ("home", "away")}
