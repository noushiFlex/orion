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

from orion import leagues, mapping
from orion.features.base import FEATURES, Context, Feature, MatchLevel
from orion.sources.transfermarkt import Transfermarkt

log = logging.getLogger(__name__)


@FEATURES.register
class Lineup(Feature):
    name = "lineup"

    def __init__(self, window: int = 10, min_valued: int = 8):
        self.window, self.min_valued = window, min_valued

    def compute(self, ctx: Context) -> MatchLevel:
        tm = Transfermarkt().load(ctx.cfg)
        names = mapping.load_or_build("transfermarkt", tm.dropna(subset=["home_goals", "away_goals"]), ctx.played)
        tm = mapping.apply(tm, names, "transfermarkt")
        for side in ("home", "away"):  # too few valued starters: the XI value is not comparable
            thin = tm[f"{side}_n_valued"].fillna(0) < self.min_valued
            tm.loc[thin, [f"{side}_value_m", f"{side}_log_value"]] = np.nan
        tm = tm.assign(**{f"{s}_id": tm["div"].map(lambda d: leagues.get(d).country) + "/" + tm[s] for s in ("home", "away")})
        tm = self._with_usual(tm)

        pairs = ctx.matches[["match_id", "div", "date", "home", "away"]].merge(
            tm.rename(columns={"date": "tm_date"}), on=["div", "home", "away"])
        pairs["gap"] = (pairs["date"] - pairs["tm_date"]).abs()
        pairs = pairs[pairs["gap"] <= pd.Timedelta(days=1)].sort_values("gap").drop_duplicates("match_id")
        log.info("lineup: XI values for %d matches", int(pairs["home_log_value"].notna().sum()))

        frame = pairs[["match_id"]].copy()
        for side in ("home", "away"):
            frame[f"{side}_log_value"] = pairs[f"{side}_log_value"].to_numpy()
            frame[f"{side}_vs_usual"] = (pairs[f"{side}_log_value"] - pairs[f"{side}_usual"]).to_numpy()
            frame[f"{side}_value_m"] = pairs[f"{side}_value_m"].to_numpy()
        frame["log_diff"] = frame["home_log_value"] - frame["away_log_value"]
        return MatchLevel(frame)

    def _with_usual(self, tm: pd.DataFrame) -> pd.DataFrame:
        """Each side's usual XI value: mean over the team's previous `window` games (shift(1): never this game)."""
        long = pd.concat([
            pd.DataFrame({"row": tm.index, "side": side, "team": tm[f"{side}_id"], "date": tm["date"],
                          "log_value": tm[f"{side}_log_value"]}) for side in ("home", "away")
        ]).sort_values(["team", "date"], kind="stable")
        long["usual"] = long.groupby("team")["log_value"].transform(
            lambda s: s.shift(1).rolling(self.window, min_periods=3).mean())
        out = tm.copy()
        for side in ("home", "away"):
            out[f"{side}_usual"] = long[long["side"] == side].set_index("row")["usual"]
        return out
