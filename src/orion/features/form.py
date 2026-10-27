"""Rolling form: mean of each stat (for and against) and points over the last N matches.

`stats` picks from goals, shots, sot, corners, xg, fouls, yellow, red. The default keeps the v1 feature set, so
older configs build the same columns.
"""

from orion.features.base import FEATURES, Context, Feature, TeamSeries, team_long


@FEATURES.register
class Form(Feature):
    name = "form"

    def __init__(self, windows: list[int] = (5,), min_periods: int = 3,
                 stats: list[str] = ("goals", "shots", "sot", "corners", "xg")):
        self.windows = list(windows)
        self.min_periods = min_periods
        self.stats = list(stats)

    def compute(self, ctx: Context) -> TeamSeries:
        long = team_long(ctx.played)
        stats = [f"{s}_{side}" for s in self.stats for side in ("for", "against")] + ["points"]
        out = long[["team", "date"]].rename(columns={"date": "available_at"})
        grouped = long.groupby("team", sort=False)[stats]
        for w in self.windows:
            rolled = grouped.rolling(w, min_periods=self.min_periods).mean().reset_index(level=0, drop=True)
            out = out.join(rolled.add_suffix(f"_w{w}"))
        out["n_played"] = long.groupby("team").cumcount() + 1
        return TeamSeries(out)
