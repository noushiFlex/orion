"""Players suspended after the club's previous league game (red card, second yellow, 5th/10th/15th booking).

Per side: n_banned, banned_m (market value, EUR m), n_regulars_banned. Approximate: league-specific rules and
cup bookings are ignored. Cards come from Transfermarkt events (frozen at 2026-07-06).
"""

from orion.features.base import FEATURES, Context, Feature, MatchLevel
from orion.features.tm import linked_games, per_side
from orion.sources.transfermarkt import Transfermarkt


@FEATURES.register
class Suspensions(Feature):
    name = "suspensions"

    def __init__(self, window: int = 10, min_starts: int = 5):
        self.window, self.min_starts = window, min_starts

    def compute(self, ctx: Context) -> MatchLevel:
        banned = Transfermarkt().suspended(ctx.cfg, self.window, self.min_starts)
        return MatchLevel(per_side(linked_games(ctx), banned, ["n_banned", "banned_m", "n_regulars_banned"]))
