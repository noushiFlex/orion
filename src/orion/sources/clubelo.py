"""ClubElo ratings via the xgabora/Club-Football-Match-Data mirror (the ClubElo API itself returns 502).

Snapshots on the 1st and 15th of each month since 2000. Values up to 2025-06-01 are ClubElo's; later ones are
the mirror author's provisional continuation. The mirror's Matches.csv is used only to learn team-name mappings.
"""

from pathlib import Path

import pandas as pd

from orion.config import Config
from orion.http import download
from orion.schema import RATINGS, RESULTS
from orion.sources.base import SOURCES, Source, require_csv_header

PROVISIONAL_FROM = pd.Timestamp("2025-06-15")


@SOURCES.register
class ClubElo(Source):
    name = "clubelo"
    folder = "xgabora-club-football"
    BASE = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data/main/data/{file}"
    FILES = {"EloRatings.csv": b"date", "Matches.csv": b"Division"}

    def fetch(self, cfg: Config, refresh: bool = False) -> list[Path]:
        paths = []
        for file, marker in self.FILES.items():
            dest = self.raw_dir / file
            if refresh or not dest.exists():
                paths.append(download(self.BASE.format(file=file), dest, self.name, validate=require_csv_header(marker)))
        return paths

    def load(self, cfg: Config) -> pd.DataFrame:
        return RATINGS.validate(pd.read_csv(self.raw_dir / "EloRatings.csv"))

    def load_results(self, cfg: Config) -> pd.DataFrame:
        cols = {"Division": "div", "MatchDate": "date", "HomeTeam": "home", "AwayTeam": "away",
                "FTHome": "home_goals", "FTAway": "away_goals"}
        df = pd.read_csv(self.raw_dir / "Matches.csv", usecols=list(cols), low_memory=False).rename(columns=cols)
        df = df[df["div"].isin(cfg.leagues)].dropna(subset=["home_goals", "away_goals"])
        return RESULTS.validate(df)
