"""football-data.co.uk "new" leagues (ARG, BRA, USA, ...): one multi-season file per country.

Results and closing odds only: no shots, cards, referee or pre-closing odds. Calendar-year seasons ("2016") are
filed under the season starting that year ("1617"), so a walk-forward split by season start never trains on a
match played after the test season began.
"""

import logging
from pathlib import Path

import pandas as pd

from orion import leagues, seasons
from orion.config import Config
from orion.http import download
from orion.schema import MATCHES
from orion.sources.base import SOURCES, Source, require_csv_header

log = logging.getLogger(__name__)

RENAME = {"Date": "date", "Time": "time", "Home": "home", "Away": "away", "HG": "home_goals", "AG": "away_goals",
          "PSCH": "odds_close_home", "PSCD": "odds_close_draw", "PSCA": "odds_close_away"}


@SOURCES.register
class FootballDataExtra(Source):
    name = "football_data_extra"
    folder = "football-data-co-uk-extra"
    URL = "https://football-data.co.uk/new/{code}.csv"

    def _codes(self, cfg: Config) -> list[str]:
        return [c for c in cfg.leagues if leagues.get(c).source == self.name]

    def fetch(self, cfg: Config, refresh: bool = False) -> list[Path]:
        return [download(self.URL.format(code=c), self.raw_dir / f"{c}.csv", self.name,
                         validate=require_csv_header(b"Country"))
                for c in self._codes(cfg) if refresh or not (self.raw_dir / f"{c}.csv").exists()]

    def load(self, cfg: Config) -> pd.DataFrame:
        frames = []
        for code in self._codes(cfg):
            path = self.raw_dir / f"{code}.csv"
            if not path.exists():
                log.warning("missing %s (run `orion fetch football_data_extra`)", path.name)
                continue
            df = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in RENAME or c == "Season", low_memory=False)
            df["season"] = df.pop("Season").astype(str).map(lambda s: seasons.code(int(s[:4])))
            frames.append(df.rename(columns=RENAME).assign(div=code))
        if not frames:
            return MATCHES.validate(pd.DataFrame(columns=["match_id"]))
        df = pd.concat(frames, ignore_index=True)
        df = df[df["season"].isin(cfg.seasons)]
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed", errors="coerce")
        df = df.dropna(subset=["date", "home", "away"])
        for side in ("home", "away"):
            df[side] = df[side].astype(str).str.strip()
            df[f"{side}_id"] = df["div"] + "/" + df[side]
        df["match_id"] = df["div"] + ":" + df["date"].dt.strftime("%Y-%m-%d") + ":" + df["home"] + ":" + df["away"]
        df = df.drop_duplicates("match_id")
        return MATCHES.validate(df).sort_values(["date", "match_id"]).reset_index(drop=True)
