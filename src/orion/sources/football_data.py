"""football-data.co.uk: results, match stats, referee and closing odds. The reference source for team names."""

import logging
import time
from pathlib import Path

import pandas as pd

from orion import leagues, progress
from orion.config import Config
from orion.http import download
from orion.schema import MATCHES
from orion.sources.base import SOURCES, Source, require_csv_header

log = logging.getLogger(__name__)

RENAME = {
    "Div": "div", "Date": "date", "Time": "time", "HomeTeam": "home", "AwayTeam": "away", "Referee": "referee",
    "FTHG": "home_goals", "FTAG": "away_goals", "HS": "home_shots", "AS": "away_shots",
    "HST": "home_sot", "AST": "away_sot", "HC": "home_corners", "AC": "away_corners",
    "HF": "home_fouls", "AF": "away_fouls", "HY": "home_yellow", "AY": "away_yellow",
    "HR": "home_red", "AR": "away_red",
    "PSCH": "odds_close_home", "PSCD": "odds_close_draw", "PSCA": "odds_close_away",  # Pinnacle closing
    "AvgH": "odds_pre_home", "AvgD": "odds_pre_draw", "AvgA": "odds_pre_away",  # market average, pre-closing
}
# Before 2019/20 the pre-closing market average was published as BbAv*; used when Avg* is absent.
LEGACY = {"BbAvH": "odds_pre_home", "BbAvD": "odds_pre_draw", "BbAvA": "odds_pre_away"}


@SOURCES.register
class FootballData(Source):
    name = "football_data"
    folder = "football-data-co-uk"
    SEASON_URL = "https://football-data.co.uk/mmz4281/{season}/{div}.csv"
    FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"

    def fetch(self, cfg: Config, refresh: bool = False) -> list[Path]:
        check = require_csv_header(b"Div")
        todo = [(self.SEASON_URL.format(season=season, div=div), self.raw_dir / f"{div}_{season}.csv")
                for div in cfg.leagues for season in cfg.seasons
                if refresh or season == cfg.current_season or not (self.raw_dir / f"{div}_{season}.csv").exists()]
        todo.append((self.FIXTURES_URL, self.raw_dir / "fixtures.csv"))
        paths = []
        with progress.task("football-data", total=len(todo)) as bar:
            for url, dest in todo:
                bar.advance(0, label=dest.name)
                paths.append(download(url, dest, self.name, validate=check))
                bar.advance()
                time.sleep(1)
        return paths

    def load(self, cfg: Config) -> pd.DataFrame:
        frames = []
        for div in cfg.leagues:
            for season in cfg.seasons:
                path = self.raw_dir / f"{div}_{season}.csv"
                if not path.exists():
                    log.warning("missing %s (run `orion fetch football_data`)", path.name)
                    continue
                frames.append(_read(path).assign(season=season))
        fixtures = self.raw_dir / "fixtures.csv"
        if fixtures.exists():  # scheduled matches; listed after played ones so duplicates keep the result
            fx = _read(fixtures)
            fx = fx[fx["div"].isin(cfg.leagues)].drop(columns=["home_goals", "away_goals"], errors="ignore")
            frames.append(fx.assign(season=cfg.current_season))
        df = pd.concat(frames, ignore_index=True)

        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed", errors="coerce")
        df = df.dropna(subset=["date", "home", "away"])
        for side in ("home", "away"):
            df[side] = df[side].astype(str).str.strip()
        country = df["div"].map(lambda c: leagues.get(c).country)
        df["home_id"] = country + "/" + df["home"]
        df["away_id"] = country + "/" + df["away"]
        df["match_id"] = df["div"] + ":" + df["date"].dt.strftime("%Y-%m-%d") + ":" + df["home"] + ":" + df["away"]
        df = df.drop_duplicates("match_id", keep="first")
        return MATCHES.validate(df).sort_values(["date", "match_id"]).reset_index(drop=True)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip",
                     usecols=lambda c: c in RENAME or c in LEGACY, low_memory=False)
    legacy = {old: new for old, new in LEGACY.items() if old in df.columns and new not in df.columns.map(RENAME.get)}
    df = df.drop(columns=[c for c in LEGACY if c in df.columns and c not in legacy])
    return df.rename(columns={**RENAME, **legacy})
