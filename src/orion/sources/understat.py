"""Understat: per-match expected goals for the Big Five leagues. Raw JSON is stored unmodified.

Unofficial endpoint (https://understat.com/getLeagueData/<league>/<start year>); it can change without notice.
Team names differ from football-data.co.uk and are mapped by orion.mapping.
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd

from orion import leagues, progress, seasons
from orion.config import Config
from orion.http import download
from orion.schema import XG
from orion.sources.base import SOURCES, Source

log = logging.getLogger(__name__)


def _check(body: bytes) -> None:
    if "dates" not in json.loads(body):
        raise ValueError("Understat response has no 'dates' key")


@SOURCES.register
class Understat(Source):
    name = "understat"
    folder = "understat"
    URL = "https://understat.com/getLeagueData/{league}/{year}"

    def _covered(self, cfg: Config):
        for div in cfg.leagues:
            key = leagues.get(div).understat
            if key:
                for season in cfg.seasons:
                    yield div, key, season, self.raw_dir / f"{div}_{season}.json"

    def fetch(self, cfg: Config, refresh: bool = False) -> list[Path]:
        todo = [(self.URL.format(league=key, year=seasons.start_year(season)), dest)
                for div, key, season, dest in self._covered(cfg)
                if refresh or season == cfg.current_season or not dest.exists()]
        paths = []
        with progress.task("understat", total=len(todo)) as bar:
            for url, dest in todo:
                bar.advance(0, label=dest.name)
                paths.append(download(url, dest, self.name, headers={"X-Requested-With": "XMLHttpRequest"},
                                      validate=_check))
                bar.advance()
                time.sleep(1.5)
        return paths

    def load(self, cfg: Config) -> pd.DataFrame:
        rows = []
        for div, _, season, path in self._covered(cfg):
            if not path.exists():
                log.warning("missing %s (run `orion fetch understat`)", path.name)
                continue
            for m in json.loads(path.read_text())["dates"]:
                if m["isResult"]:
                    rows.append({
                        "div": div, "date": m["datetime"][:10], "home": m["h"]["title"], "away": m["a"]["title"],
                        "home_goals": m["goals"]["h"], "away_goals": m["goals"]["a"],
                        "home_xg": m["xG"]["h"], "away_xg": m["xG"]["a"],
                    })
        return XG.validate(pd.DataFrame(rows))
