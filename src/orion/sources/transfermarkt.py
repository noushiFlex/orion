"""Transfermarkt (dcaribou/transfermarkt-datasets, DuckDB): starting line-ups and player market values.

Top divisions only (see leagues.py). The upstream pipeline is paused: games stop at 2026-07-06, valuations at
2026-06-12, so line-up features are missing for later matches (models fall back on the other features).
Team names are mapped to football-data names by orion.mapping, from the game results.
"""

from pathlib import Path

import duckdb
import pandas as pd

from orion import leagues
from orion.config import Config
from orion.http import download
from orion.schema import RESULTS
from orion.sources.base import SOURCES, Source

# Value of each starter = his latest valuation dated strictly before the match day (ASOF JOIN, no look-ahead).
_XI_SQL = """
with g as (
    select game_id, competition_id, date, home_club_id, away_club_id, home_club_name, away_club_name,
           home_club_goals, away_club_goals
    from games where competition_id in (select unnest($comps)) and date >= $since
),
starters as (
    select l.game_id, l.club_id, l.player_id, g.date
    from game_lineups l join g using (game_id) where l.type = 'starting_lineup'
),
valued as (
    select s.game_id, s.club_id, v.market_value_in_eur as value
    from starters s asof left join player_valuations v on s.player_id = v.player_id and s.date > v.date
),
xi as (
    select game_id, club_id, count(*) as n_starters, count(value) as n_valued,
           sum(value) / 1e6 as value_m, avg(ln(greatest(value, 1))) as log_value
    from valued group by all
)
select g.competition_id, g.date, g.home_club_name as home, g.away_club_name as away,
       g.home_club_goals as home_goals, g.away_club_goals as away_goals,
       h.n_valued as home_n_valued, h.value_m as home_value_m, h.log_value as home_log_value,
       a.n_valued as away_n_valued, a.value_m as away_value_m, a.log_value as away_log_value
from g
left join xi h on h.game_id = g.game_id and h.club_id = g.home_club_id
left join xi a on a.game_id = g.game_id and a.club_id = g.away_club_id
"""


@SOURCES.register
class Transfermarkt(Source):
    name = "transfermarkt"
    folder = "transfermarkt"
    URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb"

    @property
    def db(self) -> Path:
        return self.raw_dir / "transfermarkt-datasets.duckdb"

    def fetch(self, cfg: Config, refresh: bool = False) -> list[Path]:
        if self.db.exists() and not refresh:
            return []
        return [download(self.URL, self.db, self.name)]

    def load(self, cfg: Config) -> pd.DataFrame:
        """One row per covered game: results plus starting-XI value for each side, in Transfermarkt names."""
        comps = {lg.transfermarkt: lg.code for lg in map(leagues.get, cfg.leagues) if lg.transfermarkt}
        if not comps:
            return pd.DataFrame()
        with duckdb.connect(str(self.db), read_only=True) as con:
            df = con.execute(_XI_SQL, {"comps": list(comps), "since": f"{2000 + int(cfg.seasons[0][:2])}-07-01"}).df()
        df.insert(0, "div", df.pop("competition_id").map(comps))
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_results(self, cfg: Config) -> pd.DataFrame:
        return RESULTS.validate(self.load(cfg))
