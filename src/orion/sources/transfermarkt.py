"""Transfermarkt (dcaribou/transfermarkt-datasets, DuckDB): games, line-ups, cards, managers, player values.

Top divisions only (see leagues.py). The upstream pipeline is paused: games stop at 2026-07-06, valuations at
2026-06-12, so features built on it are missing for later matches (models fall back on the other features).
Team names are mapped to football-data names by orion.mapping, from the game results.

Every query values a player with his latest valuation dated strictly before the match day (ASOF JOIN).
"Regulars" of a club for a game = players who started at least `min_starts` of the club's previous `window`
league games (never the game itself).
"""

from pathlib import Path

import duckdb
import pandas as pd

from orion import leagues
from orion.config import Config
from orion.http import download
from orion.paths import RAW
from orion.schema import RESULTS
from orion.sources.base import SOURCES, Source

INJURIES = RAW / "salimt-football-datasets/repo/datalake/transfermarkt/player_injuries/player_injuries_updated_20260929.csv"

_CLUB_GAMES = """
club_games as (
    select game_id, date, club_id,
           row_number() over (partition by club_id order by date, game_id) as k
    from (select game_id, date, home_club_id as club_id from games
          where competition_id in (select unnest($comps)) and date >= $since
          union all
          select game_id, date, away_club_id from games
          where competition_id in (select unnest($comps)) and date >= $since)
),
starts as (
    select cg.club_id, cg.k, l.player_id
    from club_games cg join game_lineups l on l.game_id = cg.game_id and l.club_id = cg.club_id
    where l.type = 'starting_lineup'
),
regulars as (
    select cg.game_id, cg.club_id, cg.date, s.player_id
    from club_games cg join starts s on s.club_id = cg.club_id and s.k between cg.k - $window and cg.k - 1
    group by all having count(*) >= $min_starts
)
"""

_GAMES_SQL = """
select game_id, competition_id, date, home_club_id, away_club_id, home_club_name as home, away_club_name as away,
       home_club_goals as home_goals, away_club_goals as away_goals,
       home_club_manager_name as home_manager, away_club_manager_name as away_manager
from games where competition_id in (select unnest($comps)) and date >= $since
"""

_XI_SQL = """
with g as (
    select game_id, date from games where competition_id in (select unnest($comps)) and date >= $since
),
starters as (
    select l.game_id, l.club_id, l.player_id, g.date
    from game_lineups l join g using (game_id) where l.type = 'starting_lineup'
),
valued as (
    select s.game_id, s.club_id, v.market_value_in_eur as value
    from starters s asof left join player_valuations v on s.player_id = v.player_id and s.date > v.date
)
select game_id, club_id, count(value) as n_valued, sum(value) / 1e6 as value_m,
       avg(ln(greatest(value, 1))) as log_value
from valued group by all
"""

# Regulars injured on match day: injury started at least 2 days before (so it was public) and not yet over.
_INJURED_SQL = "with " + _CLUB_GAMES + """,
injuries as (select player_id, from_date, end_date from read_csv($injuries)),
out as (
    select distinct r.game_id, r.club_id, r.date, r.player_id
    from regulars r join injuries i on i.player_id = r.player_id
     and i.from_date <= r.date - interval 2 day and i.end_date >= r.date
),
regular_value as (
    select r.game_id, r.club_id, count(*) as n_regulars, sum(coalesce(v.market_value_in_eur, 0)) / 1e6 as value_m
    from regulars r asof left join player_valuations v on r.player_id = v.player_id and r.date > v.date
    group by all
),
out_value as (
    select o.game_id, o.club_id, count(*) as n_out, sum(coalesce(v.market_value_in_eur, 0)) / 1e6 as out_m
    from out o asof left join player_valuations v on o.player_id = v.player_id and o.date > v.date
    group by all
)
select rv.game_id, rv.club_id, rv.n_regulars, rv.value_m as regulars_m,
       coalesce(ov.n_out, 0) as n_out, coalesce(ov.out_m, 0) as out_m
from regular_value rv left join out_value ov using (game_id, club_id)
"""

# Suspended for a game: sent off (red or second yellow) in the club's previous league game, or reached the 5th,
# 10th or 15th booking of the season there. An approximation: real rules vary by league and cups also count.
_SUSPENDED_SQL = "with " + _CLUB_GAMES + """,
offences as (
    select e.game_id, e.club_id, e.player_id
    from game_events e
    where e.type = 'Cards' and (regexp_matches(e.description, '^(\\d+\\. )?(Red card|Second yellow)')
                                or regexp_matches(e.description, '^(5|10|15)\\. Yellow card'))
),
banned as (
    select distinct cur.game_id, cur.club_id, cur.date, o.player_id
    from club_games cur
    join club_games prev on prev.club_id = cur.club_id and prev.k = cur.k - 1
    join offences o on o.game_id = prev.game_id and o.club_id = cur.club_id
),
banned_value as (
    select b.game_id, b.club_id, count(*) as n_banned, sum(coalesce(v.market_value_in_eur, 0)) / 1e6 as banned_m,
           count(r.player_id) as n_regulars_banned
    from banned b
    asof left join player_valuations v on b.player_id = v.player_id and b.date > v.date
    left join regulars r on r.game_id = b.game_id and r.club_id = b.club_id and r.player_id = b.player_id
    group by all
)
select cg.game_id, cg.club_id, coalesce(bv.n_banned, 0) as n_banned, coalesce(bv.banned_m, 0) as banned_m,
       coalesce(bv.n_regulars_banned, 0) as n_regulars_banned
from club_games cg left join banned_value bv using (game_id, club_id)
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

    def _query(self, cfg: Config, sql: str, **params) -> pd.DataFrame:
        comps = [lg.transfermarkt for lg in map(leagues.get, cfg.leagues) if lg.transfermarkt]
        since = f"{2000 + int(cfg.seasons[0][:2])}-07-01"
        with duckdb.connect(str(self.db), read_only=True) as con:
            df = con.execute(sql, {"comps": comps, "since": since, **params}).df()
        # games.game_id is text upstream while line-up/event tables use integers: normalise every id.
        for col in ("game_id", "club_id", "home_club_id", "away_club_id"):
            if col in df.columns:
                df[col] = df[col].astype("int64")
        return df

    def load(self, cfg: Config) -> pd.DataFrame:
        """One row per covered game, in Transfermarkt team names, with football-data division codes."""
        to_div = {lg.transfermarkt: lg.code for lg in map(leagues.get, cfg.leagues) if lg.transfermarkt}
        df = self._query(cfg, _GAMES_SQL)
        df.insert(0, "div", df.pop("competition_id").map(to_div))
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_results(self, cfg: Config) -> pd.DataFrame:
        return RESULTS.validate(self.load(cfg).dropna(subset=["home_goals", "away_goals"]))

    def xi_values(self, cfg: Config) -> pd.DataFrame:
        """Per (game_id, club_id): valued starters, total XI value (EUR m), mean log value."""
        return self._query(cfg, _XI_SQL)

    def injured_regulars(self, cfg: Config, window: int, min_starts: int) -> pd.DataFrame:
        """Per (game_id, club_id): regulars, their value, and how many of them (and how much value) are injured."""
        return self._query(cfg, _INJURED_SQL, injuries=str(INJURIES), window=window, min_starts=min_starts)

    def suspended(self, cfg: Config, window: int, min_starts: int) -> pd.DataFrame:
        """Per (game_id, club_id): players banned after the previous league game, their value, regulars among them."""
        return self._query(cfg, _SUSPENDED_SQL, window=window, min_starts=min_starts)
