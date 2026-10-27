"""Transfermarkt SQL (injured regulars, suspensions) on a tiny synthetic DuckDB."""

from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from orion.sources import transfermarkt as tm

DATES = pd.date_range("2020-08-01", periods=9, freq="7D")  # games 1..9, one a week


@pytest.fixture
def source(tmp_path, monkeypatch):
    db = tmp_path / "tm.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""create table games (game_id varchar, competition_id varchar, date date, home_club_id int,
                   away_club_id int, home_club_name varchar, away_club_name varchar, home_club_goals int,
                   away_club_goals int, home_club_manager_name varchar, away_club_manager_name varchar)""")
    con.execute("create table game_lineups (game_id int, club_id int, player_id int, type varchar)")
    con.execute("create table player_valuations (player_id int, date date, market_value_in_eur double)")
    con.execute("""create table game_events (game_id int, club_id int, player_id int, type varchar,
                   description varchar)""")
    for i, d in enumerate(DATES, start=1):
        con.execute("insert into games values (?, 'GB1', ?, 1, 2, 'A', 'B', 1, 0, 'M1', 'M2')", [str(i), d.date()])
        con.execute("insert into game_lineups values (?, 1, 10, 'starting_lineup')", [i])  # player 10: regular
    con.execute("insert into game_lineups values (1, 1, 11, 'starting_lineup')")  # player 11: started once
    con.execute("insert into player_valuations values (10, '2020-01-01', 10e6), (10, ?, 20e6)", [DATES[6].date()])
    con.execute("insert into game_events values (5, 1, 10, 'Cards', 'Red card  , Serious foul')")
    con.execute("insert into game_events values (2, 1, 11, 'Cards', '5. Yellow card  , Foul')")
    con.execute("insert into game_events values (3, 1, 10, 'Cards', '4. Yellow card  , Foul')")
    con.close()
    injuries = tmp_path / "injuries.csv"
    pd.DataFrame({"player_id": [10, 10], "from_date": ["2020-09-01", "2020-09-25"],
                  "end_date": ["2020-09-20", "2020-10-30"]}).to_csv(injuries, index=False)
    monkeypatch.setattr(tm.Transfermarkt, "db", property(lambda self: db))
    monkeypatch.setattr(tm, "INJURIES", injuries)
    return tm.Transfermarkt(), SimpleNamespace(leagues=("E0",), seasons=("2021",))


def _by_game(df):
    return df[df["club_id"] == 1].set_index("game_id")


def test_injured_regulars(source):
    src, cfg = source
    inj = _by_game(src.injured_regulars(cfg, window=3, min_starts=2))
    assert inj.loc[5, "n_out"] == 0  # injury starts 2020-09-01, game 5 is 2020-08-29
    assert inj.loc[6, "n_out"] == 1 and inj.loc[6, "out_m"] == 10  # 2020-09-05
    assert inj.loc[7, "out_m"] == 10  # valuation dated on match day (2020-09-12) is not visible yet
    assert inj.loc[8, "out_m"] == 20  # 2020-09-19
    assert inj.loc[9, "n_out"] == 0  # 2020-09-26: second injury started 1 day before, not public yet
    assert 1 not in inj.index or inj.loc[1, "n_regulars"] == 0  # no history before game 1


def test_suspensions(source):
    src, cfg = source
    ban = _by_game(src.suspended(cfg, window=3, min_starts=2))
    assert ban.loc[6, "n_banned"] == 1 and ban.loc[6, "n_regulars_banned"] == 1 and ban.loc[6, "banned_m"] == 10
    assert ban.loc[3, "n_banned"] == 1 and ban.loc[3, "n_regulars_banned"] == 0  # 5th booking, not a regular
    assert ban.loc[4, "n_banned"] == 0  # a 4th booking is not a ban
    assert ban.loc[5, "n_banned"] == 0
