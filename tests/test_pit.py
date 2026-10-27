import pandas as pd

from orion import pit
from orion.features import TeamSeries


def _matches(dates):
    return pd.DataFrame({"match_id": [f"m{i}" for i in range(len(dates))], "date": pd.to_datetime(dates),
                         "home_id": "ENG/A", "away_id": "ENG/B"})


def test_same_day_and_future_values_are_excluded():
    series = TeamSeries(pd.DataFrame({"team": ["ENG/A"] * 3, "available_at": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-20"]),
                                      "v": [1.0, 2.0, 3.0]}))
    out = pit.join(_matches(["2024-01-10", "2024-01-11", "2023-12-31"]), series, "x")
    assert out["home_x_v"].tolist()[:2] == [1.0, 2.0]  # 01-10 value is not visible on 01-10
    assert pd.isna(out["home_x_v"].iloc[2])  # nothing before the first value
    assert out["away_x_v"].isna().all()  # other team untouched


def test_stale_values_are_dropped():
    series = TeamSeries(pd.DataFrame({"team": ["ENG/A"], "available_at": pd.to_datetime(["2024-01-01"]), "v": [1.0]}),
                        max_age=pd.Timedelta(days=30))
    out = pit.join(_matches(["2024-01-15", "2024-03-01"]), series, "x")
    assert out["home_x_v"].iloc[0] == 1.0
    assert pd.isna(out["home_x_v"].iloc[1])
