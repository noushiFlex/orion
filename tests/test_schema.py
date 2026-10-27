import pandas as pd
import pytest

from orion.schema import RESULTS, SchemaError


def _results(**overrides):
    row = {"div": "E0", "date": "2024-01-01", "home": "A", "away": "B", "home_goals": 1, "away_goals": 0}
    return pd.DataFrame([{**row, **overrides}])


def test_valid_frame_passes():
    assert len(RESULTS.validate(_results())) == 1


def test_missing_required_column_fails():
    with pytest.raises(SchemaError, match="missing column 'home'"):
        RESULTS.validate(_results().drop(columns="home"))


def test_unparseable_value_fails_loudly():
    with pytest.raises(SchemaError, match="unparseable"):
        RESULTS.validate(_results(home_goals="two"))


def test_negative_goals_fail():
    with pytest.raises(SchemaError, match="below"):
        RESULTS.validate(_results(away_goals=-1))
