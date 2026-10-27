"""The core guarantee: a match's features depend only on matches played on earlier days."""

import pandas as pd
import pytest

from orion.dataset import build_features
from orion.features import Context

FEATURE_SETS = [
    {"form": {"windows": [3], "min_periods": 1}},
    {"elo": {}},
    {"rest": {}},
]


@pytest.mark.parametrize("features", FEATURE_SETS, ids=lambda f: next(iter(f)))
def test_features_ignore_the_future(matches, features):
    full = build_features(Context(None, matches), features)
    cutoff = matches["date"].iloc[len(matches) // 2]
    past = build_features(Context(None, matches[matches["date"] <= cutoff]), features)
    cols = past.feature_cols
    expected = full.frame.set_index("match_id").loc[past.frame["match_id"], cols]
    pd.testing.assert_frame_equal(past.frame.set_index("match_id")[cols], expected)


@pytest.mark.parametrize("features", FEATURE_SETS, ids=lambda f: next(iter(f)))
def test_features_ignore_own_result(matches, features):
    target = matches.index[len(matches) // 2]
    changed = matches.copy()
    changed.loc[target, ["home_goals", "away_goals", "home_shots", "home_xg"]] = [9, 0, 40, 5.0]
    before = build_features(Context(None, matches), features)
    after = build_features(Context(None, changed), features)
    mid = matches.loc[target, "match_id"]
    row = lambda ds: ds.frame.set_index("match_id").loc[mid, ds.feature_cols]
    pd.testing.assert_series_equal(row(before), row(after))


def test_scheduled_matches_get_features(matches):
    fixture = matches.iloc[[0]].copy()
    fixture["date"] = matches["date"].max() + pd.Timedelta(days=7)
    fixture["match_id"] = "E0:future"
    fixture[["home_goals", "away_goals"]] = float("nan")
    ds = build_features(Context(None, pd.concat([matches, fixture], ignore_index=True)), {"elo": {}, "rest": {}})
    row = ds.frame.set_index("match_id").loc["E0:future"]
    assert row[ds.feature_cols].notna().all()
    assert pd.isna(row["outcome"])
