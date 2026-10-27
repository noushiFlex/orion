import numpy as np
import pytest

from orion.dataset import build_features
from orion.features import Context
from orion.models import MODELS

from conftest import make_matches


@pytest.mark.parametrize("name", ["poisson", "gbm"])
def test_models_return_valid_probabilities(name):
    ds = build_features(Context(None, make_matches(rounds=60)), {"form": {"windows": [5]}, "elo": {}})
    played = ds.played
    model = MODELS.get(name)(ds.feature_cols, seed=0).fit(played)
    p = model.predict_proba(played)
    assert p.shape == (len(played), 3)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-9)
    assert (p >= 0).all()
