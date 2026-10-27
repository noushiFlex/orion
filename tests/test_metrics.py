import math

import numpy as np
import pytest

from orion.evaluation.metrics import brier, log_loss, rps
from orion.models.base import combine, fit_weights

Y = np.array([0, 1, 2])


def test_perfect_forecast_scores_zero():
    p = np.eye(3)
    assert log_loss(Y, p) == pytest.approx(0, abs=1e-12)
    assert brier(Y, p) == 0
    assert rps(Y, p) == 0


def test_uniform_forecast():
    p = np.full((3, 3), 1 / 3)
    assert log_loss(Y, p) == pytest.approx(math.log(3))
    assert brier(Y, p) == pytest.approx(2 / 3)


def test_rps_punishes_distant_misses_more():
    home_win = np.array([0])
    assert rps(home_win, np.array([[0, 1, 0]])) < rps(home_win, np.array([[0, 0, 1]]))


def test_stacking_prefers_the_informative_member():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    good = np.full((len(y), 3), 0.15)
    good[np.arange(len(y)), y] = 0.7
    noise = rng.dirichlet([1, 1, 1], len(y))
    w = fit_weights({"good": good, "noise": noise}, y)
    assert w["good"] > 0.9 and sum(w.values()) == pytest.approx(1, abs=1e-3)


def test_combine_renormalises():
    a, b = np.array([[0.5, 0.3, 0.2]]), np.array([[0.2, 0.3, 0.5]])
    p = combine({"a": a, "b": b}, {"a": 3, "b": 1})
    np.testing.assert_allclose(p, [[0.425, 0.3, 0.275]])
