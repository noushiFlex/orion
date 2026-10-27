import numpy as np
import pandas as pd
import pytest

from orion.schema import MATCHES


def make_matches(rounds: int = 30, teams=("A", "B", "C", "D", "E", "F"), seed: int = 0) -> pd.DataFrame:
    """Synthetic league: every team plays once a week against a random opponent."""
    rng = np.random.default_rng(seed)
    rows, day = [], pd.Timestamp("2020-08-01")
    for _ in range(rounds):
        order = rng.permutation(teams)
        for h, a in zip(order[::2], order[1::2]):
            row = {"div": "E0", "season": "2021", "date": day, "home": h, "away": a,
                   "home_goals": rng.poisson(1.5), "away_goals": rng.poisson(1.1)}
            row["referee"] = rng.choice(["R1", "R2", "R3"])
            for stat, lam in (("shots", 12), ("sot", 4), ("corners", 5), ("fouls", 11), ("yellow", 2), ("red", 0.1)):
                row[f"home_{stat}"], row[f"away_{stat}"] = rng.poisson(lam), rng.poisson(lam)
            row["home_xg"], row["away_xg"] = rng.gamma(2, 0.7), rng.gamma(2, 0.6)
            for o, base in (("home", 2.2), ("draw", 3.4), ("away", 3.3)):
                avg = base * rng.uniform(0.7, 1.3)
                row[f"odds_pre_{o}"], row[f"odds_max_{o}"], row[f"odds_b365_{o}"] = avg, avg * 1.05, avg * 0.98
            rows.append(row)
        day += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    df["home_id"], df["away_id"] = "ENG/" + df["home"], "ENG/" + df["away"]
    df["match_id"] = df["div"] + ":" + df["date"].dt.strftime("%Y-%m-%d") + ":" + df["home"] + ":" + df["away"]
    return MATCHES.validate(df)


@pytest.fixture
def matches() -> pd.DataFrame:
    return make_matches()
