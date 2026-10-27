import pandas as pd

from orion.mapping import apply, build_name_map

from conftest import make_matches


def test_renamed_teams_are_recovered_despite_date_shift():
    reference = make_matches()
    renames = {"A": "Alpha FC", "B": "Beta United"}
    source = reference.copy()
    source[["home", "away"]] = source[["home", "away"]].replace(renames)
    source["date"] = source["date"] + pd.Timedelta(days=1)

    m = build_name_map(source, reference)
    accepted = m[m["accepted"]].set_index("source_name")["ref_name"]
    assert accepted["Alpha FC"] == "A" and accepted["Beta United"] == "B" and accepted["C"] == "C"
    assert len(apply(source, m, "test")) == len(source)
