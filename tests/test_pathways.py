import pandas as pd
import pytest
from src.pathways import jaccard, compare_sets, split_robust_fragile


def test_jaccard():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert round(jaccard({"a", "b"}, {"a"}), 3) == 0.5


def test_jaccard_empty_sets():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0


def test_compare_sets():
    sets = {
        "full": {"p1", "p2", "p3"},
        "expr_only": {"p1", "p2"},
        "meth_only": {"p4"},
    }
    df = compare_sets(sets, reference="full", threshold=0.6)
    assert set(df["scenario"]) == {"expr_only", "meth_only"}
    assert bool(df.loc[df["scenario"] == "expr_only", "robust"].iloc[0]) is True
    assert bool(df.loc[df["scenario"] == "meth_only", "robust"].iloc[0]) is False


def test_compare_sets_missing_reference():
    with pytest.raises(KeyError):
        compare_sets({"a": {"x"}}, reference="full")


def test_compare_sets_single_scenario_returns_empty_with_columns():
    df = compare_sets({"full": {"p1"}}, reference="full")
    assert df.empty
    assert list(df.columns) == ["scenario", "jaccard_vs_reference", "n_pathways", "robust"]


def test_split():
    sets = {
        "full": {"p1", "p2", "p3"},
        "expr_only": {"p1", "p2"},
        "meth_only": {"p4"},
    }
    robust, fragile = split_robust_fragile(sets, reference="full", threshold=0.6)
    assert "p1" in robust
    assert "p4" in fragile
