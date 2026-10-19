import pandas as pd
import pytest
from src.availability import build_mask, select_samples, scenario_table


def test_mask_shape_and_values():
    omics = {
        "expression": pd.DataFrame({"g1": [1.0, 2.0]}, index=["s1", "s2"]),
        "methylation": pd.DataFrame({"m1": [0.1]}, index=["s1"]),
    }
    mask = build_mask(omics)
    assert list(mask.columns) == ["expression", "methylation"]
    assert mask.loc["s1", "methylation"] == 1
    assert mask.loc["s2", "methylation"] == 0


def test_build_mask_rejects_empty():
    with pytest.raises(ValueError):
        build_mask({})


def test_build_mask_rejects_duplicated_index():
    omics = {"expression": pd.DataFrame({"g1": [1.0, 2.0]}, index=["s1", "s1"])}
    with pytest.raises(ValueError):
        build_mask(omics)


def test_select_samples():
    mask = pd.DataFrame(
        {"expression": [1, 1, 1], "methylation": [1, 0, 1]},
        index=["s1", "s2", "s3"],
    )
    assert list(select_samples(mask, ["expression"])) == ["s1", "s2", "s3"]
    assert list(select_samples(mask, ["expression", "methylation"])) == ["s1", "s3"]


def test_select_samples_rejects_unknown():
    mask = pd.DataFrame({"expression": [1]}, index=["s1"])
    with pytest.raises(KeyError):
        select_samples(mask, ["methylation"])


def test_scenario_table_columns():
    mask = pd.DataFrame(
        {"expression": [1, 1], "methylation": [1, 0]},
        index=["s1", "s2"],
    )
    table = scenario_table(mask, {"full": ["expression", "methylation"], "expr_only": ["expression"]})
    assert set(table["scenario"]) == {"full", "expr_only"}
    assert set(table.columns) == {"scenario", "required", "n_required", "n_samples"}
