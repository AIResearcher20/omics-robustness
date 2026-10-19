import numpy as np
import pandas as pd
import pytest
from src.ml_sensitivity import (
    train_rank,
    crossval_score,
    compare_importance,
    stable_features,
)


def make_data(seed=0, n=80, p=20):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, p)),
        index=[f"s{i}" for i in range(n)],
        columns=[f"g{i}" for i in range(p)],
    )
    signal = (X["g0"] + X["g1"]).to_numpy()
    y = pd.Series((signal > np.median(signal)).astype(int), index=X.index)
    return X, y


def test_train_rank_sorted():
    X, y = make_data()
    imp = train_rank(X, y, seed=1)
    assert imp.is_monotonic_decreasing
    assert imp.index[0] in {"g0", "g1"}


def test_index_mismatch_raises():
    X, y = make_data()
    y_shifted = y.copy()
    y_shifted.index = [f"x{i}" for i in range(len(y_shifted))]
    with pytest.raises(ValueError):
        train_rank(X, y_shifted)


def test_missing_values_raise():
    X, y = make_data()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        train_rank(X, y)


def test_non_numeric_raises():
    X, y = make_data()
    X["g0"] = X["g0"].astype(str)
    with pytest.raises(ValueError):
        train_rank(X, y)


def test_crossval_returns_dict():
    X, y = make_data()
    out = crossval_score(X, y, folds=4)
    assert set(out) == {"mean", "std", "folds", "n_samples", "n_features"}
    assert 0.0 <= out["mean"] <= 1.0
    assert out["folds"] == 4


def test_crossval_rejects_small_classes():
    X, _ = make_data(n=10, p=5)
    y = pd.Series([0] * 9 + [1], index=X.index)
    with pytest.raises(ValueError):
        crossval_score(X, y, folds=5)


def test_compare_importance_counts():
    imp_a = pd.Series([0.5, 0.3, 0.2], index=["g0", "g1", "g2"])
    imp_b = pd.Series([0.4, 0.4, 0.2], index=["g1", "g0", "g2"])
    df = compare_importance({"a": imp_a, "b": imp_b}, top_n=2)
    assert set(df["gene"]) == {"g0", "g1"}
    assert df.loc[df["gene"] == "g0", "n_scenarios"].iloc[0] == 2


def test_compare_importance_empty_returns_columns():
    df = compare_importance({}, top_n=10)
    assert df.empty
    assert list(df.columns) == ["gene", "n_scenarios", "scenarios"]


def test_stable_features_requires_min():
    imp_a = pd.Series([0.5, 0.3, 0.2], index=["g0", "g1", "g2"])
    imp_b = pd.Series([0.4, 0.4, 0.2], index=["g1", "g0", "g2"])
    stable = stable_features({"a": imp_a, "b": imp_b}, top_n=2, min_scenarios=2)
    assert set(stable) == {"g0", "g1"}


def test_stable_features_rejects_bad_min():
    imp = pd.Series([0.5, 0.5], index=["g0", "g1"])
    with pytest.raises(ValueError):
        stable_features({"a": imp}, top_n=1, min_scenarios=2)


def test_top_n_validation():
    imp = pd.Series([0.5, 0.5], index=["g0", "g1"])
    with pytest.raises(ValueError):
        compare_importance({"a": imp}, top_n=0)
