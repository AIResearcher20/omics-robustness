import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _validate_inputs(expression, labels, folds, n_estimators, top_n=1):
    if not isinstance(expression, pd.DataFrame):
        raise TypeError("expression must be a DataFrame")
    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a Series")
    if expression.shape[0] != labels.shape[0]:
        raise ValueError("expression and labels must have the same number of rows")
    if not expression.index.equals(labels.index):
        raise ValueError("expression and labels indices must match in the same order")
    if expression.shape[0] == 0 or expression.shape[1] == 0:
        raise ValueError("expression is empty")
    if expression.columns.duplicated().any():
        raise ValueError("expression has duplicated column names")
    if not all(np.issubdtype(t, np.number) for t in expression.dtypes):
        raise ValueError("expression must be numeric")
    values = expression.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("expression contains non-finite values")
    if labels.isna().any():
        raise ValueError("labels contain missing values")
    if labels.nunique() < 2:
        raise ValueError("labels must contain at least two classes")
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if n_estimators < 1:
        raise ValueError("n_estimators must be >= 1")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    counts = labels.value_counts()
    if counts.min() < folds:
        raise ValueError(
            f"smallest class has {counts.min()} samples, fewer than folds={folds}"
        )


def train_rank(expression, labels, seed=42, n_estimators=200, top_n=1):
    _validate_inputs(expression, labels, folds=2, n_estimators=n_estimators, top_n=top_n)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(expression, labels)
    importance = pd.Series(model.feature_importances_, index=expression.columns)
    return importance.sort_values(ascending=False)


def crossval_score(expression, labels, seed=42, folds=5, n_estimators=200):
    _validate_inputs(expression, labels, folds=folds, n_estimators=n_estimators, top_n=1)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = "roc_auc" if labels.nunique() == 2 else "roc_auc_ovr_weighted"
    scores = cross_val_score(model, expression, labels, cv=cv, scoring=scoring)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "folds": folds,
        "n_samples": int(expression.shape[0]),
        "n_features": int(expression.shape[1]),
    }


def compare_importance(importance_dict, top_n=100):
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if not importance_dict:
        return pd.DataFrame(columns=["gene", "n_scenarios", "scenarios"])
    top_sets = {name: set(imp.head(top_n).index) for name, imp in importance_dict.items()}
    all_genes = set().union(*top_sets.values())
    rows = []
    for gene in sorted(all_genes):
        present = sorted(name for name, s in top_sets.items() if gene in s)
        rows.append({
            "gene": gene,
            "n_scenarios": len(present),
            "scenarios": ",".join(present),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["n_scenarios", "gene"], ascending=[False, True]).reset_index(drop=True)


def stable_features(importance_dict, top_n=100, min_scenarios=None):
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if not importance_dict:
        return []
    n = len(importance_dict)
    if min_scenarios is None:
        min_scenarios = n
    if not isinstance(min_scenarios, int) or min_scenarios < 1 or min_scenarios > n:
        raise ValueError(f"min_scenarios must be between 1 and {n}")
    df = compare_importance(importance_dict, top_n=top_n)
    return df[df["n_scenarios"] >= min_scenarios]["gene"].tolist()
