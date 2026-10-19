import pandas as pd

_COMPARE_COLUMNS = ["scenario", "jaccard_vs_reference", "n_pathways", "robust"]


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def compare_sets(sets, reference, threshold=0.7):
    if reference not in sets:
        raise KeyError(f"reference {reference} not in sets")
    ref = sets[reference]
    rows = []
    for name, s in sets.items():
        if name == reference:
            continue
        score = jaccard(ref, s)
        rows.append({
            "scenario": name,
            "jaccard_vs_reference": round(score, 3),
            "n_pathways": len(s),
            "robust": bool(score >= threshold),
        })
    if not rows:
        return pd.DataFrame(columns=_COMPARE_COLUMNS)
    return pd.DataFrame(rows).sort_values("jaccard_vs_reference", ascending=False).reset_index(drop=True)


def split_robust_fragile(sets, reference, threshold=0.7):
    if reference not in sets:
        raise KeyError(f"reference {reference} not in sets")
    ref = sets[reference]
    robust = set()
    fragile = set()
    for name, s in sets.items():
        if name == reference:
            continue
        if jaccard(ref, s) >= threshold:
            robust |= s
        else:
            fragile |= s - ref
    return sorted(robust), sorted(fragile)
