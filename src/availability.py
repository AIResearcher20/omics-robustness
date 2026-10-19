import pandas as pd


def build_mask(omics):
    if not omics:
        raise ValueError("omics must contain at least one modality")
    samples = None
    for name, df in omics.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"omics[{name}] must be a DataFrame")
        if df.index.has_duplicates:
            raise ValueError(f"omics[{name}] has duplicated sample index")
        samples = df.index if samples is None else samples.union(df.index)
    mask = pd.DataFrame(0, index=samples, columns=list(omics.keys()))
    for name, df in omics.items():
        mask.loc[df.index, name] = 1
    return mask


def select_samples(mask, required):
    if not required:
        raise ValueError("required must contain at least one modality")
    missing = [r for r in required if r not in mask.columns]
    if missing:
        raise KeyError(f"required modalities missing from mask: {missing}")
    return mask.index[mask[required].all(axis=1)]


def scenario_table(mask, scenarios):
    if not scenarios:
        raise ValueError("scenarios must contain at least one entry")
    rows = []
    for name, required in scenarios.items():
        samples = select_samples(mask, required)
        rows.append({
            "scenario": name,
            "required": ",".join(required),
            "n_required": len(required),
            "n_samples": len(samples),
        })
    return pd.DataFrame(rows).sort_values("n_samples", ascending=False).reset_index(drop=True)


def reference_mask(mask, reference_required):
    return select_samples(mask, reference_required)
