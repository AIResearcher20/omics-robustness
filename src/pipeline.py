import numpy as np
import pandas as pd

from src.availability import build_mask, select_samples, scenario_table
from src.pathways import compare_sets, split_robust_fragile
from src.ml_sensitivity import train_rank, crossval_score, compare_importance, stable_features
from src.report import write_json, write_text


def synthetic_omics(n_samples=120, n_genes=200, n_meth=100, seed=42):
    if n_samples < 20 or n_genes < 10 or n_meth < 5:
        raise ValueError("synthetic_omics: sizes too small")
    rng = np.random.default_rng(seed)
    samples = [f"S{i:03d}" for i in range(n_samples)]
    genes = [f"G{i:03d}" for i in range(n_genes)]
    probes = [f"M{i:03d}" for i in range(n_meth)]

    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    rng.shuffle(labels)

    expr = rng.normal(size=(n_samples, n_genes))
    expr[:, :10] += labels[:, None] * 1.5
    expression = pd.DataFrame(expr, index=samples, columns=genes)

    meth = rng.normal(size=(n_samples, n_meth))
    meth[:, :5] += labels[:, None] * 0.8
    methylation = pd.DataFrame(meth, index=samples, columns=probes)

    mutation = pd.DataFrame(
        rng.integers(0, 2, size=(n_samples, 20)),
        index=samples,
        columns=[f"MU{i}" for i in range(20)],
    )

    keep_meth = rng.choice(samples, size=int(n_samples * 0.7), replace=False)
    methylation = methylation.loc[sorted(keep_meth)]

    return {
        "expression": expression,
        "methylation": methylation,
        "mutation": mutation,
    }, pd.Series(labels, index=samples)


def synthetic_pathway_logic(importance_dict, top_n=50):
    sets = {}
    for name, imp in importance_dict.items():
        top = set(imp.head(top_n).index)
        sets[name] = {f"PW_{g}" for g in top if g.startswith("G")}
    return sets


def run():
    omics, labels = synthetic_omics()
    mask = build_mask(omics)

    scenarios = {
        "full": ["expression", "methylation", "mutation"],
        "expr_meth": ["expression", "methylation"],
        "expr_only": ["expression"],
    }

    table = scenario_table(mask, scenarios)

    reference_samples = select_samples(mask, scenarios["full"])
    reference_definition = {
        "name": "full",
        "required": scenarios["full"],
        "n_samples": int(len(reference_samples)),
        "note": "reference is the subset of samples with all required modalities, not the full cohort",
    }

    importance_dict = {}
    model_scores = []

    for name, required in scenarios.items():
        samples = select_samples(mask, required)
        if len(samples) < 20:
            model_scores.append({
                "scenario": name,
                "n_samples": int(len(samples)),
                "skipped": True,
                "reason": "fewer than 20 samples",
            })
            continue
        expr = omics["expression"].loc[samples]
        y = labels.loc[samples]
        try:
            imp = train_rank(expr, y)
            score = crossval_score(expr, y)
        except ValueError as exc:
            model_scores.append({
                "scenario": name,
                "n_samples": int(len(samples)),
                "skipped": True,
                "reason": str(exc),
            })
            continue
        importance_dict[name] = imp
        model_scores.append({
            "scenario": name,
            "n_samples": int(len(samples)),
            "n_features": score["n_features"],
            "auc_mean": round(score["mean"], 3),
            "auc_std": round(score["std"], 3),
            "folds": score["folds"],
            "skipped": False,
        })

    logic_sets = synthetic_pathway_logic(importance_dict, top_n=50)
    if "full" in logic_sets and len(logic_sets) > 1:
        logic_comparison = compare_sets(logic_sets, reference="full", threshold=0.7)
        logic_robust, logic_fragile = split_robust_fragile(logic_sets, reference="full", threshold=0.7)
    else:
        logic_comparison = pd.DataFrame()
        logic_robust, logic_fragile = [], []

    feature_stability = compare_importance(importance_dict, top_n=50)
    stable = stable_features(importance_dict, top_n=50)

    write_json("results/scenario_table.json", table.to_dict(orient="records"))
    write_json("results/reference_definition.json", reference_definition)
    write_json("results/model_scores.json", model_scores)
    write_json("results/feature_stability.json", feature_stability.to_dict(orient="records"))
    write_json("results/stable_features.json", stable)
    write_json("results/synthetic_pathway_logic_comparison.json", logic_comparison.to_dict(orient="records"))
    write_json("results/synthetic_pathway_logic_robust.json", logic_robust)
    write_json("results/synthetic_pathway_logic_fragile.json", logic_fragile)

    write_text("results/README.txt", [
        "Phase 1 outputs.",
        "All pathway identifiers are synthetic (PW_Gxxx) and are used only to test set-comparison logic.",
        "No biological pathway has been identified in this phase.",
        "Model scores are reported for pipeline validation only.",
        "The model uses expression only; scenarios differ in sample selection, not in feature modality.",
        "The 'full' scenario is the subset of samples with all required modalities, not the full cohort.",
    ])


if __name__ == "__main__":
    run()
