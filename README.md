# omics-robustness
# nsclc-missing-omics-sensitivity

Sensitivity analysis of pathway discovery to realistic missing-omics patterns in lung adenocarcinoma and lung squamous cell carcinoma.

## Research question

To what extent do cohort-dependent patterns of missing omics data alter the biological pathways discovered in LUAD and LUSC, and can a reproducible sensitivity-analysis framework distinguish robust pathway-level signals from findings that depend on assay availability and sample selection?

## Status

Phase 1: pipeline skeleton, synthetic data, unit tests, CI.

Phase 1 does not identify biological pathways. All pathway identifiers in Phase 1 outputs are synthetic and exist only to test set-comparison logic. Model scores in Phase 1 are reported for pipeline validation only.

## Data

Phase 1 uses synthetic data generated inside `src/pipeline.py`. No external data is downloaded.

Phase 2 will connect the pipeline to TCGA-LUAD and TCGA-LUSC open access data. Sources and checksums will be recorded in `docs/data.md`.

## Usage

    python -m pip install -e ".[dev]"
    pytest tests/ -v
    python -m src.pipeline

## Outputs

- `results/scenario_table.json`
- `results/reference_definition.json`
- `results/model_scores.json`
- `results/feature_stability.json`
- `results/stable_features.json`
- `results/synthetic_pathway_logic_comparison.json`
- `results/synthetic_pathway_logic_robust.json`
- `results/synthetic_pathway_logic_fragile.json`
- `results/README.txt`

## Reproducibility

- Package versions pinned in `pyproject.toml`.
- Random seed fixed at 42.
- CI runs on every push touching `src/`, `tests/`, `pyproject.toml`, or the workflow file.

## Limitations

- Phase 1 model uses expression only. Scenarios differ in sample selection, not in feature modality.
- The `full` scenario is the subset of samples with all required modalities, not the full cohort.
- Phase 1 results are not biological findings.
