# Methods

## Scope

The project evaluates how missing-omics patterns affect pathway-level findings in NSCLC, and whether a reproducible framework can separate robust signals from availability-dependent ones.

## Availability mask

Each sample is represented by a binary vector over omics types, built from the actual presence of each modality in the input matrices.

## Scenarios

Scenarios are defined by the subset of modalities required for a sample to be included. Only samples with all required modalities are used in a given scenario. The `full` scenario is therefore the subset of samples with all required modalities, not the full cohort.

## Pathway comparison

Pathway sets discovered in each scenario are compared to the full-data reference using the Jaccard index. Pathways above a fixed threshold are labelled robust; the remainder are labelled fragile.

In Phase 1, this logic is exercised on synthetic identifiers. No biological pathway is identified.

## Feature stability

A Random Forest classifier is trained on gene expression within each scenario. Feature importance is used only as a comparative signal across scenarios, not as evidence of biological mechanism. Genes ranked in the top-N in every scenario are labelled stable.

In Phase 1, the model uses expression only. Scenarios differ in sample selection, not in feature modality. The Phase 1 model comparison therefore does not measure the independent effect of adding or removing an omics modality.

## Reproducibility

Random seeds are fixed. Package versions are pinned in `pyproject.toml`. CI runs the unit tests on every push touching `src/`, `tests/`, or `pyproject.toml`.
