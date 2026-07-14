# Analysis module

This package validates whether Smina scores rank supplied DYRK1A actives ahead
of supplied DUDE-Z decoys. It validates a docking **ranking harness**; it does
not turn property-matched decoys into experimentally confirmed non-binders.

## Input contract

Both score files must contain unique, non-empty `molecule_id` values and a
`score_kcal_mol` column. The default direction is `lower_is_better`, matching
Smina affinity scores. Duplicate IDs, active/decoy overlap, infinite scores,
non-numeric text, and active-reference mismatches stop the run.

Blank scores are expected for some harness failures but require a declared
policy:

- `error` (default): stop and force a choice;
- `exclude`: evaluate ranking conditional on successful docking;
- `rank_last`: retain failed rows at a tied rank below every observed score,
  evaluating the end-to-end workflow.

The run log always retains class-specific input counts and docking coverage.

## Ranking choices

- ROC AUC measures global active/decoy ordering.
- EF at 1%, 5%, and 10% measures early recovery at explicit screening budgets.
- BEDROC with alpha 20 supplies a continuous early-recognition metric.
- Mann-Whitney U is a two-sided distributional test and is not treated as a
  performance magnitude. A directional probability and rank-biserial effect
  size are reported beside it.

Equal docking scores are one threshold, not permission to use CSV order as a
scientific tie-breaker. EF and BEDROC therefore average over all positions in a
tied score block. With completely tied scores, AUC is 0.5 and EF is 1.0 no
matter whether active or decoy rows appeared first.

Class-stratified percentile bootstrap intervals use a fixed recorded seed.
They quantify compound-level sampling variability conditional on this curated
benchmark. They do not include repeated-docking uncertainty, analogue-series
dependence, or uncertainty caused by how DUDE-Z selected decoys.

## Run

```bash
python -m src.analysis.run_analysis \
  --scores data/actives_scores.csv \
  --decoy-scores data/decoys_scores.csv \
  --active-reference data/reference/dyrk1a_actives_chembl.csv \
  --missing-policy error \
  --name harness_validation \
  --outdir results
```

Every run writes input SHA-256 hashes, Git commit and dirty state, exact metric
parameters, software versions, machine-readable JSON, static figures, and a
self-contained HTML report. Harness settings are marked as an analysis-time
snapshot because a current config file cannot prove how an older score CSV was
generated.
