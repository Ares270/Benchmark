# DYRK1A Generative Docking Benchmark

![status: campaigns complete](https://img.shields.io/badge/status-campaigns--complete-brightgreen)
![license: MIT](https://img.shields.io/badge/license-MIT-green)
![target: DYRK1A 7O7K](https://img.shields.io/badge/target-DYRK1A%207O7K-lightgrey)
![docking: Smina / Vina scoring](https://img.shields.io/badge/docking-Smina%20%2F%20Vina-lightgrey)

A reproducible docking-enrichment benchmark comparing three frozen generative
molecular models and a naive chemical-space control against DYRK1A kinase
through a shared Smina pipeline.

---

## Overview

This benchmark measures what a generative molecular model contributes to a
docking campaign once the measuring instrument is held fixed. Four arms submit
an identical raw molecule budget to one shared pipeline — molecule intake, a
pre-registered pre-docking gate, Meeko ligand preparation, and Smina docking
against a single hashed DYRK1A receptor — and are reported on generation
quality, pipeline survival, docking-score distribution, and evaluated-parent
chemistry. The arms are ordered by how much target information each model
receives at inference: a property-matched chemical-space baseline, then Gen1
(no target information), Gen2 (the DYRK1A protein sequence), and Gen3 (the
locked 7O7K pocket geometry).

Three commitments define the design. No model is trained or fine-tuned locally;
every arm runs a frozen, hash-authenticated public checkpoint. No model ever
sees a docking score, so the evaluation cannot be optimised against. And
failure is a reported result: invalid, duplicate, gate-failing,
preparation-failing, and docking-failing molecules are counted and published,
never replaced or topped up.

## Arms

| Arm | Model | Conditioning | What it isolates |
| --- | --- | --- | --- |
| Baseline | ChEMBL 37 property-matched draw | None | The chemical-space floor: what property-matched space alone scores |
| Gen1 | GuacaMol SMILES-LSTM | None (target-unaware) | A pure learned chemical prior with no target information |
| Gen2 | WarmMolGenOne | DYRK1A protein sequence | Sequence-conditioned generation |
| Gen3 | Molexar 10M Omni | Locked 7O7K pocket geometry | 3D-pocket-conditioned generation |

The baseline's primary mode is `property_matched`, locked before any model
score is inspected; a `uniform` mode exists as a sensitivity floor and never
substitutes for the primary control.

### Frozen model boundary

Each model arm is pinned to a specific published checkpoint, by repository
revision and by the SHA-256 of every weight and tokenizer artefact it loads.
The pins live in `configs/`, and `python -m src.benchmark doctor` verifies them
before a run starts.

| | Gen2 | Gen3 |
| --- | --- | --- |
| Repository | `gokceuludogan/WarmMolGenOne` | `fairydance/molexar-10m-omni` |
| Revision | `1cbaeb72ba926f070c8f10349e9ee7041083135a` | `39be4ae96102506f6852b7627b27df0ae1ac8186` |
| Architecture | Encoder–decoder: 12-layer protein RoBERTa encoder, 6-layer ChemBERTa decoder | `MolexarForCausalLM` with a GVP protein-pocket encoder, 16 layers |
| Model input | DYRK1A sequence (UniProt Q13627, residues 127–485, matching the 7O7K kinase-domain construct) | The 7O7K pocket atom graph at the locked docking centre |
| Sampling | `do_sample`, temperature 1.0, `top_k` 50, `top_p` 0.95 | `do_sample`, temperature 0.8, `top_k` 50, `top_p` 0.95 |
| License | MIT | MIT |

Nothing else reaches a model. No arm receives known actives, decoys, a
receptor grid, docking scores, property targets, a pharmacophore, gate
outcomes, or any post-generation optimisation signal. Every arm uses seed
`20260801`, draws 10,000 raw molecules, and passes a uniformly selected
1,000-row branch to intake with no validity filtering, deduplication,
replacement, or top-up inside the generator.

### Why this is a gradient, not an ablation

This is an information gradient, not an architecture-only ablation. The four
arms differ in architecture, training corpus, and conditioning simultaneously.
Any difference between them must be discussed as the joint result of those
differences. Relatedly, "no local training" does not mean "no learned target
biology": each model's pretraining corpus is described in its specification,
and where an exact target-disjointness audit is not possible, no disjointness
is claimed.

## Repository structure

```
src/        Pipeline source: molecule intake, docking harness, arm runners, analysis
data/       Receptor, pocket definition, reference actives, and cohort inputs
configs/    Frozen per-arm model specifications (checkpoint revision, hashes, sampling)
docs/       Pre-registration records, protocol specifications, and runbooks
tests/      Unit and integration tests for every pipeline stage
cloud/      Google Batch container definition for distributed docking chunks
results/    Run output root; each completed run writes an immutable directory here
```

Only `results/run_history.csv`, an append-only ledger of attempted runs, is
version-controlled; run directories themselves are local artefacts.

## Environment setup

```bash
conda env create -f environment.yml
conda activate dyrk1a-bench
```

Gen3 additionally requires an isolated Python 3.13 environment, because Molexar
pins PyTorch 2.9.1 and Transformers 5.8.0 while the validated docking stack is
older. The two environments are deliberately not merged; the Gen3 runner
crosses the boundary automatically and authenticates its inputs and outputs on
each side. Per-arm checkpoint revisions, weight hashes, and sampling parameters
are recorded in `configs/gen2_warmmolgenone.json` and
`configs/gen3_molexar.json`, and the Gen1 architecture and sampling contract in
[docs/GEN1_GUACAMOL_SMILES_LSTM_SPEC.md](docs/GEN1_GUACAMOL_SMILES_LSTM_SPEC.md).

## Model bundles and generated inputs

Five of the inputs `doctor` checks are deliberately **not** version-controlled:
three published model bundles, the isolated Gen3 runtime, and the naive
baseline cohort. They are large, redistributable from their own upstream
sources under their own licences, or regenerable from a recorded seed. **A
fresh clone therefore fails preflight until these are in place. That is
expected, not a broken checkout.**

Everything below lands under `Models & Miscellaneous/` or `data/generated/`,
both Git-ignored. No bundle is re-hosted here. Every revision and file hash is
pinned in `configs/` or the Gen1 spec, and `doctor` checks each artefact
against its recorded byte count and SHA-256 before a model is ever
deserialised — so a wrong revision or a truncated download fails loudly
instead of quietly changing a result.

### Gen1 — GuacaMol SMILES-LSTM checkpoint

Target path `Models & Miscellaneous/model_final_0.473.pt`, from
`BenevolentAI/guacamol_baselines` at pinned commit
`ae43219c89d5db134028336243f508606d81995e`, directory
`smiles_lstm_hc/pretrained_model` (MIT). Expected identity: 101,148,783 bytes,
SHA-256 `da1be33b519868ba3a0befabe5b018d247bfd65ccaae7d5fdac4c804f23580f3`.
The matching `model_final_0.473.json` config sits beside it in the same
upstream directory. Full provenance and the loader's authentication contract
are in
[docs/GEN1_GUACAMOL_SMILES_LSTM_SPEC.md](docs/GEN1_GUACAMOL_SMILES_LSTM_SPEC.md).

### Gen2 — WarmMolGenOne and its SMILES tokenizer

```bash
huggingface-cli download gokceuludogan/WarmMolGenOne \
  --revision 1cbaeb72ba926f070c8f10349e9ee7041083135a \
  --local-dir "Models & Miscellaneous"

huggingface-cli download seyonec/PubChem10M_SMILES_BPE_450k \
  --revision c18fccd09b3326bf2d4633412c256d7db872156d \
  --local-dir "Models & Miscellaneous/PubChem10M_SMILES_BPE_450k"
```

The encoder-decoder weights (`pytorch_model.bin`, 766,604,630 bytes) land in
the `Models & Miscellaneous/` root; the protein tokenizer ships with them and
the SMILES tokenizer comes from the second repository. Per-file hashes are in
`configs/gen2_warmmolgenone.json`.

### Gen3 — Molexar bundle and isolated runtime

```bash
huggingface-cli download fairydance/molexar-10m-omni \
  --revision 39be4ae96102506f6852b7627b27df0ae1ac8186 \
  --local-dir "Models & Miscellaneous/molexar-10m-omni"
```

Gen3 also needs its own interpreter at
`Models & Miscellaneous/molexar-env/bin/python`, because Molexar pins a newer
PyTorch and Transformers than the validated docking stack. Create it with
Python 3.13 and install the runtime recorded in `configs/gen3_molexar.json`
(torch 2.9.1+cpu, transformers 5.8.0, tokenizers 0.22.2, rdkit 2025.9.6,
numpy 2.4.4, scipy 1.17.1), then the two official sources at their pinned
revisions:

```bash
conda create -p "Models & Miscellaneous/molexar-env" python=3.13 -y
# install the pinned runtime from configs/gen3_molexar.json, then:
#   github.com/fairydance/Molexar          @ d874224a577822fe2c19ad99143d58125a369d11
#   github.com/fairydance/Fragment-SELFIES @ 09f00f8aa5420e2da89fc68bfc6793f4868e9653
```

The Gen3 runner crosses the environment boundary itself and authenticates its
inputs and outputs on each side.

### Naive baseline cohort

Regenerable; nothing needs downloading except the ChEMBL 37 source. Fetch the
bulk chemreps file recorded in
`data/external/chembl_37_chemreps/source_provenance.json` (ChEMBL 37,
DOI 10.6019/CHEMBL.database.37, SHA-256
`ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7`) into
`data/external/chembl_37_chemreps/`, then rebuild the registered cohort:

```bash
python -m src.generation.naive_baseline \
  --mode property_matched \
  --source-file data/external/chembl_37_chemreps/chembl_37_chemreps.txt.gz \
  --source-description "ChEMBL 37 bulk chemreps, full set, no filtering" \
  --actives data/reference/dyrk1a_actives_chembl.csv \
  --n 10000 \
  --seed 20260801 \
  --outdir data/generated/naive_property_matched_20260801
```

The seed and output path are the registered ones, so this reproduces the exact
cohort the campaign used. `src/utils/record_source_provenance.py` re-records
the source hash if you refetch.

## Preflight check

```bash
python -m src.benchmark doctor
```

This authenticates every input a run depends on before any compute is spent:
the Smina binary, the receptor and its locked docking box, the reference
actives, the cached naive baseline cohort, all three model bundles against
their recorded hashes, and the isolated Gen3 runtime. It generates nothing and
docks nothing. A successful check reports that all four registered arms can be
launched; it exits non-zero and names each missing or mismatched input
otherwise.

On a fresh clone the first five checks pass and the last five fail, because the
model bundles and the generated cohort are not version-controlled. See
[Model bundles and generated inputs](#model-bundles-and-generated-inputs) for
how to obtain each one. The receptor, the docking box, and the reference
actives are tracked, so the harness-validation path needs no downloads.

## Tests

```bash
python -m pytest tests
```

Scope the command to `tests`. A bare `pytest` from the repository root also
collects the vendored third-party suites under `Models & Miscellaneous/` once
the Gen3 sources are installed, and those fail on their own imports.

## Running the benchmark

Run names must be new on every attempt; the runners refuse to write into an
existing directory. Replace the upper-case placeholders below, and do not wrap
them in angle brackets — Bash reads those as redirection.

**The registered four-arm campaign.** This is the single command for the real
comparison. It exposes no scientific knobs: each arm uses seed `20260801`,
produces or draws its locked 10,000-molecule cohort, submits exactly 1,000 raw
rows to the shared pipeline, and the fairness-gated four-arm comparison is
written at the end. It stops at the first failure.

```bash
python -m src.benchmark run all results/REGISTERED_RUN_NAME --workers 4
```

**A pilot plumbing check for one arm.** Pilot runs are explicitly labelled as
such and are not registered results.

```bash
python -m src.benchmark run gen3 results/PILOT_RUN_NAME --pilot --raw-n 100 --dock-n 10 --workers 4
```

**The labelled validation cohort.** This builds a reproducible cohort of
accepted DYRK1A actives plus property-matched decoys, docks it, and writes the
labelled enrichment report that tests the harness itself.

```bash
python -m src.harness.build_cohort results/VALIDATION_RUN_NAME --n-actives 10 --decoys-per-active 500 --seed 42
python -m src.harness.run_local results/VALIDATION_RUN_NAME --name VALIDATION_RUN_NAME --workers 4 --missing-policy error
```

`--workers` is an operational setting, not a scientific one. The receptor, box,
exhaustiveness, docking seed, number of modes, energy range, and per-ligand
timeout live in `src/harness/config.py` as a single source of truth; changing
any of them makes earlier and later arms incomparable and requires rerunning
every arm in the comparison.

## Metrics

Metrics fall into three buckets, kept deliberately separate.

- **Harness validation** — computed only on the labelled active/decoy cohort:
  ROC AUC, BEDROC, enrichment factors at fixed list fractions, and bootstrap
  confidence intervals. These judge the instrument, not any generative model.
- **Generative quality** — per arm: SMILES validity, parent uniqueness,
  survival at each funnel stage (submitted, accepted at intake, passed the
  pre-docking gate, prepared, successfully scored), and measured compute cost.
- **Per-molecule quality** — docking-score distribution, ligand efficiency, and
  twelve separately reported chemical properties (molecular weight, cLogP,
  TPSA, hydrogen-bond donors and acceptors, rotatable bonds, ring and aromatic
  ring counts, fraction Csp³, formal charge, QED, and synthetic accessibility).

Ligand efficiency — docking score divided by heavy atom count — is computed and
reported alongside the raw score, never in place of it. Smina's scoring
function sums per-atom contributions, so a larger molecule scores better for
free; reporting both lets molecular size and docking performance be read side
by side. Ligand efficiency does not enter any ranking metric.

### The funnel

Survival is reported as absolute counts at six fixed points, so attrition is
visible rather than averaged away:

1. raw generated or source rows;
2. rows submitted to intake;
3. molecules accepted at intake;
4. molecules that passed the shared pre-docking gate;
5. prepared PDBQT files;
6. successfully scored molecules.

Each stage authenticates the previous stage's output against a recorded hash
and refuses to continue on a mismatch. A completed run directory preserves its
exact input and subsample, those hashes, per-stage summaries, the list of
failures with reasons, timings, scores, and the rendered report. The gate is
shared by every unlabelled arm and is never tuned per model.

### What is deliberately not computed

Candidate cohorts have no activity labels, so ROC AUC, BEDROC, and enrichment
are deliberately not computed for them. No composite score is produced and no
arm is declared a winner: the trade-offs between validity, uniqueness, funnel
survival, chemistry, docking distribution, and compute are left visible, and
interpreting them is the reader's job. A docking score is a computational
prioritisation signal; it is not evidence of biochemical activity.

## Design commitments

- **No local training or fine-tuning.** Every model arm runs a frozen public
  checkpoint pinned by repository revision and weight SHA-256.
- **No docking-score reward.** No model receives a score, a gate outcome, or
  any evaluation feedback, so the measurement cannot be optimised against.
- **The information gradient is the experiment.** Arms differ in conditioning
  by design; this is not presented as a controlled architecture-only ablation.
- **Failure is reported, never repaired.** Molecules lost at any funnel stage
  are counted and published. There is no top-up, no replacement, and no
  per-model gate tuning.
- **Pre-registered gate and thresholds.** Both were committed and tagged before
  any registered run (`predock-gate-v1`, `validation-threshold-v1`).
- **Hash-chained provenance at every stage.** Each stage authenticates its
  inputs against hashes recorded by the previous stage and refuses to proceed
  on a mismatch.
- **Immutable run directories.** Runs are assembled in a working directory and
  atomically renamed on completion; an existing name is never overwritten.

## Pre-registration

Two sets of constants are frozen in git history, tagged, and dated before any
registered run was executed. The generative-arm pre-docking gate constants are
frozen at commit `fd1574a`, tagged `predock-gate-v1` (2026-07-30). The harness
validation thresholds — ROC AUC ≥ 0.65 and EF₁% ≥ 5.00 on the labelled
active/decoy cohort — are frozen at commit `9de5567`, tagged
`validation-threshold-v1` (2026-08-30). The purpose of fixing both in advance
is that a threshold chosen after seeing the number is not a threshold but a
knob; because the tags predate every registered run, the bar cannot be moved to
accommodate a result. See
[docs/PREDOCK_GATE_AND_NAIVE_BASELINE_SPEC.md](docs/PREDOCK_GATE_AND_NAIVE_BASELINE_SPEC.md)
and
[docs/HARNESS_VALIDATION_THRESHOLD.md](docs/HARNESS_VALIDATION_THRESHOLD.md).

## Status

Both registered campaigns are complete and are reported in full in the preprint.

The harness validation ran against the pre-registered thresholds: ROC AUC 0.658
(95% CI [0.642, 0.673]) against a floor of 0.650, and EF₁% 3.12 (95% CI
[2.21, 4.10]) against a floor of 5.00. The harness cleared one criterion and
missed the other. The result is reported as registered; the thresholds were
not revised afterwards.

The registered four-arm comparison ran on 2026-11-14.

A second negative cohort of 384 compounds with measured weak or absent DYRK1A
potency was docked separately and is never blended with the property-matched
decoys.

Pilot and smoke runs existed only to prove plumbing; their numbers are not
results and are not cited anywhere.

## Citation

```bibtex
@misc{georgiou2026dyrk1a,
  author = {Georgiou, Ares},
  title  = {DYRK1A Generative Docking Benchmark},
  year   = {2026},
  note   = {Preprint in preparation}
}
```

## License

MIT. See [LICENSE](LICENSE).
