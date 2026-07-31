# Benchmark

Reproducible DYRK1A generative-model benchmarking, docking, and analysis.

Current workflow decisions and commands:

- [Decoy, HPC, and candidate-analysis protocol](docs/DECOY_HPC_AND_CANDIDATE_PROTOCOL.md)

## Running a labelled benchmark locally

This is the laptop path: build a cohort, then prepare + dock + analyse it in one
process. It is for anything that fits on 4 CPU cores in a sitting. The chunking
scripts in `src/harness/chunks.py` are for cluster scheduling and are not needed
here — a local run calls the exact same preparation, docking, and analysis
modules, so results do not drift between the two paths.

### The template

One line. Edit the three numbers/names in it, paste the whole thing.

```bash
python -m src.harness.build_cohort results/RUNNAME --n-actives 10 --decoys-per-active 30 --seed 42 && python -m src.harness.run_local results/RUNNAME --name RUNNAME --workers 4
```

To adapt it, change three things and nothing else:

- `RUNNAME` — in all three places. Any name you like, no spaces.
- `--n-actives 10` — how many actives.
- `--decoys-per-active 30` — how many decoys per active.

The `&&` means the docking only starts if the cohort built cleanly, so a typo in
the first half cannot leave you docking a half-built cohort.

Prerequisites, once per terminal:

```bash
conda activate dyrk1a-bench
cd ~/projects/Benchmark
```

Two things that will bite if you improvise on this:

- **Do not wrap the values in `< >`.** Bash reads those as file redirection and
  the command dies with a syntax error before Python ever starts.
- **Do not split it into a variable block plus commands** unless you paste the
  whole block in one go. Shell variables vanish when you open a new terminal,
  and an empty `$VAR` silently expands to nothing — argparse then reports
  `expected one argument`, which looks like a broken script but is an empty
  variable.

### What each knob does

**`build_cohort`** — decides *which molecules* are in the run.

| Flag | What it sets |
| --- | --- |
| `results/RUNNAME` | The cohort directory to create. Name it for what the run is, e.g. `smoke_5x50`, `pilot_50x50`. |
| `--n-actives` | **The number of actives in the run.** Sampled from the 1,219 accepted DYRK1A actives. |
| `--decoys-per-active` | **The number of decoys carried over per active.** Total decoys = actives × this. Omit the flag to take all 50 assigned decoys per active. |
| `--seed 42` | The RNG seed for the active sample. Same seed + same active count = the same molecules, every time. Change it only to deliberately get a *different* subset (e.g. to check a result is not an artefact of one draw). |
| `--first` | Take the actives first-by-sorted-ChEMBL-ID instead of sampling. Reproducible but **not representative** — ChEMBL order tracks deposition date, so it clusters chemical series. Do not use it for anything you will report. |

Sizing the run: with `A` actives and `D` decoys per active, total molecules
docked = `A + (A × D)`. Docking is roughly 15–40 s per ligand per core at
`EXHAUSTIVENESS = 8`, so with 4 workers estimate `(A + A×D) × ~25 s / 4`.
A 5 × 10 smoke test (55 molecules) is a few minutes;
5 × 50 (255 molecules) is well under an hour; 50 × 50 (2,550 molecules) is an
overnight job and is about the ceiling for this laptop.

Decoys are **never re-sampled** here. `select_decoys.py` property-matched each
decoy to a specific parent active once; `build_cohort` only carries that pairing
over, best-ranked matches first. Picking decoys independently of the actives
would break the matching and quietly turn a matched benchmark into an unmatched
one — which inflates enrichment.

**`run_local`** — decides *how the run executes and is scored*.

| Flag | What it sets |
| --- | --- |
| `results/RUNNAME` | The cohort directory from step 1. Must already contain `actives.smi` and `decoys.smi`. |
| `--name` | The run identifier used in the results directory and `run_history.csv`. Defaults to the cohort directory name. |
| `--workers 4` | Parallel Smina jobs. 4 = one per physical core on the submarine. Do not raise it: each job already gets `SMINA_CPU = 1`, and oversubscribing slows the whole run. |
| `--analysis-outdir` | Where the report is written. Defaults to `results/`. |
| `--missing-policy` | How molecules that failed to dock are handled: `error` (default — stop and make you look at it), `exclude`, or `rank_last`. Keep `error` unless you have already inspected the failures and decided what they are. |
| `--chemistry` | Adds chemical profiling from the intake tables. Requires `--active-intake` and `--decoy-intake`. **Only meaningful for a full cohort** — intake statistics describe every accepted molecule, not the docked subset, so on a 5-active smoke test the profile does not describe what you actually docked. |

Docking parameters themselves (box centre/size, exhaustiveness, seed, num_modes,
energy range, timeout) are **not** command-line flags. They live in
`src/harness/config.py` and are identical for every cohort and every generation
by design — changing them per run would make runs incomparable. If you change
one, it changes for everything and previous results must be re-run.

### Outputs

```
results/RUNNAME/
    actives.smi, decoys.smi     # the cohort inputs
    cohort.json                 # what was selected, seed, input hashes
    ligands/{actives,decoys}/   # prepared 3D PDBQT
    docking/{actives,decoys}/   # poses + scores.csv
    run_local.json              # timing, hardware, input/output hashes
results/RUNNAME_<timestamp>/
    report.html, metrics.json, figures/, interactive/
results/run_history.csv         # one appended row per run
```

Both JSON records hash their inputs and outputs, so any run can be traced back
to the exact molecule set and code path that produced it.

### Worked examples

Each is one line — paste the whole line.

```bash
# Smoke test: 5 actives, 10 decoys each = 50 decoys, 55 molecules total
python -m src.harness.build_cohort results/smoke_5x50 --n-actives 5 --decoys-per-active 10 --seed 42 && python -m src.harness.run_local results/smoke_5x50 --name smoke_5x50 --workers 4

# Fuller smoke: 5 actives, all 50 assigned decoys each = 250 decoys, 255 total
python -m src.harness.build_cohort results/smoke_5x250 --n-actives 5 --seed 42 && python -m src.harness.run_local results/smoke_5x250 --name smoke_5x250 --workers 4

# Pilot enrichment: 50 actives, 20 decoys each = 1,000 decoys (overnight)
python -m src.harness.build_cohort results/pilot_50x20 --n-actives 50 --decoys-per-active 20 --seed 42 && python -m src.harness.run_local results/pilot_50x20 --name pilot_50x20 --workers 4
```

A note on reading small runs: ROC-AUC and enrichment factors from a 5-active
cohort are not evidence about the target. They confirm the pipeline runs end to
end and the report renders. Enrichment numbers only become worth quoting once
the active set is large enough for the confidence interval to be narrower than
the effect you are claiming.
