# Pre-Dock Gate & Naive Baseline — Implementation Specification

**Project:** DYRK1A Generative Benchmark
**Repo:** `~/projects/Benchmark/`
**Spec written:** 2026-07-30
**Status:** DECISIONS LOCKED. Implementation not started.
**Audience:** the implementing agent (Claude Code) — and the author, who must be able
to defend every constant in this document without reading the code.

---

## 0. Read this before writing a line

This document specifies **two** deliverables:

| Part | Module | What it is |
| --- | --- | --- |
| **A** | `src/generation/filter.py` + `filter_config.py` | The pre-dock gate. Every molecule from every arm passes through it. |
| **B** | `src/generation/naive_baseline.py` | The naive baseline generator (two variants). |

The gate is the more important of the two. It is the single point where a fairness
mistake contaminates all four arms simultaneously and silently.

**Two rules govern every decision below. If a proposed change violates either, reject it.**

> **Rule 1 — You cannot filter on a metric you intend to report.**
> QED, SA score, Lipinski compliance, PAINS flags, molecular weight, TPSA and every
> other reported property are **results**, not gates. Filtering on a reported property
> truncates the distribution you are describing.

> **Rule 2 — The gate is justified by the measurement instrument, not by chemical taste.**
> The only admissible justification for excluding a molecule is *"outside this range,
> the number Smina prints is an artifact of the box and the search, not an estimate of
> binding."* "It doesn't look like a drug" is never a justification.

Corollary that resolves most arguments: **a molecule that could be docked but was
excluded for aesthetic reasons is a fairness bug.** A molecule that produces a
meaningless number is correctly excluded.

---

## 1. The four tiers of molecule loss

The gate handles exactly one of these. Confusing the tiers is the primary failure mode.

| Tier | Name | Cause | Owner | Judgment involved |
| --- | --- | --- | --- | --- |
| **0** | Intake failure | RDKit cannot parse/sanitize; duplicate parent | `src/harness/intake.py` (exists) | none |
| **1** | **Instrument range** | Outside Smina's valid operating range | **`src/generation/filter.py` (BUILD THIS)** | minimal, derived |
| **2** | Preparation failure | No 3D embed; Meeko cannot write PDBQT | `src/harness/prepare_ligands.py` (exists) | none |
| **3** | Docking failure | Smina crash, timeout, no score | `src/harness/dock.py` (exists) | none |

Tiers 0, 2, 3 are **failures** — the molecule could not be processed. Tier 1 is the
only **decision**, and it is deliberately tiny.

---

## 2. Why a high garbage rate must be reported, not filtered away

The intuition to resist: *"a fair filter should protect models from being judged on
their garbage."* It should not.

A model that emits 99% unusable molecules and 1% excellent ones is a **different and
worse tool** than one that emits 100% mediocre molecules, even if their docking-score
distributions on survivors look identical. The literature is full of docking scores
reported on hand-picked top-N outputs with the survival rate omitted. That omission is
the thing this benchmark exists to correct.

Therefore:

- The gate does **not** protect models from their own failure rate.
- The **survival rate is a first-class reported metric**, in the generative-quality
  bucket, permanently separate from per-molecule docking scores.
- The two are **never** combined into a composite score. (Consistent with the existing
  repo-wide no-composite-scores principle.)

---

## 3. PART A — The pre-dock gate

### 3.1 Locked constants

Create `src/generation/filter_config.py` containing exactly this and nothing else:

```python
"""Pre-dock instrument-range gate. Constants derived 2026-07-30 from the 1,219
DYRK1A actives in data/reference/dyrk1a_actives_chembl.csv. See
docs/PREDOCK_GATE_AND_NAIVE_BASELINE_SPEC.md for the derivation.

PRE-REGISTERED: this file is committed and tagged before any arm generates a
molecule. Do not edit after the tag without a documented, dated amendment.
"""

FILTER_SCHEMA_VERSION = 1

# Elements with AutoDock/Vina atom-type parameters. Anything else either fails
# Meeko preparation or receives an unparameterized (meaningless) score.
ALLOWED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})

MIN_HEAVY_ATOMS = 10
MAX_HEAVY_ATOMS = 62
MAX_ROTATABLE_BONDS = 15
```

### 3.2 Derivation of every constant — the defense

Percentiles measured over all 1,219 actives (`[min, p1, p50, p99, max]`):

```
heavy_atoms  [12, 15, 25, 41, 46]
rot_bonds    [ 0,  0,  3,  9, 14]
```

Derivation rule, declared before the numbers were seen: **floor = p1 × 0.7,
ceiling = p99 × 1.5**, rounded outward to an integer.

| Constant | Arithmetic | Value | Headroom vs. observed actives |
| --- | --- | --- | --- |
| `MIN_HEAVY_ATOMS` | 15 × 0.7 = 10.5 → round down | **10** | 2 heavy atoms *below* the smallest known active (12) |
| `MAX_HEAVY_ATOMS` | 41 × 1.5 = 61.5 → round up | **62** | 16 heavy atoms *above* the largest known active (46) |
| `MAX_ROTATABLE_BONDS` | 9 × 1.5 = 13.5 → round up | **15** | 1 *above* the most flexible known active (14) |

**The one-sentence defense:** *No bound touches the chemical space of any known DYRK1A
active. The admitted range strictly contains the observed active range in every
dimension, with substantial headroom above.*

**Second, independent anchor for `MAX_ROTATABLE_BONDS`:** Smina/Vina search reliability
at `EXHAUSTIVENESS = 8` degrades as torsional degrees of freedom increase; Vina's PDBQT
format hard-caps at 32 torsions. A ~15-torsion ceiling sits well inside the regime where
the configured search effort still explores conformational space adequately. This bound
is therefore justified twice — by the actives *and* by the instrument.

**Note on the target's character (do not skip):** median 25 heavy atoms, median MW ~357 Da.
DYRK1A's known chemistry is dominated by small, flat, hinge-binding scaffolds (harmine,
indirubins, INDY). This is a **small-ligand target**. Diffusion models pretrained on
CrossDocked2020 — which contains many larger pockets — will routinely emit 50–60
heavy-atom molecules. The generous ceiling is deliberate and non-negotiable: a tight
ceiling would silently penalize gen3 for a training-distribution mismatch that has
nothing to do with molecular quality.

### 3.3 Gate implementation

`src/generation/filter.py`

**Public function:**

```python
def apply_gate(records, config_module) -> GateResult
```

- `records`: iterable of `(molecule_id, parent_smiles)` — parents, post-intake.
- Evaluate the three gates **independently** on every molecule. Do **not** short-circuit
  on first failure. A molecule that fails both size and torsion bounds must be recorded
  as failing both, so per-gate exclusion counts are honest.
- Compute properties on the **parent** structure with explicit RDKit calls:
  - heavy atoms → `mol.GetNumHeavyAtoms()`
  - rotatable bonds → `rdMolDescriptors.CalcNumRotatableBonds(mol)`
  - elements → iterate `mol.GetAtoms()`, collect `GetSymbol()`
- Hydrogens: evaluate elements on the molecule **with** explicit or implicit H permitted;
  `H` is in `ALLOWED_ELEMENTS` so both representations behave identically.

**Outputs, written to the arm's output directory:**

| File | Contents |
| --- | --- |
| `gate_decisions.csv` | one row **per submitted molecule**: `molecule_id`, `parent_smiles`, `heavy_atoms`, `rotatable_bonds`, `disallowed_elements` (semicolon-joined, blank if none), `pass_elements`, `pass_size`, `pass_torsions`, `passed` |
| `gate_pass.smi` | `SMILES<space>molecule_id`, survivors only, input order preserved |
| `gate_summary.json` | see below |

`gate_summary.json` must contain:

- `stage: "predock_gate"`, `schema_version`
- `filter_schema_version` and the **SHA-256 of `filter_config.py`**
- the literal constant values used (do not make the reader open the config)
- `n_submitted`, `n_passed`, `pass_fraction`
- `exclusions_by_gate`: `{elements: X, size_low: Y, size_high: Z, torsions: W}`
  — note these **may sum to more than** `n_submitted − n_passed` because a molecule can
  fail multiple gates; state that explicitly in the JSON as a `counting_note` string
- `disallowed_element_histogram`: `{"B": 3, "Si": 1, ...}`
- input/output file records via the existing `src/harness/runtime.py` helpers
  (`file_record`, `write_json_atomic`, `hardware_record`, `timing_record`)
- an `interpretation` block stating verbatim: *"This gate declares the range over which
  Smina scores are treated as measurements rather than artifacts. It is not a
  drug-likeness filter. Exclusion counts are a reported per-arm metric."*

**Constraints:**
- Follow existing repo conventions: invoked as `python -m src.generation.filter`, refuses
  to write into an existing output directory, atomic JSON writes, `allow_nan=False`.
- **Import constants from `filter_config.py`. Never inline a numeric literal.** A number
  appearing in two places is a number that will disagree with itself.
- The gate is **stateless and arm-agnostic.** It receives no arm name, no model identity,
  no flags that alter thresholds. There is no `--relaxed` option. If a caller could
  change the gate's behaviour per arm, the gate is not a fair gate.

### 3.4 Known weaknesses — state these in the docstring, do not hide them

An honest module documents its own soft spots. Write these into `filter.py`'s
module docstring:

1. **Rule 1 is bent, not obeyed, for size.** Heavy-atom count correlates strongly
   (ρ ≈ 0.95) with molecular weight, which *is* a reported property. Filtering on size
   therefore does truncate a reported distribution. Mitigation: the truncation points
   (10 and 62) lie entirely outside the observed active range (12–46), so within the
   region of scientific interest the reported distribution is unaffected. Exclusions are
   reported per gate.
2. **The bounds are mildly target-informed.** They are derived from DYRK1A actives, so a
   legitimately dockable 70-heavy-atom molecule is excluded for reasons unrelated to its
   quality. Mitigation: generous headroom, plus per-arm exclusion reporting so any arm
   losing a large fraction to the size gate is visible rather than buried.
3. **`CalcNumRotatableBonds` is a proxy, not the true torsion count.** Meeko's active
   torsion count in the final PDBQT can differ (amide handling, ring systems,
   `RIGID_MACROCYCLES = True`). The gate uses the RDKit definition because it is
   computable pre-preparation. Any residual mismatch surfaces as a Tier-2 preparation
   failure, which is already recorded.
4. **The torsion-degradation claim needs a literature anchor before publication.** The
   qualitative statement is uncontroversial; a citable quantitative source has not yet
   been recorded. Add it to the open-items list.

### 3.5 Explicitly forbidden — do not implement any of these

If a future prompt asks for one of these, refuse and point at this section.

- ❌ QED, SA score, or any drug-likeness threshold
- ❌ Lipinski / Veber / Rule-of-Three filters
- ❌ PAINS, Brenk, or other structural-alert filters
- ❌ Molecular weight, TPSA, logP, HBD, HBA, ring-count, or formal-charge bounds
- ❌ Any similarity or novelty threshold against the actives
- ❌ Top-N selection by *any* score prior to docking
- ❌ Per-arm threshold variation of any kind
- ❌ "Topping up" an arm after exclusions to restore a round number

The last one is worth its own sentence: `compare_candidates.py` already refuses to run
unless all arms share an **identical raw submitted-molecule budget**, and already
declines to replace invalid or duplicate molecules because their loss *counts as model
performance*. Topping up after the gate would violate a contract the repo already
enforces in code.

---

## 4. Pipeline position and the subsampling decision

```
  10,000 raw submissions per arm per seed
        │
        ├─────────────────────────────────────────────┐
        │                                             │
        ▼                                             ▼
  harness.intake (ALL 10,000)               random subsample → 1,000
  → validity, uniqueness, novelty,                    │
    QED, SA, scaffolds                                ▼
    = GENERATIVE-QUALITY bucket              harness.intake (the 1,000)
    (free metrics, no docking)                        │  Tier 0
                                                      ▼
                                            generation.filter  ← THE GATE (Tier 1)
                                                      │
                                                      ▼
                                            harness.prepare_ligands  Tier 2
                                                      │
                                                      ▼
                                            harness.dock             Tier 3
                                                      │
                                                      ▼
                                            analysis.run_candidates
```

**Decision: subsample from the raw 10,000, before intake — not from the accepted set.**

Why: it keeps the docking funnel anchored to an identical raw budget across all four
arms, so every loss between "submitted" and "scored" is visible in one funnel. Sampling
from the accepted set instead would make `scored_per_submitted` conditional on passing
intake, which quietly hides each arm's invalidity rate from the headline yield.

**Consequence, accepted deliberately:** arms end up with *different docking cohort sizes*.
A low-survival arm gets fewer scored molecules and therefore wider bootstrap confidence
intervals on its score distribution. That is correct behaviour, not a defect — a model
that yields 340 usable molecules out of 1,000 genuinely has a less certain profile, and
the interval should say so.

**Subsampling requirements:**
- `numpy.random.default_rng(seed)` with an explicitly declared integer seed per
  (arm, replicate), recorded in `run_log.json`.
- **Write the selected molecule IDs to `subsample_ids.txt`** and hash it. A subsample that
  cannot be regenerated and re-verified is not reproducible.
- Sampling is **without replacement, uniform** over the raw submissions. No stratification,
  no weighting, no preference for any property.

---

## 5. PART B — The naive baseline

### 5.1 What it is for

The falsification baseline. It answers the first question any reviewer asks: *do the
generative models beat drawing molecules out of a hat?* Two variants, because there are
two hats.

| Variant | Name | What it tests |
| --- | --- | --- |
| **A** | `uniform` | Unfiltered random draw from the source library. The dumbest possible competitor. Any model losing to this is broken. |
| **B** | `property_matched` | Random draw matched to the DYRK1A actives' physicochemical distribution. The **hard** baseline. |

Variant B is the interesting one: it isolates how much of a generative model's apparent
performance comes from *learning chemistry* versus merely *emitting molecules in the right
size and lipophilicity range*.

### 5.2 Source library — and the honest caveat

**Source: the ChEMBL bulk SMILES release** (`chembl_<version>_chemreps.txt.gz` from the
ChEMBL FTP `latest/` directory).

Do **not** hardcode a version number. Resolve whatever `latest/` currently serves, then
record the resolved filename, version string, download URL, and SHA-256 in `summary.json`.
Follow the established pattern in `src/utils/fetch_dude_decoys.py`: download raw, keep raw,
normalize separately, hash everything, record the citation.

Why ChEMBL over ZINC: the actives came from ChEMBL, so the property space is directly
comparable; and REINVENT's prior is ChEMBL-trained, which sharpens the gen2 question to
*"did reinforcement learning find anything its own training distribution did not already
contain?"*

**Caveat that must appear in the baseline's `interpretation` block and in the paper's
limitations:** this baseline is a **chemical-space control, not a training-set holdout.**
REINVENT's prior saw ChEMBL; MOSES-family gen1 models saw ZINC. A true holdout library
unavailable to all three models remains an open project decision.

**Therefore: make the source swappable.** `--source-file` takes a path plus a required
`--source-description` string, both recorded. When a holdout library is chosen later, the
module does not change — only its arguments do.

### 5.3 Variant A — uniform

1. Read the source SMILES.
2. RDKit parse; drop unparseable rows (count them).
3. Extract parent (largest fragment) using the **same deterministic rule as
   `intake.py`**: largest heavy-atom count → largest MW → canonical isomeric SMILES.
4. Deduplicate on parent canonical SMILES.
5. Exclude any molecule whose **InChIKey** matches one of the 1,219 actives. (InChIKey =
   a hashed exact-structure identifier; used here rather than SMILES matching because it
   is canonical across toolkits.) Record the removal count — expected to be a handful.
6. Sample `N` uniformly without replacement using a declared seed.

**No property filtering, no drug-likeness filtering.** This variant is *supposed* to
include peptides and oversized natural products, and is *supposed* to lose a substantial
fraction at the Tier-1 gate. That loss is the measurement.

### 5.4 Variant B — property-matched

Match on the same five continuous properties plus exact formal charge used by
`select_decoys.py`: MW, cLogP, HBD, HBA, rotatable bonds; charge matched exactly.

**Scaling.** Per-property standard deviation across the 1,219 actives. **No minimum-scale
floor.**

> Justification, and a small win for the project: `select_decoys.py` applies
> `MINIMUM_SCALES = [25.0 Da, 0.50 logP, 1 HBD, 1 HBA, 1 RotB]`. The scales actually used
> in production were MW 76.77, cLogP 1.106, HBD 1.280, HBA 1.763, RotB 2.042 — **every one
> exceeds its floor, so the floor never activated on this dataset.** The baseline can
> therefore use raw standard deviations and produce identical scaling without importing a
> constant that has no published justification.

**Matching algorithm** — deliberately simpler than `select_decoys.py`:

1. Partition candidates by formal charge; charge classes never mix.
2. Build a `scipy.spatial.cKDTree` per charge class in scaled 5-D property space.
3. Repeat until `N` unique molecules are collected:
   a. Draw one active uniformly at random **with replacement**.
   b. Query its `k = 500` nearest candidates in the matching charge class.
   c. Randomly select one not already used; add to the output set.
4. Maintain a global used-set so every baseline molecule is unique.
5. If a draw finds no unused candidate in the neighbourhood, retry with a different
   active; after a declared retry ceiling, **raise** with the exact counts. Never silently
   return fewer than `N`.

Two details that matter:

- **Random pick within the neighbourhood, not nearest.** Always taking the nearest
  neighbour would produce a set *tighter* than the actives' own distribution — a
  baseline artificially concentrated at the distribution's centre. Random-within-k
  reproduces the marginals instead of collapsing them.
- **Draw actives with replacement.** The target is the actives' *distribution*, not a
  per-active quota. There is no 50-per-active structure here and no bipartite matcher.

**No Tanimoto / topology gate.** This is the single most important difference from
`select_decoys.py`. The decoy selector's `max_tanimoto = 0.5` cap exists to make decoys
*probably not actives*. A baseline must not have it: excluding molecules that structurally
resemble known DYRK1A chemistry would handicap the baseline in precisely the direction
that flatters the generative models. If the reimplementation contains a similarity
threshold anywhere, it is wrong.

### 5.5 Build fresh — do not import from `select_decoys.py`

Rebuild the matching logic in `naive_baseline.py`. Reasons:

- Different objective: distribution matching vs. per-active quota.
- Different uniqueness semantics: global unique set vs. bipartite assignment.
- The topology gate must be **absent**, and parameterizing `select_decoys.py` with a flag
  that disables its own scientific safeguard is a trap waiting for a future maintainer.

**Guard the duplication with a test instead of shared code:** assert that the per-property
standard deviations computed by `naive_baseline.py` match the `matching_scales` recorded in
`data/external/dyrk1a_decoys_v1_20260719/selection.json` to within float tolerance. That
cross-validates the two implementations without coupling them — arguably stronger evidence
than a shared function, since it proves two independent code paths agree.

### 5.6 CLI and run matrix

```bash
python -m src.generation.naive_baseline \
  --mode uniform|property_matched \
  --source-file PATH \
  --source-description "ChEMBL <version> bulk chemreps, full set, no filtering" \
  --actives data/reference/dyrk1a_actives_chembl.csv \
  --n 10000 \
  --seed 20260801 \
  --outdir data/generated/naive_<mode>_seed<seed>
```

Run matrix: 2 modes × 3 seeds = **6 runs**, 60,000 molecules total.

Declared seeds: **20260801, 20260802, 20260803.** Same three seeds for every arm in the
benchmark, so replicate `n` of one arm is comparable to replicate `n` of another.

Outputs per run: `molecules.smi`, `selection.json` (parameters, source record + hash,
scales, funnel counts, exclusions, timing, hardware, `interpretation` block), and — for
variant B — `matched_pairs.csv` recording which active each molecule was drawn against and
its scaled property distance, for auditability.

### 5.7 Sample size, locked for the whole benchmark

**10,000 generated per arm per seed. Free metrics on all 10,000. Dock a random 1,000.**

Total docking load: 4 arms × 3 seeds × 1,000 = **12,000 dockings**, before Tier-1/2/3
attrition. Every arm gets the same numbers. Validity, uniqueness, novelty and enrichment
factor all shift with N, so comparing a 10,000-molecule arm to a 1,000-molecule arm is not
a comparison.

---

## 6. Reporting shape — identical for all four arms

```
 1,000 submitted (raw, identical across all arms and seeds)
    −A  invalid SMILES                      [Tier 0]
    −B  duplicate parent                    [Tier 0]
    −C  outside instrument range            [Tier 1, itemized by gate]
   ─────
     N  dockable
    −D  preparation failed                  [Tier 2, itemized by reason]
    −E  docking failed / timeout            [Tier 3, itemized by status]
   ─────
     M  scored
```

Two headline rates, both **generative-quality** metrics:

- `dockable_per_submitted = N / 1000`
- `scored_per_submitted   = M / 1000`

These two numbers are the "99% garbage" detector. They are reported beside the score
distribution and **never averaged into it**.

Keep the three metric buckets separate, as established earlier in the project:

| Bucket | Question | Example metrics |
| --- | --- | --- |
| Harness validation | Can the ruler tell actives from decoys? | ROC-AUC, BEDROC, EF |
| Generative quality | What fraction of output is usable at all? | validity, uniqueness, novelty, survival rates |
| Per-molecule quality | How good are the molecules that survived? | docking score, QED, SA |

---

## 7. Tests — required before either module is considered done

Add to `tests/`, matching existing `unittest` conventions
(`python -m unittest discover -s tests -t .`; pytest is not installed).

**`tests/test_generation_filter.py`**
1. A molecule at each bound is admitted; one atom / one torsion past each bound is rejected
   (boundary inclusivity is explicit and tested, not assumed).
2. Boron-containing and silicon-containing molecules are rejected, and appear in
   `disallowed_element_histogram`.
3. A molecule failing two gates simultaneously is recorded as failing **both**;
   `exclusions_by_gate` sums to more than the exclusion count, and `counting_note` is present.
4a. All 1,219 DYRK1A actives pass `MIN_HEAVY_ATOMS`, `MAX_HEAVY_ATOMS`, and
   `MAX_ROTATABLE_BONDS`. These bounds were derived from the actives, so this is a
   genuine self-consistency check. **If this test fails, the bounds are wrong.**
4b. Exactly 1,217 of 1,219 actives pass the complete gate. The two exclusions are
   `CHEMBL4288096` and `CHEMBL5176894`, both excluded solely by the element gate
   for silicon. Assert both IDs explicitly by name and assert that both pass the
   size and torsion gates. `ALLOWED_ELEMENTS` was derived from AutoDock/Vina atom-type
   parameterization, not from the actives; requiring the actives to satisfy a bound
   they had no role in setting was an unearned assumption in the original spec.
5. `gate_summary.json` records the SHA-256 of `filter_config.py`, and the recorded constant
   values equal the imported ones.
6. The gate is deterministic: same input twice → byte-identical `gate_decisions.csv`.
7. There is no code path by which caller-supplied arguments alter a threshold.

**`tests/test_generation_naive_baseline.py`**
1. Same seed twice → identical molecule set; different seed → different set.
2. All returned molecules are unique parents.
3. No returned molecule's InChIKey matches an active.
4. Variant B: per-property standard deviations match `selection.json`'s recorded
   `matching_scales` within tolerance (the cross-validation from §5.5).
5. Variant B: no formal-charge class mixing between a molecule and its matched active.
6. Variant B: the output's mean MW and cLogP fall within a declared tolerance of the
   actives' means — i.e. the matching demonstrably worked.
7. Variant B contains **no** similarity/Tanimoto threshold: assert that at least one
   returned molecule exceeds 0.5 Tanimoto to some active. (If this test cannot pass on a
   large draw, investigate — a topology gate may have crept in.)
8. Insufficient pool → raises **before** creating any output directory.
9. `--n` larger than the deduplicated eligible pool raises with exact counts.

---

## 8. Pre-registration — do this before generating anything

The order is the point. A filter written after seeing results is not a filter, it is a
knob.

1. Implement Part A. Run the tests. Confirm all 1,219 actives pass the derived
   size and torsion bounds, and that only the two pre-declared silicon actives
   fail the complete gate.
2. Commit `filter_config.py`, `filter.py`, this spec (into `docs/`), and the tests.
3. **Tag the commit:** `git tag -a predock-gate-v1 -m "Pre-dock gate constants frozen before any generative arm sampling"`
4. Only then implement Part B and generate molecules.
5. Every downstream run records `filter_config.py`'s SHA-256 in its `run_log.json`.

The git timestamp then proves the gate predates every result, which converts *"trust me,
I didn't tune this"* into evidence. Same provenance machinery already used for docking,
applied one layer up.

---

## 9. Open items created by this spec

- [ ] Record a citable source for the torsion/search-reliability claim behind
      `MAX_ROTATABLE_BONDS` (§3.4 item 4).
- [ ] Decide whether a true training-set holdout library replaces or supplements the
      ChEMBL baseline source (§5.2).
- [ ] Confirm the resolved ChEMBL version and record its hash.
- [ ] Note in the harness open-items list that `MINIMUM_SCALES` never activated in
      production (§5.4) — this partially closes an existing unjustified-constant issue.
- [ ] Confirm `CHEMBL4288096` and `CHEMBL5176894` contain silicon via PubChem before
      publication; note both in the paper's limitations section.

---
