# Harness Validation Threshold — Pre-Registered Criteria

**Project:** DYRK1A Generative Benchmark
**Written:** 2026-08-30
**Status:** committed before any registered validation run has been executed.
**Scope:** declares, in advance, the numbers at which the docking harness is
judged fit or unfit for cross-arm comparison.

---

## 0. What this document is for

The harness is the measuring instrument: receptor preparation, box placement,
Meeko ligand preparation, Smina invocation, and the Vina scoring function.
Every result in this benchmark is a number this instrument printed.

An instrument that cannot separate known DYRK1A binders from property-matched
non-binders is not measuring binding. Any cross-arm comparison built on it is
comparing noise. This document fixes the bar *before* the measurement, so the
bar cannot be moved to accommodate the result.

A threshold chosen after seeing the number is not a threshold. It is a knob.

---

## 1. Metrics, defined

Three metrics, all computed on a labelled cohort of accepted DYRK1A actives
plus property-matched DUD-E decoys. Decoys are **presumed** negatives, not
experimentally confirmed non-binders.

**ROC AUC** (area under the receiver operating characteristic curve). Rank
every molecule by docking score, best first. AUC is the probability that a
randomly chosen active outranks a randomly chosen decoy. 0.5 is a coin flip;
1.0 is perfect separation. It weights the whole ranked list equally.

**EF_x%** (enrichment factor at x percent). Take the top x% of the ranked list.
EF is the proportion of actives found there, divided by the proportion expected
by chance. EF = 1.0 means the top of the list is no better than random. 

**BEDROC** (Boltzmann-enhanced discrimination of ROC), alpha = 20.0. A version
of AUC that weights early ranks heavily. Recorded for completeness; not used as
a pass/fail criterion here.

---

## 2. The criteria

The harness is declared **fit for cross-arm comparison** if, on a
registered validation cohort meeting the minimum size.

- **ROC AUC >= 0.65 **
- **EF 1%   >= 5.00 **

Both must hold. Failing either means the harness is unfit.

### Rationale

Docking enrichment against kinases with well-defined ATP-binding pockets
typically lands in the AUC 0.65-0.80 range in the published literature.

The enrichment-factor cutoff is stated at a percentile with adequate
resolution at the declared cohort size.




( the criteria in §2 apply only to a registered validation run with ≥1,000 actives; pilot runs are exempt.)

---

## 3. Redocking criterion (separate capability)

Pose reproduction and active/decoy discrimination are different things. A
harness can place ligands correctly and still fail to rank them, and vice versa.

**Criterion: top-scored redocked pose of the 7O7K co-crystal ligand (6ZV,
abemaciclib) must fall below 2.0 A in-place RMSD from the deposited
coordinates**, computed without superposition (`obrms` without alignment, or
`rdMolAlign.CalcRMS`).

Honesty note, recorded rather than hidden: a value of 1.578 A was already
measured on 2026-07-05, under a docking box 30x28x35 A. The 2.0 A figure is the standard
convention in the docking literature, not a number reverse-engineered to
accommodate that result.

---
## 4. Amendment policy — what may change, and when

### 4.1 Pilot phase (before the registered validation run)

Runs recorded with `role="pilot"` exist to shake out defects. During this
phase, any aspect of the harness may be changed for any reason. Pilot
enrichment numbers are not results, are not cited, and do not trigger this
document's criteria. The registered validation run is the first run to which
§2 applies.

### 4.2 Defect repair — always permitted

A defect is behaviour that contradicts what the harness is documented to do:
duplicate dockings, misjoined score tables, silent failures, non-unique
molecule IDs, incorrect receptor. Repairing a defect is permitted at any
time, in any phase, and requires no amendment beyond a commit that states
what was broken and how it was detected.

A defect must be characterised **independently of the enrichment result.**

### 4.3 Label-blind instrument QC — always permitted

The following diagnostics are computed on unlabelled scores and may be run,
failed, and acted on freely:

- distinct-molecule and distinct-ligand-hash counts equal submitted counts
- per-molecule score standard deviation across three Smina seeds
  (sampling convergence)
- Tier-3 timeout and failure counts
- redock RMSD under the production box

Changes made in response to these diagnostics are permitted without
amendment, because no enrichment metric was consulted. Each such change is
recorded with the diagnostic output that motivated it.

### 4.4 Post-result tuning — the thing this document forbids

After a registered validation run has produced an enrichment number, changing
box dimensions, exhaustiveness, scoring function, or receptor preparation
**in response to that number** requires:

1. a dated amendment to this document stating the change and its motivation,
2. a fresh validation run under a new tag,
3. reporting of both the original and the revised result.

The original result is not deleted. Retuning an instrument until it passes is
how a benchmark becomes a self-fulfilling prophecy; this clause does not
prohibit the path, it makes taking it visible.


---

## 5. Configuration frozen at this commit

The criteria above are meaningless unless the instrument they judge is fixed.
As of this commit, the docking configuration in `src/harness/config.py` — box
centre, box dimensions, exhaustiveness, number of modes, scoring function,
seed policy, and receptor PDBQT hash — is the configuration under test. Every
validation run records these values and the receptor hash in its `run_log.json`,
so any reported result can be traced to the exact instrument that produced it.

