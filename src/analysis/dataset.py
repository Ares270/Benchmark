"""Strict loading and validation for active/decoy score tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


class AnalysisInputError(ValueError):
    """Raised when an input would make the benchmark ambiguous or invalid."""


@dataclass(frozen=True)
class AnalysisDataset:
    """Validated rows used for ranking plus a JSON-serializable audit trail."""

    frame: pd.DataFrame
    audit: dict                                         # Bundle up these 4 thigns,     
    missing_policy: str                                 # provenance to payload connection
    score_direction: str





#####  File Path  ######

def _require_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise AnalysisInputError(f"Input file does not exist: {path}")
    return path




#### Readable error messages ####

def _sample(values: pd.Series, limit: int = 8) -> str:                      # Grabs the problematic pd.Series
    return ", ".join(map(str, values.astype(str).head(limit).tolist()))     # to later display that instead of flooding the terminal
















#########   data gatekeeper   ######### GATEKEEPER 1
 
def load_score_table(
    path: Path,
    cohort: str,
    *,
    id_column: str = config.ID_COLUMN,
    score_column: str = config.SCORE_COLUMN,
) -> tuple[pd.DataFrame, dict]:
    """Load one harness CSV without silently repairing schema or bad values."""

    path = _require_file(path)
    header = pd.read_csv(path, nrows=0)                                                         # check only first row
    missing_columns = [c for c in (id_column, score_column) if c not in header.columns]         # inspect name before loading file 
    if missing_columns:
        raise AnalysisInputError(
            f"{path} is missing required column(s) {missing_columns}; "
            f"found {list(header.columns)}"
        )

    raw = pd.read_csv(path, dtype={id_column: "string"})            
    ids = raw[id_column].astype("string").str.strip()                                                # ID inspection
    missing_ids = ids.isna() | ids.eq("")                                                            # checks if missing
    if missing_ids.any():                                                                            # and crash there
        rows = (np.flatnonzero(missing_ids.to_numpy()) + 2).tolist()[:8]    # +2 for indexing        # also force id as a label so its never
        raise AnalysisInputError(f"{path} has blank molecule IDs on CSV rows {rows}")                # used as arithmetic

    duplicate_ids = ids[ids.duplicated(keep=False)]                                              # ID inspection
    if not duplicate_ids.empty:                                                                  # chekcks if duplicates
        raise AnalysisInputError(                                                                # raise error
            f"{path} contains duplicate molecule IDs: {_sample(duplicate_ids)}"
        )

    raw_scores = raw[score_column]                                                               # go to score column
    scores = pd.to_numeric(raw_scores, errors="coerce")                                          # force it to be numeric
    nonempty = raw_scores.notna() & raw_scores.astype("string").str.strip().ne("")               
    malformed = nonempty & scores.isna()
    if malformed.any():                        # Corrupt scores vs Ligands that
        examples = raw_scores[malformed]       # docked and produced no score                    # check if malformed NaN values exist
        raise AnalysisInputError(              # one crashes, the other is handled               # and throw an error
            f"{path} contains non-numeric scores: {_sample(examples)}"
        )

    finite_mask = scores.notna() & np.isfinite(scores.to_numpy(dtype=float, na_value=np.nan))    # check if infinite inf/-inf values exist
    nonfinite = scores.notna() & ~finite_mask                                                    # and throw an error
    if nonfinite.any():
        raise AnalysisInputError(
            f"{path} contains infinite scores for: {_sample(ids[nonfinite])}"
        )

    out = pd.DataFrame({"molecule_id": ids, "score": scores.astype(float)})
    for optional in ("status", "reason"):
        if optional in raw.columns:
            out[optional] = raw[optional].fillna("").astype(str).str.strip()

    if "status" in out.columns:                                                                   # rows that may have no score
        status = out["status"].str.lower()
        success = status.isin({"ok", "cached"})
        if (success & out["score"].isna()).any():
            bad = out.loc[success & out["score"].isna(), "molecule_id"]
            raise AnalysisInputError(
                f"{path} marks rows successful but provides no score: {_sample(bad)}"
            )
        failure_with_score = status.ne("") & ~success & out["score"].notna()
        if failure_with_score.any():
            bad = out.loc[failure_with_score, "molecule_id"]
            raise AnalysisInputError(
                f"{path} provides scores for rows marked failed: {_sample(bad)}"
            )

    n_total = int(len(out))
    n_scored = int(out["score"].notna().sum())
    audit = {
        "cohort": cohort,
        "path": str(path.resolve()),
        "n_input": n_total,
        "n_scored": n_scored,
        "n_missing_score": n_total - n_scored,
        "coverage": n_scored / n_total if n_total else 0.0,                                        # Tally 
        "status_counts": (
            {str(k): int(v) for k, v in out["status"].value_counts(dropna=False).items()}
            if "status" in out.columns else {}                                                     # if data survives all this
        ),                                                                                         # return an "out"    - cleaned pd.Dataframe
    }                                                                                              # return an "audit"  - metadata dictionary
    return out, audit














###################    Cross Reference Validator   ################### GATEKEEPER 2

def _validate_reference(
    active_ids: set[str],
    decoy_ids: set[str],
    reference_path: Path,
    reference_id_column: str,
) -> dict:
    reference_path = _require_file(reference_path)
    header = pd.read_csv(reference_path, nrows=0)                    # Header Peek                   
    if reference_id_column not in header.columns:
        raise AnalysisInputError(
            f"{reference_path} lacks reference ID column {reference_id_column!r}; "
            f"found {list(header.columns)}"
        )
    ref = pd.read_csv(reference_path, dtype={reference_id_column: "string"})
    ref_ids_series = ref[reference_id_column].astype("string").str.strip()
    if (ref_ids_series.isna() | ref_ids_series.eq("")).any():                               # Blank Check
        raise AnalysisInputError(f"{reference_path} contains blank reference IDs")
    if ref_ids_series.duplicated().any():
        dupes = ref_ids_series[ref_ids_series.duplicated(keep=False)]                       # Duplicate Check
        raise AnalysisInputError(
            f"{reference_path} contains duplicate reference IDs: {_sample(dupes)}"
        )

    ref_ids = set(ref_ids_series.astype(str))                               # Convert column to native python set
    unknown_actives = sorted(active_ids - ref_ids)                          # for fast set comparisons 
    if unknown_actives:
        raise AnalysisInputError(
            "Scored-active IDs absent from the active reference: "          # find unknown Actives
            + ", ".join(unknown_actives[:8])                                # present in local active pool
        )                                                                   # missing from master reference file
    known_active_decoys = sorted(decoy_ids & ref_ids)
    if known_active_decoys:
        raise AnalysisInputError(                                           # Calculates intersection
            "Decoy IDs also occur in the active reference: "                # Find Actives also present in decoy pool   
            + ", ".join(known_active_decoys[:8])
        )
    return {
        "path": str(reference_path.resolve()),                              # if all crosschecks are good, return ts
        "id_column": reference_id_column,
        "n_reference_ids": len(ref_ids),
        "n_reference_ids_not_in_scored_actives": len(ref_ids - active_ids),
    }


















###################       MISSING SCORE POLICY ETC          ####################

def build_dataset(
    active_scores_path: Path,
    decoy_scores_path: Path,
    reference_path: Path | None = None,
    *,
    missing_policy: str = config.MISSING_SCORE_POLICY,
    score_direction: str = config.SCORE_DIRECTION,
    id_column: str = config.ID_COLUMN,
    score_column: str = config.SCORE_COLUMN,
    reference_id_column: str = config.REFERENCE_ID_COLUMN,
) -> AnalysisDataset:
    """Combine validated cohorts and apply an explicit missing-score policy."""

    if missing_policy not in config.VALID_MISSING_SCORE_POLICIES:
        raise AnalysisInputError(f"Unknown missing-score policy: {missing_policy}")
    if score_direction not in config.VALID_SCORE_DIRECTIONS:
        raise AnalysisInputError(f"Unknown score direction: {score_direction}")

    actives, active_audit = load_score_table(                                                   # calls load_score_table
        active_scores_path, "actives", id_column=id_column, score_column=score_column           # to independently scrub and load
    )                                                                                           # the actives and decoys
    decoys, decoy_audit = load_score_table(                                                     # then check for empty files/data leakage
        decoy_scores_path, "decoys", id_column=id_column, score_column=score_column
    )
    if actives.empty or decoys.empty:
        raise AnalysisInputError("Both input cohorts must contain at least one row")

    active_ids = set(actives["molecule_id"].astype(str))
    decoy_ids = set(decoys["molecule_id"].astype(str))
    overlap = sorted(active_ids & decoy_ids)
    if overlap:
        raise AnalysisInputError(
            "Molecule IDs occur in both active and decoy inputs: "
            + ", ".join(overlap[:8])
        )

    reference_audit = None
    if reference_path is not None:
        reference_audit = _validate_reference(                              # Delegate work to _validate_reference
            active_ids, decoy_ids, reference_path, reference_id_column      # for cross reference checking  
        )

    actives = actives.assign(label=1)
    decoys = decoys.assign(label=0)
    frame = pd.concat([actives, decoys], ignore_index=True, sort=False)
    frame["score_imputed"] = frame["score"].isna()
    n_missing = int(frame["score_imputed"].sum())

    if n_missing and missing_policy == "error":                                                  # ERROR POLICY
        raise AnalysisInputError(                                                                # If even 1 row is missing a score   
            f"{n_missing} rows have no score. Choose --missing-policy exclude "                  # then crash
            "for a score-conditional analysis or rank_last for an end-to-end analysis."         
        )
    if missing_policy == "exclude":                                                              # EXCLUDE POLICY 
        frame = frame.loc[~frame["score_imputed"]].copy()                                        # drops all missing rows
    elif missing_policy == "rank_last" and n_missing:                                            # returns smaller but 100% scored dataset
        observed = frame.loc[~frame["score_imputed"], "score"].to_numpy(float)
        if observed.size == 0:
            raise AnalysisInputError("Cannot rank failures last when every score is missing")    # RANK_LAST POLICY   
        if score_direction == "lower_is_better":                                                 # apply mathematical penalty by
            sentinel = float(np.nextafter(observed.max(), np.inf))                               # pushing the non scored to the bottom
        else:                                                                                    # works by finding the true worst value
            sentinel = float(np.nextafter(observed.min(), -np.inf))                              # and generating a sentinel value
        frame.loc[frame["score_imputed"], "score"] = sentinel                                    # which is microscopically worse than the worse value
                                                                                                 # all missing scores are replaced with this sentinel value
    counts = frame["label"].value_counts()
    if int(counts.get(1, 0)) == 0 or int(counts.get(0, 0)) == 0:
        raise AnalysisInputError("The selected missing-score policy leaves an empty class")

    audit = {
        "actives": active_audit,
        "decoys": decoy_audit,
        "reference": reference_audit,
        "missing_policy": missing_policy,
        "n_analyzed": int(len(frame)),
        "n_analyzed_actives": int((frame["label"] == 1).sum()),
        "n_analyzed_decoys": int((frame["label"] == 0).sum()),
        "n_imputed_rank_last": int(frame["score_imputed"].sum()),
    }
    return AnalysisDataset(
        frame=frame.reset_index(drop=True),
        audit=audit,
        missing_policy=missing_policy,
        score_direction=score_direction,
    )
