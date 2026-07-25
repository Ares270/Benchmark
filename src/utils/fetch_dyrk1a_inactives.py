"""Fetch a conservative measured-inactive DYRK1A cohort from ChEMBL.

This cohort is an orthogonal, high-label-confidence control.  It is not large
enough to replace the property-matched presumed-decoy cohort and must remain a
separate label in reports.

Inactive evidence:
  * biochemical DYRK1A IC50 in nM;
  * no ChEMBL data-validity comment;
  * exact IC50 >= 10,000 nM, or a '>' / '>=' lower bound >= 10,000 nM.

Conflict exclusion:
  * remove the entire molecule if any exact/upper-bound record is <= 1,000 nM.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from src.harness import runtime


TARGET_CHEMBL_ID = "CHEMBL2292"
ACTIVE_CONFLICT_NM = 1000.0
INACTIVE_THRESHOLD_NM = 10000.0
PAGE_LIMIT = 1000
API_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
SCHEMA_VERSION = 1


class InactiveFetchError(ValueError):
    """Raised when ChEMBL data cannot support an unambiguous cohort."""




###### Quick cleaning up of the messy relation symbol that CheMBl hands u ######### 

def _relation(value: object) -> str:
    return str(value or "").strip().strip("'")


###   Valuer to float, safely   ###

def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None      # if it cant, return none, and reject the negative (only positivity in this code)


def _validity_is_clean(record: dict) -> bool:       # Checks that warning-label field
    value = record.get("data_validity_comment")     # Returns true only if data_validity_comment is blank/missing
    return value is None or not str(value).strip()




#######     apples-to-apples gate     #######

def _record_is_in_scope(record: dict) -> bool:
    return (
        str(record.get("target_chembl_id", TARGET_CHEMBL_ID)).strip()
        == TARGET_CHEMBL_ID
        and str(record.get("standard_type", "")).strip().upper() == "IC50"  # an IC50
        and str(record.get("standard_units", "")).strip().lower() == "nm"   # in nM
        and str(record.get("assay_type", "")).strip().upper() == "B"        # and biochemical (type B)
    )








def _is_active_conflict(record: dict) -> bool:
    value = _number(record.get("standard_value"))                               # Returns true if one record looks active
    return (                                                                    # an exact-or-upper-bound (=, <, <=) 
        value is not None                                                       # value at or below 1,000 nM
        and _relation(record.get("standard_relation")) in {"=", "<", "<="}
        and value <= ACTIVE_CONFLICT_NM
    )


def _is_conservative_inactive(record: dict) -> bool:
    value = _number(record.get("standard_value"))
    if value is None or not _validity_is_clean(record):
        return False
    relation = _relation(record.get("standard_relation"))
    return (
        relation == "=" and value >= INACTIVE_THRESHOLD_NM
    ) or (
        relation in {">", ">="} and value >= INACTIVE_THRESHOLD_NM
    )











#############       BODY OF THE SCRIPT       ##############

def select_conservative_inactives(records: list[dict]) -> tuple[list[dict], dict]:
    """Apply molecule-level inactive evidence and active-conflict exclusion."""

    by_molecule: dict[str, list[dict]] = {}
    counts = {
        "raw_records": len(records),
        "out_of_scope_records": 0,
        "records_without_molecule_id": 0,
        "records_without_smiles": 0,
        "inactive_evidence_records": 0,
        "active_conflict_records": 0,
        "molecules_with_inactive_evidence": 0,
        "molecules_excluded_for_active_conflict": 0,
        "selected_unique_molecules": 0,
    }
    for record in records:                                                  # Walks in two passes
        if not _record_is_in_scope(record):                                     
            counts["out_of_scope_records"] += 1                                     # First pass:
            continue                                                                       # bucket every in-scope record by which molecule it belongs to
        molecule_id = str(record.get("molecule_chembl_id", "")).strip()                    # tallying counts as it goes
        if not molecule_id:
            counts["records_without_molecule_id"] += 1                              # Second pass:
            continue                                                                       # for each molecule, keep it only if it has
        by_molecule.setdefault(molecule_id, []).append(record)                             # inactive evidence and no active-conflict record anywhere
        if _is_conservative_inactive(record):
            counts["inactive_evidence_records"] += 1
        if _is_active_conflict(record):
            counts["active_conflict_records"] += 1                                  # Then it picks one representative SMILES
                                                                                    # records the min/max values
    selected = []                                                                   # which relations appeared
    for molecule_id, molecule_records in sorted(by_molecule.items()):               # and a label describing the evidence kind (exact / lower-bound / both)   
        inactive_records = [
            record for record in molecule_records if _is_conservative_inactive(record)
        ]
        if not inactive_records:
            continue
        counts["molecules_with_inactive_evidence"] += 1
        if any(_is_active_conflict(record) for record in molecule_records):
            counts["molecules_excluded_for_active_conflict"] += 1
            continue
        smiles_values = [
            str(record.get("canonical_smiles", "")).strip()
            for record in inactive_records
            if str(record.get("canonical_smiles", "")).strip()
        ]
        if not smiles_values:
            counts["records_without_smiles"] += len(inactive_records)
            continue
        smiles_counts = Counter(smiles_values)
        canonical_smiles = sorted(
            smiles_counts,
            key=lambda smiles: (-smiles_counts[smiles], smiles),
        )[0]
        values = [
            float(record["standard_value"])
            for record in inactive_records
        ]
        relations = [_relation(record.get("standard_relation")) for record in inactive_records]
        assay_ids = sorted(
            {
                str(record.get("assay_chembl_id", "")).strip()
                for record in inactive_records
                if str(record.get("assay_chembl_id", "")).strip()
            }
        )
        selected.append(
            {
                "molecule_chembl_id": molecule_id,
                "canonical_smiles": canonical_smiles,
                "n_inactive_measurements": len(inactive_records),
                "minimum_reported_or_bounded_ic50_nM": min(values),
                "maximum_reported_or_bounded_ic50_nM": max(values),
                "relations": ";".join(sorted(set(relations))),
                "evidence_kind": (
                    "exact_and_lower_bound"
                    if "=" in relations and any(value in {">", ">="} for value in relations)
                    else "exact"
                    if "=" in relations
                    else "lower_bound"
                ),
                "assay_chembl_ids": ";".join(assay_ids),
            }
        )
    counts["selected_unique_molecules"] = len(selected)
    return selected, counts                                     # Returns the selected molecules 
                                                                # plus a counts dictionary 
                                                                # tracking how many fell out at each stage.








def _request_url(offset: int) -> str:
    query = urllib.parse.urlencode(
        {
            "target_chembl_id": TARGET_CHEMBL_ID,
            "standard_type": "IC50",
            "standard_units": "nM",
            "assay_type": "B",
            "limit": PAGE_LIMIT,
            "offset": offset,
        }
    )
    return f"{API_URL}?{query}"






###### Same download-with-retries pattern as the fetcher script  ######

def _download_json(url: str, retries: int = 3) -> bytes:
    last_error = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "DYRK1A-Benchmark/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if not payload:
                raise InactiveFetchError(f"Empty ChEMBL response from {url}")
            return payload
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    raise InactiveFetchError(
        f"Could not download {url} after {retries + 1} attempts: {last_error}"
    )





##########      The pagination loop      ###########

def fetch_activity_pages(raw_dir: Path) -> tuple[list[dict], list[dict]]:
    """Download or resume all paginated activity records, retaining raw JSON."""

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    pages = []
    offset = 0
    while True:
        page_path = raw_dir / f"activities_offset_{offset:06d}.json"
        if not page_path.is_file():
            payload = _download_json(_request_url(offset))
            temporary = page_path.with_name(f".{page_path.name}.tmp")
            temporary.write_bytes(payload)
            temporary.replace(page_path)
        try:
            page = json.loads(page_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise InactiveFetchError(f"Cannot read ChEMBL page {page_path}: {error}")
        activities = page.get("activities")
        if not isinstance(activities, list):
            raise InactiveFetchError(f"ChEMBL page lacks activities list: {page_path}")
        records.extend(activities)
        pages.append(runtime.file_record(page_path))
        page_meta = page.get("page_meta", {})
        total_count = int(page_meta.get("total_count", len(records)))
        if len(records) >= total_count or not activities:
            break
        offset += len(activities)
    return records, pages


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise InactiveFetchError("Conservative filtering selected no molecules")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fetch_dyrk1a_inactives(outdir: Path) -> dict:
    """Fetch, select, and record a conservative measured-inactive cohort."""

    started = time.perf_counter()
    outdir = Path(outdir)
    summary_path = outdir / "summary.json"
    if summary_path.exists():
        raise InactiveFetchError(f"Completed output already exists: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    records, raw_pages = fetch_activity_pages(outdir / "raw")
    selected, counts = select_conservative_inactives(records)

    csv_path = outdir / "dyrk1a_measured_inactives.csv"
    smi_path = outdir / "dyrk1a_measured_inactives.smi"
    _write_csv(csv_path, selected)
    temporary_smi = smi_path.with_name(f".{smi_path.name}.tmp")
    temporary_smi.write_text(
        "".join(
            f"{row['canonical_smiles']} {row['molecule_chembl_id']}\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    temporary_smi.replace(smi_path)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "chembl_dyrk1a_measured_inactive_acquisition",
        "interpretation": {
            "label": "conservative measured weak/inactive DYRK1A cohort",
            "property_matched_presumed_decoy": False,
            "replacement_for_large_decoy_pool": False,
            "kept_separate_from_presumed_decoys": True,
        },
        "criteria": {
            "target_chembl_id": TARGET_CHEMBL_ID,
            "assay_type": "B (biochemical)",
            "standard_type": "IC50",
            "standard_units": "nM",
            "inactive_evidence": (
                "exact '=' >= 10000 nM or lower bound '>' / '>=' >= 10000 nM"
            ),
            "active_conflict_exclusion": (
                "exclude molecule with any '=', '<', or '<=' record <= 1000 nM"
            ),
            "data_validity_comment": "must be blank for inactive evidence",
        },
        "counts": counts,
        "source": {
            "api": API_URL,
            "raw_pages": raw_pages,
        },
        "outputs": {
            "csv": runtime.file_record(csv_path),
            "smiles": runtime.file_record(smi_path),
        },
        "timing": {"wall_seconds": time.perf_counter() - started},
    }
    runtime.write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch conservative measured-inactive DYRK1A compounds from ChEMBL"
    )
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = fetch_dyrk1a_inactives(args.outdir)
    except InactiveFetchError as error:
        parser.error(str(error))
    print(
        f"Selected {summary['counts']['selected_unique_molecules']:,} "
        f"conservative measured inactives in {args.outdir}"
    )


if __name__ == "__main__":
    main()
