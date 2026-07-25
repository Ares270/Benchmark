from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.fetch_dyrk1a_inactives import (
    InactiveFetchError,
    fetch_dyrk1a_inactives,
    select_conservative_inactives,
)


def _record(
    molecule_id: str,
    value: float,
    relation: str = "=",
    *,
    smiles: str = "CCO",
    validity: str | None = None,
    assay_type: str = "B",
) -> dict:
    return {
        "target_chembl_id": "CHEMBL2292",
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "standard_value": value,
        "standard_relation": relation,
        "standard_type": "IC50",
        "standard_units": "nM",
        "assay_type": assay_type,
        "data_validity_comment": validity,
        "assay_chembl_id": f"ASSAY_{molecule_id}",
    }


class Dyrk1aInactiveSelectionTests(unittest.TestCase):
    def test_conservative_evidence_and_active_conflicts_are_molecule_level(self):
        records = [
            _record("exact", 15000),
            _record("bounded", 10000, ">"),
            _record("conflict", 20000),
            _record("conflict", 500),
            _record("too_potent", 9999),
            _record("invalid_flag", 25000, validity="Outside typical range"),
            _record("wrong_assay", 30000, assay_type="F"),
            _record("no_smiles", 30000, smiles=""),
        ]

        selected, counts = select_conservative_inactives(records)

        self.assertEqual(
            [row["molecule_chembl_id"] for row in selected],
            ["bounded", "exact"],
        )
        self.assertEqual(selected[0]["evidence_kind"], "lower_bound")
        self.assertEqual(selected[1]["evidence_kind"], "exact")
        self.assertEqual(counts["molecules_with_inactive_evidence"], 4)
        self.assertEqual(counts["molecules_excluded_for_active_conflict"], 1)
        self.assertEqual(counts["selected_unique_molecules"], 2)

    def test_quoted_exact_relation_is_normalized(self):
        selected, _ = select_conservative_inactives(
            [_record("quoted", 10000, "'='")]
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["relations"], "=")

    def test_partial_directory_resumes_but_completed_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            outdir = Path(temporary) / "inactives"
            (outdir / "raw").mkdir(parents=True)
            records = [_record("inactive", 15000)]

            with patch(
                "src.utils.fetch_dyrk1a_inactives.fetch_activity_pages",
                return_value=(records, []),
            ):
                summary = fetch_dyrk1a_inactives(outdir)

            self.assertEqual(summary["counts"]["selected_unique_molecules"], 1)
            self.assertTrue((outdir / "summary.json").is_file())
            with self.assertRaisesRegex(
                InactiveFetchError,
                "Completed output",
            ):
                fetch_dyrk1a_inactives(outdir)


if __name__ == "__main__":
    unittest.main()
