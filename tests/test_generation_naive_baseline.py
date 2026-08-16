from __future__ import annotations

import csv
import gzip
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.generation.naive_baseline import (
    CONTINUOUS_MATCH_COLUMNS,
    NaiveBaselineError,
    _active_feature_array,
    _load_actives,
    _matching_scales,
    generate_naive_baseline,
)
from src.harness import runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
MEAN_TOLERANCES = {
    "molecular_weight": 35.0,
    "clogp": 0.75,
}


class GenerationNaiveBaselineTests(unittest.TestCase):
    def _write_actives(self, root: Path) -> Path:
        path = root / "actives.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "molecule_chembl_id",
                    "canonical_smiles",
                    "median_ic50_nM",
                    "n_measurements",
                ),
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "molecule_chembl_id": "ACTIVE_NEUTRAL_A",
                        "canonical_smiles": "CCCCCCCCCC",
                        "median_ic50_nM": 10,
                        "n_measurements": 1,
                    },
                    {
                        "molecule_chembl_id": "ACTIVE_NEUTRAL_B",
                        "canonical_smiles": "CCCCCCCCO",
                        "median_ic50_nM": 20,
                        "n_measurements": 1,
                    },
                    {
                        "molecule_chembl_id": "ACTIVE_POSITIVE",
                        "canonical_smiles": "CCCCCC[NH3+]",
                        "median_ic50_nM": 30,
                        "n_measurements": 1,
                    },
                ]
            )
        return path

    def _source_smiles(self) -> list[str]:
        neutral = []
        for carbons in range(6, 14):
            chain = "C" * carbons
            neutral.extend(
                [
                    chain,
                    chain + "O",
                    chain + "N",
                    chain + "F",
                ]
            )
        positive = [
            ("C" * carbons) + "[NH3+]"
            for carbons in range(3, 12)
        ]
        return [
            "CCCCCCCCCC",
            "CCCCCCCCO",
            "OCCCCCCCC",
            "not_a_smiles",
            *neutral,
            *positive,
        ]

    def _write_source(self, root: Path) -> Path:
        path = root / "chembl_99_chemreps.txt.gz"
        if path.exists():
            return path
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            handle.write(
                "chembl_id\tcanonical_smiles\tstandard_inchi\t"
                "standard_inchi_key\n"
            )
            for index, smiles in enumerate(self._source_smiles(), 1):
                handle.write(f"SOURCE_{index}\t{smiles}\t\t\n")
        return path

    def _run(
        self,
        root: Path,
        *,
        mode: str,
        n: int,
        seed: int,
        name: str,
    ) -> tuple[Path, dict]:
        actives = self._write_actives(root)
        source = self._write_source(root)
        outdir = root / name
        selection = generate_naive_baseline(
            mode=mode,
            source_file=source,
            source_description="Synthetic ChEMBL-like test source",
            actives_path=actives,
            n=n,
            seed=seed,
            outdir=outdir,
        )
        return outdir, selection

    def _output_smiles(self, path: Path) -> list[str]:
        return [
            line.split()[0]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_same_seed_is_identical_and_different_seed_changes_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = self._run(
                root, mode="uniform", n=12, seed=7, name="first"
            )
            second, _ = self._run(
                root, mode="uniform", n=12, seed=7, name="second"
            )
            third, _ = self._run(
                root, mode="uniform", n=12, seed=8, name="third"
            )

            self.assertEqual(
                (first / "molecules.smi").read_bytes(),
                (second / "molecules.smi").read_bytes(),
            )
            self.assertNotEqual(
                set(self._output_smiles(first / "molecules.smi")),
                set(self._output_smiles(third / "molecules.smi")),
            )
            selection = json.loads(
                (first / "selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selection["source"]["version_string"], "ChEMBL 99")
            self.assertEqual(
                selection["source"]["resolved_filename"],
                "chembl_99_chemreps.txt.gz",
            )
            self.assertTrue(
                selection["source"]["download_url"].endswith(
                    "/chembl_99/chembl_99_chemreps.txt.gz"
                )
            )
            self.assertEqual(
                selection["source"]["file"]["sha256"],
                runtime.sha256_file(root / "chembl_99_chemreps.txt.gz"),
            )

            matched_first, _ = self._run(
                root,
                mode="property_matched",
                n=18,
                seed=13,
                name="matched_first",
            )
            matched_second, _ = self._run(
                root,
                mode="property_matched",
                n=18,
                seed=13,
                name="matched_second",
            )
            matched_third, _ = self._run(
                root,
                mode="property_matched",
                n=18,
                seed=14,
                name="matched_third",
            )
            self.assertEqual(
                (matched_first / "molecules.smi").read_bytes(),
                (matched_second / "molecules.smi").read_bytes(),
            )
            self.assertNotEqual(
                set(self._output_smiles(matched_first / "molecules.smi")),
                set(self._output_smiles(matched_third / "molecules.smi")),
            )

    def test_uniform_outputs_unique_parents_and_excludes_active_inchikeys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outdir, selection = self._run(
                root, mode="uniform", n=20, seed=11, name="uniform"
            )
            smiles = self._output_smiles(outdir / "molecules.smi")
            active_keys = {
                active.inchikey for active in _load_actives(root / "actives.csv")
            }
            output_keys = {
                Chem.MolToInchiKey(Chem.MolFromSmiles(value)) for value in smiles
            }

            self.assertEqual(len(smiles), len(set(smiles)))
            self.assertTrue(active_keys.isdisjoint(output_keys))
            self.assertGreaterEqual(
                selection["exclusions"]["exact_active_inchikey"],
                2,
            )
            self.assertGreaterEqual(
                selection["exclusions"]["duplicate_parent"],
                1,
            )
            self.assertGreaterEqual(
                selection["exclusions"]["rdkit_parse_or_sanitize_failed"],
                1,
            )

    def test_property_scales_match_independent_decoy_selection_record(self):
        actives = _load_actives(
            REPO_ROOT / "data/reference/dyrk1a_actives_chembl.csv"
        )
        scales = _matching_scales(_active_feature_array(actives))
        fixture = json.loads(
            (
                REPO_ROOT
                / "tests/fixtures/dyrk1a_decoy_matching_scales.json"
            ).read_text(encoding="utf-8")
        )
        production_record = (
            REPO_ROOT / fixture["source_path"]
        )
        if production_record.is_file():
            self.assertEqual(
                runtime.sha256_file(production_record),
                fixture["source_sha256"],
            )
            recorded = json.loads(
                production_record.read_text(encoding="utf-8")
            )["parameters"]["matching_scales"]
        else:
            recorded = fixture["matching_scales"]

        for column, scale in zip(CONTINUOUS_MATCH_COLUMNS, scales):
            self.assertAlmostEqual(float(scale), float(recorded[column]))

    def test_property_matches_never_mix_formal_charge_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outdir, selection = self._run(
                root,
                mode="property_matched",
                n=24,
                seed=19,
                name="matched",
            )
            with (outdir / "matched_pairs.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 24)
            self.assertEqual(
                len({row["parent_smiles"] for row in rows}),
                len(rows),
            )
            self.assertTrue(
                all(
                    int(row["formal_charge"])
                    == int(row["matched_active_formal_charge"])
                    for row in rows
                )
            )
            self.assertEqual(
                selection["parameters"]["formal_charge_matching"],
                "exact",
            )

    def test_property_matching_reproduces_active_mw_and_clogp_means(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outdir, _ = self._run(
                root,
                mode="property_matched",
                n=30,
                seed=23,
                name="matched",
            )
            actives = _load_actives(root / "actives.csv")
            active_features = _active_feature_array(actives)
            with (outdir / "matched_pairs.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))
            output_mw = np.mean(
                [float(row["molecular_weight"]) for row in rows]
            )
            output_clogp = np.mean([float(row["clogp"]) for row in rows])

            self.assertLessEqual(
                abs(output_mw - float(np.mean(active_features[:, 0]))),
                MEAN_TOLERANCES["molecular_weight"],
            )
            self.assertLessEqual(
                abs(output_clogp - float(np.mean(active_features[:, 1]))),
                MEAN_TOLERANCES["clogp"],
            )

    def test_property_output_can_exceed_point_five_active_tanimoto(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outdir, selection = self._run(
                root,
                mode="property_matched",
                n=30,
                seed=29,
                name="matched",
            )
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=2,
                fpSize=2048,
                includeChirality=False,
            )
            active_fingerprints = [
                generator.GetFingerprint(
                    Chem.MolFromSmiles(active.parent_smiles)
                )
                for active in _load_actives(root / "actives.csv")
            ]
            maximum_similarities = []
            for smiles in self._output_smiles(outdir / "molecules.smi"):
                fingerprint = generator.GetFingerprint(
                    Chem.MolFromSmiles(smiles)
                )
                maximum_similarities.append(
                    max(
                        DataStructs.BulkTanimotoSimilarity(
                            fingerprint,
                            active_fingerprints,
                        )
                    )
                )

            self.assertGreater(max(maximum_similarities), 0.5)
            self.assertIsNone(
                selection["parameters"]["topology_or_similarity_filter"]
            )
            module_source = inspect.getsource(
                __import__(
                    "src.generation.naive_baseline",
                    fromlist=["generate_naive_baseline"],
                )
            )
            self.assertNotIn("DataStructs", module_source)
            self.assertNotIn("MorganGenerator", module_source)

    def test_property_pool_without_active_charge_raises_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actives = self._write_actives(root)
            source = root / "negative_only.smi"
            source.write_text(
                "".join(
                    f"{'C' * carbons}[O-] negative_{carbons}\n"
                    for carbons in range(3, 12)
                ),
                encoding="utf-8",
            )
            outdir = root / "matched"

            with self.assertRaisesRegex(
                NaiveBaselineError,
                "formal charge present",
            ):
                generate_naive_baseline(
                    mode="property_matched",
                    source_file=source,
                    source_description="Negative-only test pool",
                    actives_path=actives,
                    n=3,
                    seed=31,
                    outdir=outdir,
                )
            self.assertFalse(outdir.exists())

    def test_n_larger_than_deduplicated_pool_raises_with_exact_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actives = self._write_actives(root)
            source = root / "tiny.smi"
            source.write_text(
                "CCO one\nOCC duplicate\nCCN two\n",
                encoding="utf-8",
            )
            outdir = root / "uniform"

            with self.assertRaisesRegex(
                NaiveBaselineError,
                "Requested 3 molecules.*contains 2",
            ):
                generate_naive_baseline(
                    mode="uniform",
                    source_file=source,
                    source_description="Tiny test pool",
                    actives_path=actives,
                    n=3,
                    seed=37,
                    outdir=outdir,
                )
            self.assertFalse(outdir.exists())


if __name__ == "__main__":
    unittest.main()
