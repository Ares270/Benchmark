from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils.fetch_dude_decoys import (
    DudeFetchError,
    _normalize_files,
    discover_targets,
)


class DudeFetchTests(unittest.TestCase):
    def test_target_discovery_is_deduplicated_and_stable(self):
        page = """
        <a href="/targets/AA2AR">AA2AR</a>
        <a href="/targets/abl1">ABL1</a>
        <a href="/targets/aa2ar">duplicate</a>
        <a href="/subsets">not a target</a>
        """
        self.assertEqual(discover_targets(page), ["aa2ar", "abl1"])

    def test_normalization_preserves_smiles_and_prefixes_source_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a = root / "a.ism"
            source_b = root / "b.ism"
            source_a.write_text("CCO C0001\n", encoding="ascii")
            source_b.write_text("c1ccccc1 C0002\n", encoding="ascii")
            destination = root / "pool.smi"

            count = _normalize_files(
                [("aa2ar", source_a), ("abl1", source_b)],
                destination,
                id_prefix="DUDE",
            )

            self.assertEqual(count, 2)
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "CCO DUDE_AA2AR_C0001\n"
                "c1ccccc1 DUDE_ABL1_C0002\n",
            )

    def test_malformed_public_source_line_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "broken.ism"
            source.write_text("CCO\n", encoding="ascii")

            with self.assertRaisesRegex(DudeFetchError, "no identifier"):
                _normalize_files(
                    [("aa2ar", source)],
                    root / "pool.smi",
                    id_prefix="DUDE",
                )


if __name__ == "__main__":
    unittest.main()
