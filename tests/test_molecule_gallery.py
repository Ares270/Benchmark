from __future__ import annotations

import unittest

import pandas as pd

from src.analysis.molecule_gallery import render_top_bottom_galleries


class MoleculeGalleryTests(unittest.TestCase):
    def test_gallery_renders_structures_and_explains_small_run_overlap(self):
        frame = pd.DataFrame(
            {
                "molecule_id": ["m1", "m2", "m3", "m4"],
                "parent_smiles": ["CCO", "CCN", "c1ccccc1", "CC(=O)O"],
                "score": [-8.0, -7.0, -6.0, -5.0],
                "molecular_weight": [46.1, 45.1, 78.1, 60.1],
                "clogp": [-0.1, -0.0, 1.7, -0.3],
                "qed": [0.4, 0.4, 0.4, 0.4],
                "sa_score": [1.0, 1.1, 1.2, 1.3],
            }
        )
        rendered = render_top_bottom_galleries(frame, n=10)
        self.assertIn("Candidate structure gallery", rendered)
        self.assertIn("Top 4", rendered)
        self.assertIn("Bottom 4", rendered)
        self.assertIn("overlap by 4 molecules", rendered)
        self.assertIn("<svg", rendered)

    def test_gallery_requires_ranked_identity_and_structure_columns(self):
        with self.assertRaisesRegex(ValueError, "parent_smiles"):
            render_top_bottom_galleries(
                pd.DataFrame({"molecule_id": ["m1"], "score": [-7.0]})
            )


if __name__ == "__main__":
    unittest.main()
