from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.generation.run_gen2_pipeline import Gen2CampaignError, run_gen2_campaign


class Gen2CampaignDesignTests(unittest.TestCase):
    def test_smaller_nonvalidation_campaign_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(Gen2CampaignError, "require --validation"):
                run_gen2_campaign(
                    root / "model",
                    root / "tokenizer",
                    root / "output",
                    raw_n=10,
                    dock_n=5,
                )

    def test_changed_registered_seed_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(Gen2CampaignError, "require --validation"):
                run_gen2_campaign(
                    root / "model",
                    root / "tokenizer",
                    root / "output",
                    seed=20260802,
                )


if __name__ == "__main__":
    unittest.main()
