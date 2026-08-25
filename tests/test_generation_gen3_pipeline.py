from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.generation.run_gen3_pipeline import Gen3CampaignError, run_gen3_campaign


class Gen3CampaignDesignTests(unittest.TestCase):
    def test_smaller_nonvalidation_campaign_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(Gen3CampaignError, "require --validation"):
                run_gen3_campaign(
                    root / "model",
                    root / "output",
                    raw_n=10,
                    dock_n=5,
                )

    def test_changed_registered_seed_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(Gen3CampaignError, "require --validation"):
                run_gen3_campaign(
                    root / "model",
                    root / "output",
                    seed=20260802,
                )


if __name__ == "__main__":
    unittest.main()
