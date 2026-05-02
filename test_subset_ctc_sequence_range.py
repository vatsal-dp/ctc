#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from subset_ctc_sequence_range import subset_ctc_sequence_range


class SubsetCTCSequenceRangeTests(unittest.TestCase):
    def test_source_image_sidecar_masks_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            output_root = root / "subset"
            (source_root / "01").mkdir(parents=True)
            (source_root / "01_GT" / "TRA").mkdir(parents=True)

            for frame in range(395, 398):
                (source_root / "01" / f"t{frame:03d}.tif").write_text("image", encoding="utf-8")
                (source_root / "01" / f"t{frame:03d}_cp_masks.tif").write_text("mask", encoding="utf-8")
                (source_root / "01_GT" / "TRA" / f"man_track{frame:03d}.tif").write_text(
                    "gt", encoding="utf-8"
                )
            (source_root / "01_GT" / "TRA" / "man_track.txt").write_text(
                "1 395 397 0\n", encoding="utf-8"
            )

            report = subset_ctc_sequence_range(
                source_root=source_root,
                output_root=output_root,
                sequence="01",
                start_frame=395,
                end_frame=397,
                output_digits="3",
                overwrite=False,
            )

            self.assertEqual(report["images"], 3)
            self.assertTrue((output_root / "01" / "t000.tif").is_file())
            self.assertFalse((output_root / "01" / "t000_cp_masks.tif").exists())


if __name__ == "__main__":
    unittest.main()
