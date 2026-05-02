#!/usr/bin/env python3

import csv
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

    def test_writes_source_to_output_frame_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            output_root = root / "subset"
            (source_root / "01").mkdir(parents=True)
            (source_root / "01_GT" / "TRA").mkdir(parents=True)

            for frame in range(395, 397):
                (source_root / "01" / f"t{frame:04d}.tif").write_text("image", encoding="utf-8")
                (source_root / "01_GT" / "TRA" / f"man_track{frame:04d}.tif").write_text(
                    "gt", encoding="utf-8"
                )
            (source_root / "01_GT" / "TRA" / "man_track.txt").write_text(
                "15 395 396 0\n", encoding="utf-8"
            )

            subset_ctc_sequence_range(
                source_root=source_root,
                output_root=output_root,
                sequence="01",
                start_frame=395,
                end_frame=396,
                output_digits="3",
                overwrite=False,
            )

            with (output_root / "subset_frame_mapping.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertIn(
                {
                    "kind": "image",
                    "source_frame": "395",
                    "output_frame": "0",
                    "source_path": str(source_root / "01" / "t0395.tif"),
                    "output_path": str(output_root / "01" / "t000.tif"),
                    "status": "copied",
                },
                rows,
            )
            self.assertIn(
                {
                    "kind": "tra",
                    "source_frame": "396",
                    "output_frame": "1",
                    "source_path": str(source_root / "01_GT" / "TRA" / "man_track0396.tif"),
                    "output_path": str(output_root / "01_GT" / "TRA" / "man_track001.tif"),
                    "status": "copied",
                },
                rows,
            )


if __name__ == "__main__":
    unittest.main()
