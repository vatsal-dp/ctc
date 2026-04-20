#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from analyze_tracking_failures import analyze_failures
from run_tiptracking_standalone import _resolve_output_digits
from validate_ctc_result_format import ValidationError, validate_ctc_result_format


class CTCPipelineToolTests(unittest.TestCase):
    def test_output_digits_auto_and_bounds(self):
        self.assertEqual(_resolve_output_digits("auto", [Path("mask000.tif")], 2), 3)
        self.assertEqual(_resolve_output_digits("auto", [Path("mask0000.tif")], 2), 4)
        self.assertEqual(_resolve_output_digits("auto", [Path("0002243_cp_masks.tif")], 1001), 4)
        with self.assertRaises(ValueError):
            _resolve_output_digits("3", [Path("mask000.tif")], 1001)

    def test_validator_accepts_one_frame_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eval" / "BF-C2DL-HSC"
            source = Path(tmp) / "BF-C2DL-HSC"
            result_dir = root / "01_RES"
            image_dir = source / "01"
            result_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)

            tifffile.imwrite(image_dir / "t000.tif", np.zeros((6, 6), dtype=np.uint16))
            tifffile.imwrite(image_dir / "t001.tif", np.zeros((6, 6), dtype=np.uint16))

            mask0 = np.zeros((6, 6), dtype=np.uint16)
            mask0[1:3, 1:3] = 1
            mask1 = np.zeros((6, 6), dtype=np.uint16)
            mask1[1:3, 1:3] = 1
            mask1[4:5, 4:5] = 2
            tifffile.imwrite(result_dir / "mask000.tif", mask0)
            tifffile.imwrite(result_dir / "mask001.tif", mask1)
            (result_dir / "res_track.txt").write_text("1 0 1 0\n2 1 1 0\n", encoding="utf-8")

            report = validate_ctc_result_format(
                dataset_root=root,
                source_root=source,
                sequence="01",
                digits_arg="auto",
            )
            self.assertEqual(report["digits"], 3)
            self.assertEqual(report["frames"], 2)
            self.assertEqual(report["tracks"], 2)

    def test_validator_rejects_bad_mask_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eval" / "BF-C2DL-HSC"
            result_dir = root / "01_RES"
            result_dir.mkdir(parents=True)
            tifffile.imwrite(result_dir / "mask0.tif", np.zeros((4, 4), dtype=np.uint16))
            (result_dir / "res_track.txt").write_text("", encoding="utf-8")

            with self.assertRaises(ValidationError):
                validate_ctc_result_format(
                    dataset_root=root,
                    source_root=None,
                    sequence="01",
                    digits_arg="3",
                )

    def test_validator_rejects_label_missing_from_res_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eval" / "BF-C2DL-HSC"
            result_dir = root / "01_RES"
            result_dir.mkdir(parents=True)
            mask = np.zeros((4, 4), dtype=np.uint16)
            mask[1:3, 1:3] = 1
            tifffile.imwrite(result_dir / "mask000.tif", mask)
            (result_dir / "res_track.txt").write_text("", encoding="utf-8")

            with self.assertRaises(ValidationError):
                validate_ctc_result_format(
                    dataset_root=root,
                    source_root=None,
                    sequence="01",
                    digits_arg="3",
                )

    def test_failure_analysis_writes_event_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eval" / "BF-C2DL-HSC"
            source = Path(tmp) / "BF-C2DL-HSC"
            result_dir = root / "01_RES"
            gt_tra_dir = source / "01_GT" / "TRA"
            image_dir = source / "01"
            out_dir = Path(tmp) / "failures"
            result_dir.mkdir(parents=True)
            gt_tra_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)

            tifffile.imwrite(image_dir / "t000.tif", np.zeros((8, 8), dtype=np.uint16))
            tifffile.imwrite(image_dir / "t001.tif", np.zeros((8, 8), dtype=np.uint16))

            gt0 = np.zeros((8, 8), dtype=np.uint16)
            gt1 = np.zeros((8, 8), dtype=np.uint16)
            gt0[2, 2] = 1
            gt1[2, 3] = 1
            tifffile.imwrite(gt_tra_dir / "man_track000.tif", gt0)
            tifffile.imwrite(gt_tra_dir / "man_track001.tif", gt1)
            (gt_tra_dir / "man_track.txt").write_text("1 0 1 0\n", encoding="utf-8")

            res0 = np.zeros((8, 8), dtype=np.uint16)
            res1 = np.zeros((8, 8), dtype=np.uint16)
            res0[1:4, 1:4] = 1
            res1[1:4, 2:5] = 1
            res1[6:7, 6:7] = 2
            tifffile.imwrite(result_dir / "mask000.tif", res0)
            tifffile.imwrite(result_dir / "mask001.tif", res1)
            (result_dir / "res_track.txt").write_text("1 0 1 0\n2 1 1 0\n", encoding="utf-8")

            report = analyze_failures(
                dataset_root=root,
                source_root=source,
                sequence="01",
                out_dir=out_dir,
                digits_arg="auto",
                coverage_threshold=0.5,
                iou_threshold=0.5,
                split_coverage_threshold=0.2,
                jump_pixels=50.0,
                jump_factor=5.0,
                max_thumbnails=1,
            )

            self.assertGreaterEqual(report["events"], 1)
            self.assertTrue((out_dir / "failure_events.csv").is_file())
            self.assertTrue((out_dir / "failure_summary_by_frame.csv").is_file())
            self.assertTrue((out_dir / "failure_summary_by_track.csv").is_file())


if __name__ == "__main__":
    unittest.main()
