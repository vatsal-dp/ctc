#!/usr/bin/env python3

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from analyze_tracking_failures import analyze_failures
from evaluate_ctc_results import summarize_official_logs
from run_tiptracking_standalone import _resolve_output_digits
from validate_ctc_result_format import ValidationError, validate_ctc_result_format
from view_tracking_overlay import (
    _build_lineage_layout,
    _lineage_plot_segments,
    _parse_track_file,
    export_overlay_lineage_frames,
)


class CTCPipelineToolTests(unittest.TestCase):
    def test_output_digits_auto_and_bounds(self):
        self.assertEqual(_resolve_output_digits("auto", [Path("mask000.tif")], 2), 3)
        self.assertEqual(_resolve_output_digits("auto", [Path("mask0000.tif")], 2), 4)
        self.assertEqual(_resolve_output_digits("auto", [Path("0002243_cp_masks.tif")], 1001), 4)
        self.assertEqual(_resolve_output_digits("auto", [Path("mask0000.tif")], 28177), 5)
        self.assertEqual(_resolve_output_digits("5", [Path("mask000.tif")], 28177), 5)
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

    def test_validator_accepts_five_digit_frame_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eval" / "BF-C2DL-HSC"
            source = Path(tmp) / "BF-C2DL-HSC"
            result_dir = root / "01_RES"
            image_dir = source / "01"
            result_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)

            tifffile.imwrite(image_dir / "t00000.tif", np.zeros((6, 6), dtype=np.uint16))
            tifffile.imwrite(image_dir / "t00001.tif", np.zeros((6, 6), dtype=np.uint16))

            mask0 = np.zeros((6, 6), dtype=np.uint16)
            mask0[1:3, 1:3] = 1
            mask1 = np.zeros((6, 6), dtype=np.uint16)
            mask1[1:3, 1:3] = 1
            tifffile.imwrite(result_dir / "mask00000.tif", mask0)
            tifffile.imwrite(result_dir / "mask00001.tif", mask1)
            (result_dir / "res_track.txt").write_text("1 0 1 0\n", encoding="utf-8")

            report = validate_ctc_result_format(
                dataset_root=root,
                source_root=source,
                sequence="01",
                digits_arg="auto",
            )
            self.assertEqual(report["digits"], 5)
            self.assertEqual(report["frames"], 2)
            self.assertEqual(report["tracks"], 1)

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

    def test_official_log_parser_writes_penalty_and_seg_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "eval"
            result_dir = root / "01_RES"
            result_dir.mkdir(parents=True)
            (result_dir / "TRA_log.txt").write_text(
                "\n".join(
                    [
                        "----------Splitting Operations (Penalty=5)----------",
                        "T=0 Label=1",
                        "----------False Negative Vertices (Penalty=10)----------",
                        "T=2 GT_label=4",
                        "T=3 GT_label=12",
                        "----------False Positive Vertices (Penalty=1)----------",
                        "----------Redundant Edges To Be Deleted (Penalty=1)----------",
                        "[T=4 Label=15] -> [T=6 Label=11]",
                        "----------Edges To Be Added (Penalty=1.5)----------",
                        "[T=0 GT_label=1] -> [T=1 GT_label=1]",
                        "----------Edges with Wrong Semantics (Penalty=1)----------",
                        "========================================================",
                        "TRA measure: 0.750000",
                    ]
                ),
                encoding="utf-8",
            )
            (result_dir / "DET_log.txt").write_text(
                "\n".join(
                    [
                        "----------Splitting Operations (Penalty=5)----------",
                        "----------False Negative Vertices (Penalty=10)----------",
                        "T=2 GT_label=4",
                        "----------False Positive Vertices (Penalty=1)----------",
                        "T=0 Label=2",
                        "T=2 Label=7",
                        "========================================================",
                        "DET measure: 0.500000",
                    ]
                ),
                encoding="utf-8",
            )
            (result_dir / "SEG_log.txt").write_text(
                "\n".join(
                    [
                        "----------T=5 Z=0----------",
                        "GT_label=1 J=0.675441",
                        "GT_label=2 J=0",
                        "GT_label=3 J=0.494483",
                        "----------T=7 Z=2----------",
                        "GT_label=4 J=0.5",
                        "GT_label=5 J=0.499999",
                        "========================================================",
                        "SEG measure: 0.232874",
                    ]
                ),
                encoding="utf-8",
            )

            summary = summarize_official_logs(root, low_jaccard_threshold=0.5)

            self.assertEqual(summary["tra_rows"], 6)
            self.assertEqual(summary["det_rows"], 3)
            self.assertEqual(summary["seg_low_rows"], 3)
            with (root / "ctc_TRA_penalty_counts.csv").open(encoding="utf-8", newline="") as handle:
                tra_rows = list(csv.DictReader(handle))
            tra_counts = {row["penalty_type"]: row for row in tra_rows}
            self.assertEqual(tra_counts["false_negative_vertices"]["count"], "2")
            self.assertEqual(tra_counts["false_negative_vertices"]["weighted_penalty"], "20")
            self.assertEqual(tra_counts["false_negative_vertices"]["official_score"], "0.75")
            self.assertEqual(tra_counts["false_positive_vertices"]["count"], "0")

            with (root / "ctc_DET_penalty_counts.csv").open(encoding="utf-8", newline="") as handle:
                det_rows = list(csv.DictReader(handle))
            det_counts = {row["penalty_type"]: row for row in det_rows}
            self.assertEqual(det_counts["false_positive_vertices"]["count"], "2")
            self.assertEqual(det_counts["false_positive_vertices"]["official_score"], "0.5")

            with (root / "ctc_SEG_low_jaccard_objects.csv").open(encoding="utf-8", newline="") as handle:
                low_rows = list(csv.DictReader(handle))
            self.assertEqual([row["gt_label"] for row in low_rows], ["2", "3", "5"])
            self.assertEqual(low_rows[0]["official_score"], "0.232874")
            self.assertEqual(low_rows[0]["t"], "5")
            self.assertEqual(low_rows[-1]["z"], "2")

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

    def test_lineage_layout_grows_when_children_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            track_file = Path(tmp) / "res_track.txt"
            track_file.write_text("1 0 1 0\n2 2 3 1\n3 2 3 1\n", encoding="utf-8")

            track_rows = _parse_track_file(track_file)
            layout = _build_lineage_layout(track_rows)

            self.assertAlmostEqual(layout.y_positions[1], 0.5)
            self.assertAlmostEqual(layout.y_positions[2], 0.0)
            self.assertAlmostEqual(layout.y_positions[3], 1.0)

            before_split = _lineage_plot_segments(track_rows, layout, current_frame=1)
            self.assertEqual([segment["track_id"] for segment in before_split["tracks"]], [1])
            self.assertEqual(before_split["connectors"], [])

            after_split = _lineage_plot_segments(track_rows, layout, current_frame=2)
            self.assertEqual({segment["track_id"] for segment in after_split["tracks"]}, {1, 2, 3})
            self.assertEqual(
                {(segment["parent_id"], segment["child_id"]) for segment in after_split["connectors"]},
                {(1, 2), (1, 3)},
            )

    def test_overlay_lineage_export_writes_png_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            mask_dir = root / "masks"
            export_dir = root / "export"
            image_dir.mkdir()
            mask_dir.mkdir()

            for frame_idx in range(3):
                image = np.zeros((12, 12), dtype=np.uint16)
                image[2:10, 2:10] = frame_idx + 1
                tifffile.imwrite(image_dir / f"t{frame_idx:03d}.tif", image)

            mask0 = np.zeros((12, 12), dtype=np.uint16)
            mask0[2:6, 2:6] = 1
            mask1 = np.zeros((12, 12), dtype=np.uint16)
            mask1[2:6, 2:6] = 1
            mask2 = np.zeros((12, 12), dtype=np.uint16)
            mask2[2:5, 2:5] = 2
            mask2[6:9, 6:9] = 3

            tifffile.imwrite(mask_dir / "mask000.tif", mask0)
            tifffile.imwrite(mask_dir / "mask001.tif", mask1)
            tifffile.imwrite(mask_dir / "mask002.tif", mask2)

            track_file = mask_dir / "res_track.txt"
            track_file.write_text("1 0 1 0\n2 2 2 1\n3 2 2 1\n", encoding="utf-8")

            output_paths = export_overlay_lineage_frames(
                image_files=sorted(image_dir.glob("*.tif")),
                mask_files=sorted(mask_dir.glob("mask*.tif")),
                alpha=0.45,
                export_dir=export_dir,
                track_rows=_parse_track_file(track_file),
            )

            self.assertEqual([path.name for path in output_paths], [f"t{idx:03d}_overlay_lineage.png" for idx in range(3)])
            for path in output_paths:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_overlay_lineage_export_rejects_mask_label_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            mask_dir = root / "masks"
            export_dir = root / "export"
            image_dir.mkdir()
            mask_dir.mkdir()

            tifffile.imwrite(image_dir / "t000.tif", np.zeros((8, 8), dtype=np.uint16))
            bad_mask = np.zeros((8, 8), dtype=np.uint16)
            bad_mask[2:5, 2:5] = 2
            tifffile.imwrite(mask_dir / "mask000.tif", bad_mask)

            track_file = mask_dir / "res_track.txt"
            track_file.write_text("1 0 0 0\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                export_overlay_lineage_frames(
                    image_files=sorted(image_dir.glob("*.tif")),
                    mask_files=sorted(mask_dir.glob("mask*.tif")),
                    alpha=0.45,
                    export_dir=export_dir,
                    track_rows=_parse_track_file(track_file),
                )


if __name__ == "__main__":
    unittest.main()
