#!/usr/bin/env python3

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from analyze_tracking_failures import analyze_failures
from evaluate_ctc_results import summarize_official_logs
from run_ctc_training_pipeline import _run_sequence, _tracking_position_for_sequence
from run_tiptracking_standalone import (
    _compact_labels_in_place,
    _normalize_ctc_divisions,
    _resolve_output_digits,
    _split_discontinuous_tracks,
)
from ram_run_tiptracking_standalone_optimized import (
    _choose_mmap_dir,
    _compact_labels_in_place_fast,
    _looks_like_network_path,
    _normalize_ctc_divisions as _normalize_ctc_divisions_ram,
    _split_discontinuous_tracks as _split_discontinuous_tracks_ram,
)
from rescale_image_mask_pairs import rescale_dataset, resize_mask_array
from temporal_downsample_ctc_results import temporal_downsample_ctc_results
from validate_ctc_result_format import ValidationError, validate_ctc_result_format
from visualize_rescale_overlay import export_rescale_overlay_comparisons
from view_tracking_overlay import (
    _build_lineage_layout,
    _filter_lineage_track_rows,
    _lineage_plot_segments,
    _parse_track_file,
    export_overlay_lineage_frames,
)


class CTCPipelineToolTests(unittest.TestCase):
    def test_mmap_dir_selection_uses_local_output_dir_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "tracking_output"
            self.assertEqual(_choose_mmap_dir(output_dir, None, estimated_stack_bytes=1), output_dir)

    def test_network_path_detection_for_unc_paths(self):
        self.assertTrue(_looks_like_network_path(Path(r"\\server\share\tracking_output")))
        self.assertTrue(_looks_like_network_path(Path("//server/share/tracking_output")))
        self.assertFalse(_looks_like_network_path(Path("/tmp/tracking_output")))

    def test_output_digits_auto_and_bounds(self):
        self.assertEqual(_resolve_output_digits("auto", [Path("mask000.tif")], 2), 3)
        self.assertEqual(_resolve_output_digits("auto", [Path("mask0000.tif")], 2), 4)
        self.assertEqual(_resolve_output_digits("auto", [Path("0002243_cp_masks.tif")], 1001), 4)
        self.assertEqual(_resolve_output_digits("auto", [Path("mask0000.tif")], 28177), 5)
        self.assertEqual(_resolve_output_digits("5", [Path("mask000.tif")], 28177), 5)
        with self.assertRaises(ValueError):
            _resolve_output_digits("3", [Path("mask000.tif")], 1001)

    def test_discontinuous_split_prunes_after_track_capacity(self):
        mask_stack = np.zeros((1, 1, 5), dtype=np.uint16)
        mask_stack[:, :, 0] = 1
        mask_stack[:, :, 2] = 1
        mask_stack[:, :, 4] = 1
        tp_im = np.array([[1, 0, 1, 0, 1]], dtype=np.uint32)
        tp1 = tp_im != 0

        tp_im, tp1, short_pruned, capacity_pruned = _split_discontinuous_tracks(
            mask_stack=mask_stack,
            tp_im=tp_im,
            tp1=tp1,
            time_series_threshold=1,
            max_track_id=2,
        )

        self.assertEqual(short_pruned, 0)
        self.assertEqual(capacity_pruned, 1)
        self.assertEqual(tp_im.shape[0], 2)
        self.assertEqual(mask_stack.reshape(-1).tolist(), [1, 0, 2, 0, 0])

    def test_discontinuous_split_reuses_sparse_label_slots(self):
        mask_stack = np.zeros((1, 1, 5), dtype=np.uint16)
        mask_stack[:, :, 0] = 5
        mask_stack[:, :, 2] = 5
        mask_stack[:, :, 4] = 5
        tp_im = np.zeros((5, 5), dtype=np.uint32)
        tp_im[4, [0, 2, 4]] = 1
        tp1 = tp_im != 0

        tp_im, tp1, short_pruned, capacity_pruned = _split_discontinuous_tracks(
            mask_stack=mask_stack,
            tp_im=tp_im,
            tp1=tp1,
            time_series_threshold=1,
            max_track_id=5,
        )

        self.assertEqual(short_pruned, 0)
        self.assertEqual(capacity_pruned, 0)
        self.assertEqual(tp_im.shape[0], 5)
        self.assertEqual(mask_stack.reshape(-1).tolist(), [5, 0, 1, 0, 2])

    def test_ram_discontinuous_split_prunes_after_track_capacity(self):
        mask_stack = np.zeros((1, 1, 5), dtype=np.uint16)
        mask_stack[:, :, 0] = 1
        mask_stack[:, :, 2] = 1
        mask_stack[:, :, 4] = 1
        tp_im = np.array([[1, 0, 1, 0, 1]], dtype=np.uint32)
        tp1 = tp_im != 0

        tp_im, tp1, short_pruned, capacity_pruned = _split_discontinuous_tracks_ram(
            mask_stack=mask_stack,
            tp_im=tp_im,
            tp1=tp1,
            time_series_threshold=1,
            max_track_id=2,
        )

        self.assertEqual(short_pruned, 0)
        self.assertEqual(capacity_pruned, 1)
        self.assertEqual(tp_im.shape[0], 2)
        self.assertEqual(mask_stack.reshape(-1).tolist(), [1, 0, 2, 0, 0])

    def test_ram_discontinuous_split_reuses_sparse_label_slots(self):
        mask_stack = np.zeros((1, 1, 5), dtype=np.uint16)
        mask_stack[:, :, 0] = 5
        mask_stack[:, :, 2] = 5
        mask_stack[:, :, 4] = 5
        tp_im = np.zeros((5, 5), dtype=np.uint32)
        tp_im[4, [0, 2, 4]] = 1
        tp1 = tp_im != 0

        tp_im, tp1, short_pruned, capacity_pruned = _split_discontinuous_tracks_ram(
            mask_stack=mask_stack,
            tp_im=tp_im,
            tp1=tp1,
            time_series_threshold=1,
            max_track_id=5,
        )

        self.assertEqual(short_pruned, 0)
        self.assertEqual(capacity_pruned, 0)
        self.assertEqual(tp_im.shape[0], 5)
        self.assertEqual(mask_stack.reshape(-1).tolist(), [5, 0, 1, 0, 2])

    def test_compact_labels_in_place_removes_sparse_raw_ids(self):
        mask_stack = np.zeros((2, 2, 2), dtype=np.uint32)
        mask_stack[:, :, 0] = np.array([[0, 10], [70000, 0]], dtype=np.uint32)
        mask_stack[:, :, 1] = np.array([[10, 0], [0, 42]], dtype=np.uint32)

        label_count, max_label_before = _compact_labels_in_place(mask_stack)

        self.assertEqual(label_count, 3)
        self.assertEqual(max_label_before, 70000)
        np.testing.assert_array_equal(
            mask_stack[:, :, 0],
            np.array([[0, 1], [3, 0]], dtype=np.uint32),
        )
        np.testing.assert_array_equal(
            mask_stack[:, :, 1],
            np.array([[1, 0], [0, 2]], dtype=np.uint32),
        )

    def test_ram_fast_compaction_matches_standalone_compaction(self):
        original = np.zeros((2, 3, 3), dtype=np.uint32)
        original[:, :, 0] = np.array([[0, 10, 10], [70000, 0, 42]], dtype=np.uint32)
        original[:, :, 1] = np.array([[42, 0, 0], [10, 99, 0]], dtype=np.uint32)
        original[:, :, 2] = np.array([[0, 99, 70000], [0, 0, 0]], dtype=np.uint32)
        standalone_stack = original.copy()
        ram_stack = original.copy()

        standalone_result = _compact_labels_in_place(standalone_stack)
        ram_result = _compact_labels_in_place_fast(ram_stack)

        self.assertEqual(ram_result, standalone_result)
        np.testing.assert_array_equal(ram_stack, standalone_stack)

    def _division_stack(self, frame_count=3):
        stack = np.zeros((12, 12, frame_count), dtype=np.uint16)
        stack[2:8, 2:8, 0] = 1
        stack[2:5, 2:8, 1:] = 1
        stack[5:8, 2:8, 1:] = 2
        return stack

    def _division_stack_with_disappeared_mother(self, frame_count=2):
        stack = np.zeros((12, 12, frame_count), dtype=np.uint16)
        stack[2:8, 2:8, 0] = 1
        stack[2:5, 2:8, 1:] = 2
        stack[5:8, 2:8, 1:] = 3
        return stack

    def _uneven_division_stack_with_disappeared_mother(self, frame_count=2):
        stack = np.zeros((12, 12, frame_count), dtype=np.uint16)
        stack[2:10, 2:10, 0] = 1
        stack[2:8, 2:10, 1:] = 2
        stack[8:10, 2:10, 1:] = 3
        return stack

    def _division_stack_with_reused_daughter_label(self, frame_count=2):
        stack = np.zeros((12, 12, frame_count), dtype=np.uint16)
        stack[2:8, 2:8, 0] = 1
        stack[9:11, 9:11, 0] = 2
        stack[2:5, 2:8, 1:] = 2
        stack[5:8, 2:8, 1:] = 3
        return stack

    def _assert_ram_normalize_matches_standalone(self, stack, division_cooldown_frames=20):
        standalone_normalized, standalone_parent_map = _normalize_ctc_divisions(
            stack.copy(),
            division_cooldown_frames=division_cooldown_frames,
        )
        ram_normalized, ram_parent_map = _normalize_ctc_divisions_ram(
            stack.copy(),
            division_cooldown_frames=division_cooldown_frames,
        )

        self.assertEqual(ram_parent_map, standalone_parent_map)
        np.testing.assert_array_equal(ram_normalized, standalone_normalized)

    def test_normalize_ctc_divisions_creates_parent_rows(self):
        stack = self._division_stack(frame_count=2)

        normalized, parent_map = _normalize_ctc_divisions(stack, division_cooldown_frames=20)

        self.assertEqual(parent_map, {2: 1, 3: 1})
        self.assertEqual(set(np.unique(normalized[:, :, 1]).tolist()), {0, 2, 3})
        self.assertFalse(np.any(normalized[:, :, 1] == 1))

    def test_normalize_ctc_divisions_handles_two_newborn_daughters(self):
        stack = self._division_stack_with_disappeared_mother(frame_count=2)

        normalized, parent_map = _normalize_ctc_divisions(stack, division_cooldown_frames=20)

        self.assertEqual(parent_map, {2: 1, 3: 1})
        self.assertEqual(set(np.unique(normalized[:, :, 1]).tolist()), {0, 2, 3})
        self.assertFalse(np.any(normalized[:, :, 1] == 1))

    def test_ram_normalize_ctc_divisions_handles_two_newborn_daughters(self):
        stack = self._division_stack_with_disappeared_mother(frame_count=2)

        normalized, parent_map = _normalize_ctc_divisions_ram(stack, division_cooldown_frames=20)

        self.assertEqual(parent_map, {2: 1, 3: 1})
        self.assertEqual(set(np.unique(normalized[:, :, 1]).tolist()), {0, 2, 3})
        self.assertFalse(np.any(normalized[:, :, 1] == 1))

    def test_ram_normalize_ctc_divisions_keeps_uneven_daughters_new(self):
        stack = self._uneven_division_stack_with_disappeared_mother(frame_count=2)

        normalized, parent_map = _normalize_ctc_divisions_ram(stack, division_cooldown_frames=20)

        daughter_ids = set(parent_map)
        self.assertEqual(set(parent_map.values()), {1})
        self.assertEqual(len(daughter_ids), 2)
        self.assertEqual(set(np.unique(normalized[:, :, 1]).tolist()), daughter_ids | {0})
        self.assertFalse(np.any(normalized[:, :, 1] == 1))

    def test_normalize_ctc_divisions_forks_reused_daughter_label(self):
        stack = self._division_stack_with_reused_daughter_label(frame_count=2)

        normalized, parent_map = _normalize_ctc_divisions(stack, division_cooldown_frames=20)

        self.assertEqual(parent_map, {3: 1, 4: 1})
        self.assertEqual(set(np.unique(normalized[:, :, 0]).tolist()), {0, 1, 2})
        self.assertEqual(set(np.unique(normalized[:, :, 1]).tolist()), {0, 3, 4})
        self.assertFalse(np.any(normalized[:, :, 1] == 1))
        self.assertFalse(np.any(normalized[:, :, 1] == 2))

    def test_ram_normalize_ctc_divisions_forks_reused_daughter_label(self):
        stack = self._division_stack_with_reused_daughter_label(frame_count=2)

        normalized, parent_map = _normalize_ctc_divisions_ram(stack, division_cooldown_frames=20)

        self.assertEqual(set(parent_map.values()), {1})
        self.assertEqual(len(parent_map), 2)
        self.assertEqual(set(np.unique(normalized[:, :, 0]).tolist()), {0, 1, 2})
        self.assertEqual(set(np.unique(normalized[:, :, 1]).tolist()), set(parent_map) | {0})
        self.assertFalse(np.any(normalized[:, :, 1] == 1))
        self.assertFalse(np.any(normalized[:, :, 1] == 2))

    def test_ram_normalize_ctc_divisions_matches_standalone_general_label_swap(self):
        stack = np.zeros((8, 8, 3), dtype=np.uint16)
        stack[2:6, 2:6, 0] = 1
        stack[2:6, 2:6, 1] = 1
        stack[2:6, 2:6, 2] = 2

        self._assert_ram_normalize_matches_standalone(stack, division_cooldown_frames=20)

    def test_ram_normalize_ctc_divisions_matches_standalone_cooldown_zero(self):
        stack = self._division_stack(frame_count=3)
        stack[5:8, 2:8, 2] = 0
        stack[5:8, 2:8, 2] = 4

        self._assert_ram_normalize_matches_standalone(stack, division_cooldown_frames=0)

    def test_ram_normalize_ctc_divisions_matches_standalone_daughter_cooldown_rescue(self):
        stack = self._division_stack(frame_count=3)
        stack[5:8, 2:8, 2] = 0
        stack[5:8, 2:8, 2] = 4

        self._assert_ram_normalize_matches_standalone(stack, division_cooldown_frames=20)

    def test_division_cooldown_rescues_daughter_label_swap(self):
        stack = self._division_stack(frame_count=3)
        stack[5:8, 2:8, 2] = 0
        stack[5:8, 2:8, 2] = 4

        normalized, parent_map = _normalize_ctc_divisions(stack, division_cooldown_frames=20)

        self.assertEqual(parent_map, {2: 1, 5: 1})
        self.assertTrue(np.any(normalized[:, :, 2] == 2))
        self.assertFalse(np.any(normalized[:, :, 2] == 4))

    def test_division_cooldown_blocks_protected_daughter_as_new_parent(self):
        stack = self._division_stack(frame_count=3)
        stack[7:10, 2:8, 2] = 4

        normalized, parent_map = _normalize_ctc_divisions(stack, division_cooldown_frames=20)

        self.assertNotIn(4, parent_map)
        self.assertTrue(np.any(normalized[:, :, 2] == 4))
        self.assertEqual(parent_map, {2: 1, 5: 1})

    def test_division_cooldown_does_not_merge_ambiguous_daughter_swaps(self):
        stack = self._division_stack(frame_count=3)
        stack[5:8, 2:8, 2] = 0
        stack[5:8, 2:5, 2] = 4
        stack[5:8, 5:8, 2] = 5

        normalized, parent_map = _normalize_ctc_divisions(stack, division_cooldown_frames=20)

        self.assertTrue(np.any(normalized[:, :, 2] == 4))
        self.assertTrue(np.any(normalized[:, :, 2] == 5))
        self.assertFalse(np.any(normalized[:, :, 2] == 2))
        self.assertEqual(parent_map, {2: 1, 6: 1})

    def test_division_cooldown_allows_repeat_split_after_expiration(self):
        stack = self._division_stack(frame_count=4)
        stack[7:10, 2:8, 3] = 4

        normalized, parent_map = _normalize_ctc_divisions(stack, division_cooldown_frames=1)

        self.assertEqual(parent_map.get(4), 2)
        self.assertEqual(parent_map.get(2), 1)
        self.assertTrue(np.any(normalized[:, :, 3] == 4))
        self.assertFalse(np.any(normalized[:, :, 3] == 2))

    def test_division_cooldown_zero_preserves_unrescued_label_swap(self):
        stack = self._division_stack(frame_count=3)
        stack[5:8, 2:8, 2] = 0
        stack[5:8, 2:8, 2] = 4

        normalized, parent_map = _normalize_ctc_divisions(stack, division_cooldown_frames=0)

        self.assertEqual(parent_map, {2: 1, 5: 1})
        self.assertTrue(np.any(normalized[:, :, 2] == 4))
        self.assertFalse(np.any(normalized[:, :, 2] == 2))

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

    def _write_source_frames(self, source_root: Path, sequence: str, frame_count: int, shape=(6, 6)):
        image_dir = source_root / sequence
        image_dir.mkdir(parents=True)
        for frame_idx in range(frame_count):
            tifffile.imwrite(image_dir / f"t{frame_idx:03d}.tif", np.zeros(shape, dtype=np.uint16))

    def test_temporal_downsample_selects_every_16th_frame_and_rebuilds_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            input_dir = root / "01_interp_RES"
            output_dir = root / "eval" / "01_RES"
            self._write_source_frames(source, "01", frame_count=3)
            input_dir.mkdir()

            for frame_idx in range(33):
                mask = np.zeros((6, 6), dtype=np.uint16)
                if frame_idx in {0, 16, 32}:
                    mask[1:3, 1:3] = 7
                if frame_idx in {16, 32}:
                    mask[3:5, 3:5] = 13
                tifffile.imwrite(input_dir / f"mask{frame_idx:03d}.tif", mask)
            (input_dir / "res_track.txt").write_text("7 0 32 0\n13 16 32 7\n", encoding="utf-8")

            report = temporal_downsample_ctc_results(
                input_result_dir=input_dir,
                output_result_dir=output_dir,
                source_root=source,
                sequence="01",
                factor=16,
                offset=0,
            )

            self.assertEqual(report["frames"], 3)
            self.assertEqual(report["tracks"], 2)
            self.assertEqual([path.name for path in sorted(output_dir.glob("mask*.tif"))], [
                "mask000.tif",
                "mask001.tif",
                "mask002.tif",
            ])
            self.assertEqual(
                (output_dir / "res_track.txt").read_text(encoding="utf-8"),
                "1 0 2 0\n2 1 2 0\n",
            )
            np.testing.assert_array_equal(tifffile.imread(output_dir / "mask000.tif")[1:3, 1:3], np.full((2, 2), 1))
            np.testing.assert_array_equal(tifffile.imread(output_dir / "mask001.tif")[3:5, 3:5], np.full((2, 2), 2))

            validation = validate_ctc_result_format(
                dataset_root=output_dir.parent,
                source_root=source,
                sequence="01",
                digits_arg="auto",
            )
            self.assertEqual(validation["frames"], 3)
            self.assertEqual(validation["tracks"], 2)

    def test_temporal_downsample_splits_discontinuous_sampled_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            input_dir = root / "01_interp_RES"
            output_dir = root / "eval" / "01_RES"
            self._write_source_frames(source, "01", frame_count=3)
            input_dir.mkdir()

            for frame_idx in [0, 16, 32]:
                mask = np.zeros((6, 6), dtype=np.uint16)
                if frame_idx in {0, 32}:
                    mask[1:3, 1:3] = 7
                tifffile.imwrite(input_dir / f"mask{frame_idx:03d}.tif", mask)
            (input_dir / "res_track.txt").write_text("7 0 32 0\n", encoding="utf-8")

            temporal_downsample_ctc_results(
                input_result_dir=input_dir,
                output_result_dir=output_dir,
                source_root=source,
                sequence="01",
                factor=16,
                offset=0,
            )

            self.assertEqual(
                (output_dir / "res_track.txt").read_text(encoding="utf-8"),
                "1 0 0 0\n2 2 2 0\n",
            )
            self.assertEqual(set(np.unique(tifffile.imread(output_dir / "mask000.tif")).tolist()), {0, 1})
            self.assertEqual(set(np.unique(tifffile.imread(output_dir / "mask001.tif")).tolist()), {0})
            self.assertEqual(set(np.unique(tifffile.imread(output_dir / "mask002.tif")).tolist()), {0, 2})

            validation = validate_ctc_result_format(
                dataset_root=output_dir.parent,
                source_root=source,
                sequence="01",
                digits_arg="auto",
            )
            self.assertEqual(validation["tracks"], 2)

    def test_temporal_downsample_accepts_source_frame_count_without_source_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "01_interp_RES"
            output_dir = root / "eval" / "01_RES"
            input_dir.mkdir()

            for frame_idx in [0, 16, 32]:
                mask = np.zeros((6, 6), dtype=np.uint16)
                mask[1:3, 1:3] = 7
                tifffile.imwrite(input_dir / f"mask{frame_idx:03d}.tif", mask)
            (input_dir / "res_track.txt").write_text("7 0 32 0\n", encoding="utf-8")

            report = temporal_downsample_ctc_results(
                input_result_dir=input_dir,
                output_result_dir=output_dir,
                source_root=None,
                sequence="01",
                source_frame_count=3,
                factor=16,
                offset=0,
            )

            self.assertEqual(report["frames"], 3)
            self.assertEqual(
                (output_dir / "res_track.txt").read_text(encoding="utf-8"),
                "1 0 2 0\n",
            )

    def test_temporal_downsample_can_resize_outputs_to_target_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "01_interp_RES"
            output_dir = root / "eval" / "01_RES"
            input_dir.mkdir()

            for frame_idx in [0, 16, 32]:
                mask = np.zeros((8, 8), dtype=np.uint16)
                mask[2:6, 2:6] = 7
                tifffile.imwrite(input_dir / f"mask{frame_idx:03d}.tif", mask)
            (input_dir / "res_track.txt").write_text("7 0 32 0\n", encoding="utf-8")

            temporal_downsample_ctc_results(
                input_result_dir=input_dir,
                output_result_dir=output_dir,
                source_root=None,
                sequence="01",
                source_frame_count=3,
                target_shape=(4, 4),
                factor=16,
                offset=0,
            )

            resized = tifffile.imread(output_dir / "mask000.tif")
            self.assertEqual(resized.shape, (4, 4))
            self.assertEqual(resized.dtype, np.uint16)
            self.assertEqual(set(np.unique(resized).tolist()), {0, 1})

    def test_temporal_downsample_can_pad_missing_selected_frames_with_empty_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "01_interp_RES"
            output_dir = root / "eval" / "01_RES"
            input_dir.mkdir()

            for frame_idx in [0, 16]:
                mask = np.zeros((8, 8), dtype=np.uint16)
                mask[2:6, 2:6] = 7
                tifffile.imwrite(input_dir / f"mask{frame_idx:03d}.tif", mask)
            (input_dir / "res_track.txt").write_text("7 0 16 0\n", encoding="utf-8")

            report = temporal_downsample_ctc_results(
                input_result_dir=input_dir,
                output_result_dir=output_dir,
                source_root=None,
                sequence="01",
                source_frame_count=4,
                target_shape=(4, 4),
                pad_missing_with_empty=True,
                factor=16,
                offset=0,
            )

            self.assertEqual(report["missing_selected_frames"], 2)
            self.assertEqual([path.name for path in sorted(output_dir.glob("mask*.tif"))], [
                "mask000.tif",
                "mask001.tif",
                "mask002.tif",
                "mask003.tif",
            ])
            self.assertEqual(set(np.unique(tifffile.imread(output_dir / "mask002.tif")).tolist()), {0})
            self.assertEqual(tifffile.imread(output_dir / "mask002.tif").shape, (4, 4))

    def test_temporal_downsample_preserves_valid_parent_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            input_dir = root / "01_interp_RES"
            output_dir = root / "eval" / "01_RES"
            self._write_source_frames(source, "01", frame_count=3)
            input_dir.mkdir()

            for frame_idx in [0, 16, 32]:
                mask = np.zeros((6, 6), dtype=np.uint16)
                if frame_idx == 0:
                    mask[1:3, 1:3] = 1
                if frame_idx in {16, 32}:
                    mask[3:5, 3:5] = 2
                tifffile.imwrite(input_dir / f"mask{frame_idx:03d}.tif", mask)
            (input_dir / "res_track.txt").write_text("1 0 15 0\n2 16 32 1\n", encoding="utf-8")

            temporal_downsample_ctc_results(
                input_result_dir=input_dir,
                output_result_dir=output_dir,
                source_root=source,
                sequence="01",
                factor=16,
                offset=0,
            )

            self.assertEqual(
                (output_dir / "res_track.txt").read_text(encoding="utf-8"),
                "1 0 0 0\n2 1 2 1\n",
            )
            validate_ctc_result_format(
                dataset_root=output_dir.parent,
                source_root=source,
                sequence="01",
                digits_arg="auto",
            )

    def test_pipeline_temporal_downsample_dry_run_uses_intermediate_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            eval_root = root / "eval"
            log_dir = root / "logs"
            source.mkdir()

            args = SimpleNamespace(
                python="python",
                source_root=source,
                eval_root=eval_root,
                log_dir=log_dir,
                stage_gt="none",
                dry_run=True,
                skip_tracking=False,
                skip_validation=True,
                skip_evaluation=True,
                skip_failure_analysis=True,
                ctc_software_dir=None,
                mask_dir_template=None,
                mask_pattern="mask*.tif",
                time_series_threshold=1,
                output_digits="auto",
                digits="auto",
                temporal_downsample_factor=16,
                temporal_downsample_offset=0,
                strict_matlab_id_matching=True,
                metrics=["TRA", "SEG", "DET"],
                det_penalize_extra_detections=None,
            )

            self.assertEqual(_tracking_position_for_sequence("01", 16), "01_interp")
            _run_sequence(args, Path(__file__).resolve().parent, "01")

            tracking_log = (log_dir / "01_tracking.log").read_text(encoding="utf-8")
            downsample_log = (log_dir / "01_temporal_downsample.log").read_text(encoding="utf-8")
            self.assertIn("--position 01_interp", tracking_log)
            self.assertIn("temporal_downsample_ctc_results.py", downsample_log)
            self.assertIn(str(eval_root / "01_interp_RES"), downsample_log)
            self.assertIn(str(eval_root / "01_RES"), downsample_log)

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

    def test_lineage_window_filters_and_clips_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            track_file = Path(tmp) / "res_track.txt"
            track_file.write_text("1 0 10 0\n2 12 20 1\n3 30 40 0\n", encoding="utf-8")

            track_rows = _parse_track_file(track_file)
            focused_rows = _filter_lineage_track_rows(track_rows, start_frame=10, end_frame=15)
            focused_layout = _build_lineage_layout(focused_rows)

            self.assertEqual(set(focused_rows), {1, 2})

            plot_data = _lineage_plot_segments(
                focused_rows,
                focused_layout,
                current_frame=15,
                x_start=10,
                x_end=15,
                reveal_until_frame=15,
            )

            segments_by_track = {segment["track_id"]: segment for segment in plot_data["tracks"]}
            self.assertEqual(set(segments_by_track), {1, 2})
            self.assertEqual((segments_by_track[1]["x0"], segments_by_track[1]["x1"]), (10, 10))
            self.assertEqual((segments_by_track[2]["x0"], segments_by_track[2]["x1"]), (12, 15))
            self.assertEqual(
                {(segment["parent_id"], segment["child_id"]) for segment in plot_data["connectors"]},
                {(1, 2)},
            )
            self.assertEqual([point["track_id"] for point in plot_data["active"]], [2])

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

    def test_resize_mask_array_preserves_integer_labels(self):
        mask = np.zeros((8, 8), dtype=np.uint16)
        mask[0:4, 0:4] = 7
        mask[4:8, 4:8] = 13

        resized = resize_mask_array(mask, scale=0.5)

        self.assertEqual(resized.shape, (4, 4))
        self.assertEqual(resized.dtype, np.uint16)
        self.assertEqual(set(np.unique(resized).tolist()), {0, 7, 13})

    def test_rescale_dataset_and_overlay_quality_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            mask_dir = root / "masks"
            scaled_image_dir = root / "scaled_images"
            scaled_mask_dir = root / "scaled_masks"
            qa_dir = root / "qa"
            image_dir.mkdir()
            mask_dir.mkdir()

            for frame_idx in range(2):
                image = np.zeros((8, 8), dtype=np.uint16)
                image[2:6, 2:6] = 100 + frame_idx
                mask = np.zeros((8, 8), dtype=np.uint16)
                mask[0:4, 0:4] = 1
                mask[4:8, 4:8] = 2
                tifffile.imwrite(image_dir / f"t{frame_idx:03d}.tif", image)
                tifffile.imwrite(mask_dir / f"mask{frame_idx:03d}.tif", mask)

            (mask_dir / "res_track.txt").write_text("1 0 1 0\n2 0 1 0\n", encoding="utf-8")

            image_results, mask_results, copied_track_file = rescale_dataset(
                image_dir=image_dir,
                mask_dir=mask_dir,
                output_image_dir=scaled_image_dir,
                output_mask_dir=scaled_mask_dir,
                scale=0.5,
            )

            self.assertEqual(len(image_results), 2)
            self.assertEqual(len(mask_results), 2)
            self.assertTrue(copied_track_file.is_file())
            self.assertEqual(tifffile.imread(scaled_image_dir / "t000.tif").shape, (4, 4))
            scaled_mask = tifffile.imread(scaled_mask_dir / "mask000.tif")
            self.assertEqual(scaled_mask.shape, (4, 4))
            self.assertEqual(set(np.unique(scaled_mask).tolist()), {0, 1, 2})

            output_paths, csv_path, metrics_rows = export_rescale_overlay_comparisons(
                original_image_files=sorted(image_dir.glob("*.tif")),
                original_mask_files=sorted(mask_dir.glob("mask*.tif")),
                scaled_image_files=sorted(scaled_image_dir.glob("*.tif")),
                scaled_mask_files=sorted(scaled_mask_dir.glob("mask*.tif")),
                output_dir=qa_dir,
                scale=0.5,
                max_frames=1,
            )

            self.assertEqual(len(output_paths), 1)
            self.assertTrue(output_paths[0].is_file())
            self.assertTrue(csv_path.is_file())
            self.assertEqual(metrics_rows[0]["lost_labels"], "")


if __name__ == "__main__":
    unittest.main()
