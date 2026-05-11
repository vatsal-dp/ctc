#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio

from identify_segmentation_break_frames import (
    BreakEvent,
    find_break_events,
    load_all_ob_matrix,
    unique_review_frames,
)


class IdentifySegmentationBreakFramesTests(unittest.TestCase):
    def test_finds_temporary_gaps_and_final_stops_before_last_frame(self):
        all_ob = np.array(
            [
                [5, 5, 0, 0, 0, 0],
                [0, 2, 2, 0, 3, 0],
                [1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint32,
        )

        events = find_break_events(all_ob)

        self.assertEqual(
            events,
            [
                BreakEvent(
                    track_row_0_based=0,
                    break_frame_0_based=2,
                    last_nonzero_frame_0_based=1,
                    zero_run_end_0_based=5,
                    next_nonzero_frame_0_based=None,
                    size_before_break=5,
                    event_type="final_stop",
                ),
                BreakEvent(
                    track_row_0_based=1,
                    break_frame_0_based=3,
                    last_nonzero_frame_0_based=2,
                    zero_run_end_0_based=3,
                    next_nonzero_frame_0_based=4,
                    size_before_break=2,
                    event_type="temporary_gap",
                ),
            ],
        )

    def test_can_include_drop_to_zero_on_last_frame_when_requested(self):
        all_ob = np.array([[4, 4, 0]], dtype=np.uint32)

        self.assertEqual(find_break_events(all_ob), [])
        events = find_break_events(all_ob, include_last_frame=True)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].break_frame_0_based, 2)
        self.assertEqual(events[0].event_type, "final_stop")

    def test_unique_review_frames_are_sorted(self):
        events = [
            BreakEvent(2, 8, 7, 9, None, 10, "final_stop"),
            BreakEvent(4, 3, 2, 3, 4, 12, "temporary_gap"),
            BreakEvent(5, 8, 7, 8, 10, 11, "temporary_gap"),
        ]

        self.assertEqual(unique_review_frames(events), [3, 8])

    def test_loads_all_ob_from_mat_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            mat_path = Path(tmp) / "tracks.mat"
            expected = np.array([[1, 0, 2], [3, 3, 0]], dtype=np.float64)
            sio.savemat(mat_path, {"all_ob": expected})

            loaded = load_all_ob_matrix(mat_path)

        np.testing.assert_array_equal(loaded.matrix, expected)
        self.assertEqual(loaded.name, "all_ob")
        self.assertEqual(loaded.source_path, mat_path)


if __name__ == "__main__":
    unittest.main()
