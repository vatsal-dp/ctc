#!/usr/bin/env python3
"""Find cell tracks whose object-size row drops to zero before the movie ends.

The script looks for an ``all_ob``-style matrix where rows are tracks/cells and
columns are time points. It writes two files:

* ``segmentation_break_events.csv``: one row per nonzero-to-zero transition.
* ``segmentation_break_review_frames.csv``: unique frames to inspect.
"""

import argparse
import csv
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PREFERRED_MATRIX_NAMES = (
    "all_ob",
    "all_obj_sizes",
    "all_obj_size",
    "all_object_sizes",
    "all_object_size",
    "object_sizes",
    "obj_sizes",
    "cell_sizes",
)
IMAGE_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class BreakEvent:
    track_row_0_based: int
    break_frame_0_based: int
    last_nonzero_frame_0_based: int
    zero_run_end_0_based: int
    next_nonzero_frame_0_based: int | None
    size_before_break: int | float
    event_type: str


@dataclass(frozen=True)
class LoadedMatrix:
    matrix: np.ndarray
    name: str
    source_path: Path


@dataclass(frozen=True)
class EventDecision:
    event: BreakEvent
    review_required: bool
    review_category: str
    replacement_track_id: int | None = None
    evidence: str = ""


@dataclass(frozen=True)
class _LabelStats:
    area: int
    centroid_y: float
    centroid_x: float


def _native_number(value: Any) -> int | float:
    scalar = np.asarray(value).item()
    if isinstance(scalar, (np.integer, int)):
        return int(scalar)
    if isinstance(scalar, (np.floating, float)):
        as_float = float(scalar)
        return int(as_float) if as_float.is_integer() else as_float
    return scalar


def _natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _as_numeric_2d_matrix(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        return None
    return arr


def _iter_named_arrays(obj: Any, prefix: str = ""):
    matrix = _as_numeric_2d_matrix(obj)
    if matrix is not None and prefix:
        yield prefix, matrix

    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).startswith("__"):
                continue
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_named_arrays(value, name)
        return

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            name = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_named_arrays(value, name)
        return

    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size <= 100:
        for idx, value in np.ndenumerate(obj):
            name = f"{prefix}{idx}" if prefix else str(idx)
            yield from _iter_named_arrays(value, name)
        return

    if hasattr(obj, "_fieldnames"):
        for field in getattr(obj, "_fieldnames", []) or []:
            name = f"{prefix}.{field}" if prefix else field
            yield from _iter_named_arrays(getattr(obj, field), name)


def _candidate_score(name: str, variable: str | None) -> tuple[int, str]:
    lowered = name.lower()
    leaf = lowered.rsplit(".", 1)[-1].rsplit("/", 1)[-1]
    if variable:
        wanted = variable.lower()
        if leaf == wanted or lowered == wanted:
            return (0, name)
        if lowered.endswith("." + wanted) or lowered.endswith("/" + wanted):
            return (1, name)
        return (1000, name)

    for idx, preferred in enumerate(PREFERRED_MATRIX_NAMES):
        if leaf == preferred:
            return (idx, name)
    if "all_ob" in lowered:
        return (50, name)
    if ("obj" in lowered or "object" in lowered or "cell" in lowered) and "size" in lowered:
        return (75, name)
    return (1000, name)


def _choose_matrix(candidates: list[tuple[str, np.ndarray]], variable: str | None, source_path: Path) -> LoadedMatrix:
    usable = [
        (score, name, matrix)
        for name, matrix in candidates
        if (score := _candidate_score(name, variable))[0] < 1000
    ]
    if not usable:
        available = ", ".join(name for name, _ in candidates[:20]) or "none"
        wanted = variable or "one of " + ", ".join(PREFERRED_MATRIX_NAMES)
        raise ValueError(f"Could not find {wanted} in {source_path}. Available 2D numeric arrays: {available}")

    usable.sort(key=lambda item: item[0])
    _, name, matrix = usable[0]
    return LoadedMatrix(matrix=np.asarray(matrix), name=name, source_path=source_path)


def _file_is_all_zero(path: Path) -> bool:
    if path.stat().st_size == 0:
        return True
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return True
            if any(chunk):
                return False


def _load_mat(path: Path, variable: str | None) -> LoadedMatrix:
    if _file_is_all_zero(path):
        raise ValueError(f"{path} contains only zero bytes; it is not a valid MATLAB file.")
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise RuntimeError("Reading .mat files requires scipy. Install this repo's requirements first.") from exc

    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return _load_h5(path, variable)
    return _choose_matrix(list(_iter_named_arrays(data)), variable, path)


def _load_h5(path: Path, variable: str | None) -> LoadedMatrix:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Reading .h5 or MATLAB v7.3 files requires h5py. "
            "Install it on the lab PC with: python -m pip install h5py"
        ) from exc

    candidates: list[tuple[str, np.ndarray]] = []
    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: Any):
            if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
                return
            if name.startswith("#refs#"):
                return
            if len(obj.shape) != 2:
                return
            if not np.issubdtype(obj.dtype, np.number):
                return
            candidates.append((name, np.asarray(obj[()])))

        handle.visititems(visit)
    return _choose_matrix(candidates, variable, path)


def _load_pickle(path: Path, variable: str | None) -> LoadedMatrix:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return _choose_matrix(list(_iter_named_arrays(data)), variable, path)


def _candidate_files(folder: Path) -> list[Path]:
    patterns = (
        "*Tracks_MATLAB*.mat",
        "*Tracks*.h5",
        "*.mat",
        "*.h5",
        "*.pkl",
        "*.pickle",
    )
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        for path in sorted(folder.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    return files


def load_all_ob_matrix(source: Path | str, variable: str | None = None) -> LoadedMatrix:
    source_path = Path(source)
    if source_path.is_dir():
        failures: list[str] = []
        for candidate in _candidate_files(source_path):
            try:
                return load_all_ob_matrix(candidate, variable=variable)
            except Exception as exc:
                failures.append(f"{candidate.name}: {exc}")
        details = "\n  ".join(failures) if failures else "No .mat, .h5, .pkl, or .pickle files found."
        raise ValueError(f"Could not load an all_ob matrix from {source_path}.\n  {details}")

    suffix = source_path.suffix.lower()
    if suffix == ".mat":
        return _load_mat(source_path, variable)
    if suffix in {".h5", ".hdf5"}:
        return _load_h5(source_path, variable)
    if suffix in {".pkl", ".pickle"}:
        return _load_pickle(source_path, variable)
    raise ValueError(f"Unsupported input file type: {source_path}")


def resolve_mask_dir(source: Path | str, mask_dir: Path | str | None = None) -> Path | None:
    if mask_dir is not None:
        path = Path(mask_dir)
        return path if path.is_dir() else None
    source_path = Path(source)
    base = source_path if source_path.is_dir() else source_path.parent
    candidates = [base] if base.name == "tracked_masks" else [base / "tracked_masks"]
    for mask_dir in candidates:
        if mask_dir.is_dir():
            return mask_dir
    return None


def _count_tracked_mask_frames(source: Path | str) -> int | None:
    mask_dir = resolve_mask_dir(source)
    if mask_dir is not None:
        if not mask_dir.is_dir():
            return None
        count = sum(1 for path in mask_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if count > 0:
            return count
    return None


def orient_loaded_matrix(
    loaded: LoadedMatrix,
    *,
    source: Path | str,
    orientation: str = "auto",
) -> LoadedMatrix:
    if orientation == "tracks-rows":
        return loaded
    if orientation == "frames-rows":
        return LoadedMatrix(matrix=loaded.matrix.T, name=loaded.name + ".T", source_path=loaded.source_path)
    if orientation != "auto":
        raise ValueError("orientation must be one of: auto, tracks-rows, frames-rows")

    expected_frame_count = _count_tracked_mask_frames(source)
    if expected_frame_count is None:
        return loaded

    rows, cols = loaded.matrix.shape
    if rows == expected_frame_count and cols != expected_frame_count:
        return LoadedMatrix(matrix=loaded.matrix.T, name=loaded.name + ".T", source_path=loaded.source_path)
    return loaded


def find_break_events(
    matrix: np.ndarray,
    *,
    include_last_frame: bool = False,
    mode: str = "all",
    first_per_row: bool = False,
) -> list[BreakEvent]:
    all_ob = np.asarray(matrix)
    if all_ob.ndim != 2:
        raise ValueError(f"all_ob must be 2D with rows=tracks and columns=frames; got shape={all_ob.shape}")
    if mode not in {"all", "final-stops", "temporary-gaps"}:
        raise ValueError("mode must be one of: all, final-stops, temporary-gaps")

    frame_count = all_ob.shape[1]
    events: list[BreakEvent] = []
    for row_idx in range(all_ob.shape[0]):
        row = all_ob[row_idx]
        present = np.asarray(row > 0, dtype=bool)
        if not present.any() or frame_count < 2:
            continue

        zero_starts = np.flatnonzero(present[:-1] & ~present[1:]) + 1
        for zero_start in zero_starts:
            if zero_start == frame_count - 1 and not include_last_frame:
                continue

            zero_end = int(zero_start)
            while zero_end + 1 < frame_count and not present[zero_end + 1]:
                zero_end += 1

            next_nonzero = zero_end + 1 if zero_end + 1 < frame_count and present[zero_end + 1] else None
            event_type = "temporary_gap" if next_nonzero is not None else "final_stop"
            if mode == "final-stops" and event_type != "final_stop":
                continue
            if mode == "temporary-gaps" and event_type != "temporary_gap":
                continue

            events.append(
                BreakEvent(
                    track_row_0_based=int(row_idx),
                    break_frame_0_based=int(zero_start),
                    last_nonzero_frame_0_based=int(zero_start - 1),
                    zero_run_end_0_based=int(zero_end),
                    next_nonzero_frame_0_based=None if next_nonzero is None else int(next_nonzero),
                    size_before_break=_native_number(row[zero_start - 1]),
                    event_type=event_type,
                )
            )
            if first_per_row:
                break

    return events


def unique_review_frames(events: list[BreakEvent]) -> list[int]:
    return sorted({event.break_frame_0_based for event in events})


class _MaskFrameReader:
    def __init__(self, mask_dir: Path):
        self.mask_dir = mask_dir
        self.files = sorted(
            [path for path in mask_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
            key=lambda path: _natural_sort_key(path.name),
        )
        self._frame_cache: dict[int, np.ndarray] = {}
        self._stats_cache: dict[int, dict[int, _LabelStats]] = {}

    def read(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= len(self.files):
            raise IndexError(f"Frame {frame_idx} is outside tracked mask range 0..{len(self.files) - 1}.")
        cached = self._frame_cache.get(frame_idx)
        if cached is not None:
            return cached
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError("Mask-based ID-change filtering requires tifffile.") from exc
        frame = np.asarray(tifffile.imread(self.files[frame_idx]))
        if frame.ndim != 2:
            frame = np.squeeze(frame)
        if frame.ndim != 2:
            raise ValueError(f"Tracked mask {self.files[frame_idx]} is not 2D after squeeze; shape={frame.shape}")
        self._frame_cache[frame_idx] = frame
        if len(self._frame_cache) > 8:
            oldest = next(iter(self._frame_cache))
            if oldest != frame_idx:
                self._frame_cache.pop(oldest, None)
        return frame

    def stats(self, frame_idx: int) -> dict[int, _LabelStats]:
        cached = self._stats_cache.get(frame_idx)
        if cached is not None:
            return cached
        frame = self.read(frame_idx)
        ys, xs = np.nonzero(frame)
        if ys.size == 0:
            self._stats_cache[frame_idx] = {}
            return {}
        labels = frame[ys, xs].astype(np.int64, copy=False)
        unique_labels, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        sum_y = np.bincount(inverse, weights=ys, minlength=len(unique_labels))
        sum_x = np.bincount(inverse, weights=xs, minlength=len(unique_labels))
        stats = {
            int(label): _LabelStats(
                area=int(count),
                centroid_y=float(sum_y[idx] / count),
                centroid_x=float(sum_x[idx] / count),
            )
            for idx, (label, count) in enumerate(zip(unique_labels, counts))
            if int(label) != 0
        }
        self._stats_cache[frame_idx] = stats
        if len(self._stats_cache) > 8:
            oldest = next(iter(self._stats_cache))
            if oldest != frame_idx:
                self._stats_cache.pop(oldest, None)
        return stats


def _area_ratio_ok(area_a: int, area_b: int, max_area_ratio: float) -> bool:
    if area_a <= 0 or area_b <= 0:
        return False
    ratio = area_a / area_b
    return (1 / max_area_ratio) <= ratio <= max_area_ratio


def _distance_px(a: _LabelStats, b: _LabelStats) -> float:
    return float(((a.centroid_y - b.centroid_y) ** 2 + (a.centroid_x - b.centroid_x) ** 2) ** 0.5)


def _substantial_previous_labels(
    previous_frame: np.ndarray,
    current_label_mask: np.ndarray,
    current_area: int,
    min_current_fraction: float,
) -> set[int]:
    previous_values = previous_frame[current_label_mask]
    if previous_values.size == 0:
        return set()
    labels, counts = np.unique(previous_values, return_counts=True)
    return {
        int(label)
        for label, count in zip(labels, counts)
        if int(label) != 0 and current_area > 0 and (int(count) / current_area) >= min_current_fraction
    }


def _classify_break_event_with_masks(
    event: BreakEvent,
    reader: _MaskFrameReader,
    *,
    min_overlap_fraction: float,
    min_reciprocal_overlap_fraction: float,
    max_area_ratio: float,
    max_centroid_distance: float,
) -> EventDecision:
    old_track_id = event.track_row_0_based + 1
    try:
        previous_frame = reader.read(event.last_nonzero_frame_0_based)
        current_frame = reader.read(event.break_frame_0_based)
        previous_stats = reader.stats(event.last_nonzero_frame_0_based)
        current_stats = reader.stats(event.break_frame_0_based)
    except Exception as exc:
        return EventDecision(event, True, "no_mask_evidence", evidence=str(exc))

    old_stats = previous_stats.get(old_track_id)
    if old_stats is None:
        return EventDecision(
            event,
            True,
            "no_previous_mask_for_track",
            evidence=f"track_id={old_track_id} absent in previous mask frame",
        )
    if old_track_id in current_stats:
        return EventDecision(
            event,
            True,
            "all_ob_mask_mismatch",
            evidence=f"track_id={old_track_id} still present in break mask frame",
        )

    old_mask = previous_frame == old_track_id
    labels, counts = np.unique(current_frame[old_mask], return_counts=True)
    overlap_candidates: list[tuple[int, int]] = []
    weak_replacements: list[int] = []
    for label_value, count_value in zip(labels, counts):
        label = int(label_value)
        count = int(count_value)
        if label == 0 or label == old_track_id:
            continue
        current_label_stats = current_stats.get(label)
        if current_label_stats is None:
            continue
        old_overlap = count / old_stats.area
        current_overlap = count / current_label_stats.area
        if (
            old_overlap >= min_overlap_fraction
            and current_overlap >= min_reciprocal_overlap_fraction
            and _area_ratio_ok(current_label_stats.area, old_stats.area, max_area_ratio)
        ):
            overlap_candidates.append((label, count))
        elif old_overlap > 0 or current_overlap > 0:
            weak_replacements.append(label)

    if len(overlap_candidates) > 1:
        return EventDecision(
            event,
            True,
            "possible_split_or_merge",
            evidence="old track overlaps multiple current labels: "
            + ";".join(str(label) for label, _ in overlap_candidates),
        )

    if len(overlap_candidates) == 1:
        replacement_label = overlap_candidates[0][0]
        if replacement_label in previous_stats:
            return EventDecision(
                event,
                True,
                "possible_merge_or_existing_label_takeover",
                replacement_track_id=replacement_label,
                evidence=f"replacement label {replacement_label} already existed in previous frame",
            )
        current_label_mask = current_frame == replacement_label
        contributors = _substantial_previous_labels(
            previous_frame,
            current_label_mask,
            current_stats[replacement_label].area,
            min_reciprocal_overlap_fraction,
        )
        if contributors <= {old_track_id}:
            return EventDecision(
                event,
                False,
                "id_change",
                replacement_track_id=replacement_label,
                evidence=f"one-to-one overlap replacement {old_track_id}->{replacement_label}",
            )
        return EventDecision(
            event,
            True,
            "possible_merge_or_existing_label_takeover",
            replacement_track_id=replacement_label,
            evidence="replacement label has multiple previous contributors: "
            + ";".join(str(label) for label in sorted(contributors)),
        )

    centroid_candidates: list[tuple[float, int]] = []
    for label, stats in current_stats.items():
        if label == old_track_id or label in previous_stats:
            continue
        if not _area_ratio_ok(stats.area, old_stats.area, max_area_ratio):
            continue
        distance = _distance_px(old_stats, stats)
        if distance <= max_centroid_distance:
            centroid_candidates.append((distance, label))
    centroid_candidates.sort()
    if len(centroid_candidates) == 1:
        replacement_label = centroid_candidates[0][1]
        return EventDecision(
            event,
            False,
            "id_change",
            replacement_track_id=replacement_label,
            evidence=(
                f"one-to-one centroid replacement {old_track_id}->{replacement_label}; "
                f"distance_px={centroid_candidates[0][0]:.2f}"
            ),
        )
    if len(centroid_candidates) > 1:
        return EventDecision(
            event,
            True,
            "ambiguous_nearby_replacements",
            evidence="nearby new labels: " + ";".join(str(label) for _, label in centroid_candidates),
        )
    if weak_replacements:
        return EventDecision(
            event,
            True,
            "weak_spatial_replacement",
            evidence="weak overlapping labels: " + ";".join(str(label) for label in sorted(set(weak_replacements))),
        )
    return EventDecision(
        event,
        True,
        "missing_detection_candidate",
        evidence=f"no replacement label for track_id={old_track_id} at break frame",
    )


def classify_break_events_with_masks(
    events: list[BreakEvent],
    *,
    mask_dir: Path | str,
    min_overlap_fraction: float = 0.35,
    min_reciprocal_overlap_fraction: float = 0.35,
    max_area_ratio: float = 2.0,
    max_centroid_distance: float = 25.0,
) -> list[EventDecision]:
    reader = _MaskFrameReader(Path(mask_dir))
    return [
        _classify_break_event_with_masks(
            event,
            reader,
            min_overlap_fraction=min_overlap_fraction,
            min_reciprocal_overlap_fraction=min_reciprocal_overlap_fraction,
            max_area_ratio=max_area_ratio,
            max_centroid_distance=max_centroid_distance,
        )
        for event in events
    ]


def decisions_from_events(events: list[BreakEvent]) -> list[EventDecision]:
    return [
        EventDecision(event, True, "all_ob_break", evidence="mask-based ID-change filtering was not used")
        for event in events
    ]


def reviewable_events(decisions: list[EventDecision]) -> list[BreakEvent]:
    return [decision.event for decision in decisions if decision.review_required]


def _one_based(value: int | None) -> int | str:
    return "" if value is None else value + 1


def write_break_events_csv(path: Path, loaded: LoadedMatrix, decisions: list[EventDecision]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_file",
                "matrix_name",
                "matrix_rows_tracks",
                "matrix_cols_frames",
                "track_row_0_based",
                "track_row_1_based",
                "break_frame_0_based",
                "break_frame_1_based",
                "last_nonzero_frame_0_based",
                "last_nonzero_frame_1_based",
                "zero_run_end_0_based",
                "zero_run_end_1_based",
                "zero_run_length",
                "reappears_after_break",
                "next_nonzero_frame_0_based",
                "next_nonzero_frame_1_based",
                "size_before_break",
                "event_type",
                "review_required",
                "review_category",
                "replacement_track_id",
                "evidence",
            ],
        )
        writer.writeheader()
        for decision in decisions:
            event = decision.event
            writer.writerow(
                {
                    "source_file": str(loaded.source_path),
                    "matrix_name": loaded.name,
                    "matrix_rows_tracks": loaded.matrix.shape[0],
                    "matrix_cols_frames": loaded.matrix.shape[1],
                    "track_row_0_based": event.track_row_0_based,
                    "track_row_1_based": event.track_row_0_based + 1,
                    "break_frame_0_based": event.break_frame_0_based,
                    "break_frame_1_based": event.break_frame_0_based + 1,
                    "last_nonzero_frame_0_based": event.last_nonzero_frame_0_based,
                    "last_nonzero_frame_1_based": event.last_nonzero_frame_0_based + 1,
                    "zero_run_end_0_based": event.zero_run_end_0_based,
                    "zero_run_end_1_based": event.zero_run_end_0_based + 1,
                    "zero_run_length": event.zero_run_end_0_based - event.break_frame_0_based + 1,
                    "reappears_after_break": event.next_nonzero_frame_0_based is not None,
                    "next_nonzero_frame_0_based": "" if event.next_nonzero_frame_0_based is None else event.next_nonzero_frame_0_based,
                    "next_nonzero_frame_1_based": _one_based(event.next_nonzero_frame_0_based),
                    "size_before_break": event.size_before_break,
                    "event_type": event.event_type,
                    "review_required": decision.review_required,
                    "review_category": decision.review_category,
                    "replacement_track_id": "" if decision.replacement_track_id is None else decision.replacement_track_id,
                    "evidence": decision.evidence,
                }
            )


def write_review_frames_csv(path: Path, events: list[BreakEvent]) -> None:
    by_frame: dict[int, list[BreakEvent]] = {}
    for event in events:
        by_frame.setdefault(event.break_frame_0_based, []).append(event)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_frame_0_based",
                "review_frame_1_based",
                "event_count",
                "final_stop_count",
                "temporary_gap_count",
                "track_rows_1_based",
            ],
        )
        writer.writeheader()
        for frame in sorted(by_frame):
            frame_events = by_frame[frame]
            writer.writerow(
                {
                    "review_frame_0_based": frame,
                    "review_frame_1_based": frame + 1,
                    "event_count": len(frame_events),
                    "final_stop_count": sum(event.event_type == "final_stop" for event in frame_events),
                    "temporary_gap_count": sum(event.event_type == "temporary_gap" for event in frame_events),
                    "track_rows_1_based": ";".join(str(event.track_row_0_based + 1) for event in frame_events),
                }
            )


def write_review_frames_txt(path: Path, events: list[BreakEvent]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 0-based frame numbers, matching mask000.tif naming\n")
        for frame in unique_review_frames(events):
            handle.write(f"{frame}\n")


def _default_output_dir(source: Path) -> Path:
    return (source if source.is_dir() else source.parent) / "segmentation_break_review"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an all_ob/object-size matrix and list frames where a track's "
            "cell size drops from nonzero to zero."
        )
    )
    parser.add_argument("source", type=Path, help="Folder or .mat/.h5/.pkl file containing all_ob.")
    parser.add_argument("--variable", default=None, help="Matrix variable/dataset name. Default: prefer all_ob.")
    parser.add_argument(
        "--mode",
        choices=["all", "final-stops", "temporary-gaps"],
        default="all",
        help="Which nonzero-to-zero events to report. Default: all.",
    )
    parser.add_argument(
        "--first-per-row",
        action="store_true",
        help="Only report the first qualifying break for each track row.",
    )
    parser.add_argument(
        "--include-last-frame",
        action="store_true",
        help="Also report drops to zero on the final frame. Default skips them.",
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Deprecated alias for --orientation frames-rows.",
    )
    parser.add_argument(
        "--orientation",
        choices=["auto", "tracks-rows", "frames-rows"],
        default="auto",
        help=(
            "How the matrix is oriented. Default auto uses tracked_masks frame count; "
            "frames-rows transposes so rows become tracks and columns become frames."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write CSV/TXT outputs. Default: SOURCE/segmentation_break_review.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Tracked mask folder used to filter simple ID renumbering. Default: SOURCE/tracked_masks.",
    )
    parser.add_argument(
        "--no-mask-id-filter",
        action="store_true",
        help="Do not use tracked masks to exclude one-to-one ID renumbering events.",
    )
    parser.add_argument(
        "--id-change-min-overlap",
        type=float,
        default=0.35,
        help="Minimum old-mask overlap fraction for an ID-change replacement. Default: 0.35.",
    )
    parser.add_argument(
        "--id-change-min-reciprocal-overlap",
        type=float,
        default=0.35,
        help="Minimum replacement-mask overlap fraction for an ID-change replacement. Default: 0.35.",
    )
    parser.add_argument(
        "--id-change-max-area-ratio",
        type=float,
        default=2.0,
        help="Largest allowed area ratio between old and replacement masks for ID-change filtering. Default: 2.0.",
    )
    parser.add_argument(
        "--id-change-max-centroid-distance",
        type=float,
        default=25.0,
        help="Fallback max centroid distance in pixels for ID-change filtering. Default: 25.",
    )
    parser.add_argument("--prefix", default="segmentation_break", help="Output filename prefix.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        loaded = load_all_ob_matrix(args.source, variable=args.variable)
        original_name = loaded.name
        original_shape = loaded.matrix.shape
        if args.transpose:
            args.orientation = "frames-rows"
        loaded = orient_loaded_matrix(loaded, source=args.source, orientation=args.orientation)

        events = find_break_events(
            loaded.matrix,
            include_last_frame=args.include_last_frame,
            mode=args.mode,
            first_per_row=args.first_per_row,
        )
        mask_dir = None if args.no_mask_id_filter else resolve_mask_dir(args.source, args.mask_dir)
        if mask_dir is not None:
            decisions = classify_break_events_with_masks(
                events,
                mask_dir=mask_dir,
                min_overlap_fraction=args.id_change_min_overlap,
                min_reciprocal_overlap_fraction=args.id_change_min_reciprocal_overlap,
                max_area_ratio=args.id_change_max_area_ratio,
                max_centroid_distance=args.id_change_max_centroid_distance,
            )
        else:
            decisions = decisions_from_events(events)
        review_events = reviewable_events(decisions)

        output_dir = args.output_dir or _default_output_dir(args.source)
        output_dir.mkdir(parents=True, exist_ok=True)
        events_csv = output_dir / f"{args.prefix}_events.csv"
        frames_csv = output_dir / f"{args.prefix}_review_frames.csv"
        frames_txt = output_dir / f"{args.prefix}_review_frames_0_based.txt"

        write_break_events_csv(events_csv, loaded, decisions)
        write_review_frames_csv(frames_csv, review_events)
        write_review_frames_txt(frames_txt, review_events)

        final_stops = sum(event.event_type == "final_stop" for event in events)
        temporary_gaps = sum(event.event_type == "temporary_gap" for event in events)
        excluded_id_changes = sum(decision.review_category == "id_change" for decision in decisions)
        print(f"Loaded matrix: {loaded.name} from {loaded.source_path}")
        if loaded.name != original_name or loaded.matrix.shape != original_shape:
            print(f"Matrix orientation: transposed from {original_shape} to {loaded.matrix.shape}")
        else:
            expected_frames = _count_tracked_mask_frames(args.source)
            if expected_frames is not None:
                print(f"Matrix orientation: kept as loaded; tracked_masks frame count={expected_frames}")
        print(f"Matrix shape: rows/tracks={loaded.matrix.shape[0]} cols/frames={loaded.matrix.shape[1]}")
        if mask_dir is not None:
            print(f"Mask ID-change filter: enabled using {mask_dir}")
        else:
            print("Mask ID-change filter: disabled or tracked_masks folder not found")
        print(
            f"Found {len(events)} break events; excluded {excluded_id_changes} simple ID changes; "
            f"review list has {len(review_events)} events across {len(unique_review_frames(review_events))} unique frames "
            f"(final_stops={final_stops}, temporary_gaps={temporary_gaps})."
        )
        print(f"Wrote: {events_csv}")
        print(f"Wrote: {frames_csv}")
        print(f"Wrote: {frames_txt}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
