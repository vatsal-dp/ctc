#!/usr/bin/env python3

import argparse
import pickle
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


IMAGE_SUFFIXES = {".tif", ".tiff"}
MAX_UINT16 = int(np.iinfo(np.uint16).max)
PREFERRED_PICKLE_KEYS = (
    "Final_tracked_tensor",
    "tracked_masks",
    "tracked_mask",
    "track_masks",
    "mask_stack",
    "label_stack",
    "labels",
    "masks",
    "tracks",
    "seg",
    "data",
    "array",
    "arr",
)


@dataclass(frozen=True)
class TrackRun:
    old_label: int
    label: int
    begin: int
    end: int
    parent: int = 0


def _natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _minimum_digit_width(frame_count: int) -> int:
    return max(3, len(str(max(frame_count - 1, 0))))


def _parse_output_digits(value: str, frame_count: int) -> int:
    required = _minimum_digit_width(frame_count)
    if value == "auto":
        return required
    try:
        digits = int(value)
    except ValueError as exc:
        raise ValueError("--output-digits must be 'auto' or a positive integer.") from exc
    if digits < 1:
        raise ValueError("--output-digits must be a positive integer.")
    if frame_count > 10**digits:
        raise ValueError(
            f"Cannot write {frame_count} frames with {digits} digits. "
            f"Use --output-digits {required} or higher."
        )
    return digits


def _as_2d_integer_mask(mask, source: str):
    arr = np.asarray(mask)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{source} is not a 2D mask after squeeze; shape={arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.floating) and np.all(np.isfinite(arr)) and np.all(arr == np.floor(arr)):
            arr = arr.astype(np.int64, copy=False)
        else:
            raise ValueError(f"{source} has non-integer dtype {arr.dtype}.")
    if arr.size and int(arr.min()) < 0:
        raise ValueError(f"{source} contains negative labels.")
    if arr.size and int(arr.max()) > MAX_UINT16:
        raise ValueError(
            f"{source} has label {int(arr.max())}, above uint16 capacity. "
            "Relabel before CTC evaluation."
        )
    return arr


def _read_mask_dir(mask_dir: Path, pattern: str):
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"Input mask folder does not exist: {mask_dir}")
    files = sorted(mask_dir.glob(pattern), key=lambda path: _natural_sort_key(path.name))
    files = [path for path in files if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No TIFF masks found in {mask_dir} using pattern {pattern!r}.")
    frames = [_as_2d_integer_mask(tifffile.imread(str(path)), str(path)) for path in files]
    return frames


def _describe_pickle_object(obj, depth: int = 0, max_items: int = 8):
    indent = "  " * depth
    if isinstance(obj, np.ndarray):
        return [f"{indent}ndarray shape={obj.shape} dtype={obj.dtype}"]
    if isinstance(obj, dict):
        lines = [f"{indent}dict keys={list(obj.keys())[:max_items]}"]
        for key in list(obj.keys())[:max_items]:
            lines.extend(_describe_pickle_object(obj[key], depth + 1, max_items=3))
        return lines
    if isinstance(obj, (list, tuple)):
        lines = [f"{indent}{type(obj).__name__} len={len(obj)}"]
        for item in list(obj[:max_items]):
            lines.extend(_describe_pickle_object(item, depth + 1, max_items=3))
        return lines
    return [f"{indent}{type(obj).__name__}"]


def _find_array_like(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        lowered = {str(key).lower(): key for key in obj}
        for preferred in PREFERRED_PICKLE_KEYS:
            key = lowered.get(preferred.lower())
            if key is not None:
                try:
                    return _find_array_like(obj[key])
                except ValueError:
                    pass
        for value in obj.values():
            try:
                return _find_array_like(value)
            except ValueError:
                continue
    if isinstance(obj, (list, tuple)):
        if obj and all(np.asarray(item).ndim in {2, 3} for item in obj):
            return obj
        for value in obj:
            try:
                return _find_array_like(value)
            except ValueError:
                continue
    raise ValueError("Could not find a mask stack array/list in the pickle.")


def _time_axis_value(value: str, ndim: int):
    aliases = {"first": 0, "last": ndim - 1}
    if value in aliases:
        return aliases[value]
    try:
        axis = int(value)
    except ValueError as exc:
        raise ValueError("--time-axis must be auto, first, last, or an axis number.") from exc
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"--time-axis {value!r} is invalid for an array with {ndim} dimensions.")
    return axis


def _infer_time_axis(shape: tuple[int, ...]):
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D mask stack, got shape={shape}.")
    smallest_axis = int(np.argmin(shape))
    other_sizes = [size for axis, size in enumerate(shape) if axis != smallest_axis]
    if shape[smallest_axis] * 2 <= min(other_sizes):
        return smallest_axis
    return 2


def _frames_from_array_like(array_like, time_axis: str):
    if isinstance(array_like, (list, tuple)):
        frames = [_as_2d_integer_mask(item, f"pickle frame {idx}") for idx, item in enumerate(array_like)]
        if not frames:
            raise ValueError("Pickle list/tuple is empty.")
        return frames

    arr = np.asarray(array_like)
    if arr.ndim == 2:
        return [_as_2d_integer_mask(arr, "pickle array")]
    if arr.ndim != 3:
        raise ValueError(f"Pickle array must be 2D or 3D, got shape={arr.shape}.")

    axis = _infer_time_axis(tuple(arr.shape)) if time_axis == "auto" else _time_axis_value(time_axis, arr.ndim)
    moved = np.moveaxis(arr, axis, 0)
    frames = [_as_2d_integer_mask(moved[idx], f"pickle frame {idx}") for idx in range(moved.shape[0])]
    print(f"[EXPORT] pickle stack shape={arr.shape} dtype={arr.dtype} time_axis={axis}", flush=True)
    return frames


def _read_pickle(path: Path, time_axis: str, inspect_only: bool):
    with path.open("rb") as handle:
        obj = pickle.load(handle)

    if inspect_only:
        for line in _describe_pickle_object(obj):
            print(line)
        return []

    try:
        array_like = _find_array_like(obj)
    except ValueError as exc:
        print("[EXPORT] Pickle structure:", file=sys.stderr)
        for line in _describe_pickle_object(obj):
            print(line, file=sys.stderr)
        raise ValueError(
            f"{exc} Re-run with --inspect-only and share the printed structure if this pickle stores masks differently."
        ) from exc

    return _frames_from_array_like(array_like, time_axis=time_axis)


def _read_input(input_path: Path, pattern: str, time_axis: str, inspect_only: bool):
    if input_path.is_dir():
        if inspect_only:
            print(f"{input_path} is a directory; inspect-only is only useful for pickle files.")
            return []
        return _read_mask_dir(input_path, pattern)
    if input_path.suffix.lower() in {".pkl", ".pickle"}:
        return _read_pickle(input_path, time_axis=time_axis, inspect_only=inspect_only)
    raise ValueError(f"Unsupported input type: {input_path}. Use a mask folder or .pkl file.")


def _resize_nearest_label_mask(frame: np.ndarray, output_shape: tuple[int, int]):
    out_height, out_width = output_shape
    if out_height <= 0 or out_width <= 0:
        raise ValueError("--resize-spatial dimensions must be positive integers.")

    in_height, in_width = frame.shape
    if (in_height, in_width) == (out_height, out_width):
        return frame

    row_coords = (np.arange(out_height) + 0.5) * (in_height / out_height) - 0.5
    col_coords = (np.arange(out_width) + 0.5) * (in_width / out_width) - 0.5
    row_idx = np.clip(np.rint(row_coords).astype(np.int64), 0, in_height - 1)
    col_idx = np.clip(np.rint(col_coords).astype(np.int64), 0, in_width - 1)
    return np.ascontiguousarray(frame[row_idx[:, None], col_idx[None, :]])


def _transform_frames(
    frames: list[np.ndarray],
    transpose_spatial: bool,
    resize_spatial: tuple[int, int] | None,
):
    if not transpose_spatial:
        transformed = frames
    else:
        print("[EXPORT] transposing each mask frame from (Y, X) to (X, Y)", flush=True)
        transformed = [np.ascontiguousarray(frame.T) for frame in frames]

    if resize_spatial is None:
        return transformed

    print(
        f"[EXPORT] resizing each mask frame to {resize_spatial[0]}x{resize_spatial[1]} with nearest neighbor",
        flush=True,
    )
    return [_resize_nearest_label_mask(frame, resize_spatial) for frame in transformed]


def _parse_resize_spatial(value: list[int] | None):
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("--resize-spatial expects HEIGHT WIDTH.")
    height, width = value
    if height <= 0 or width <= 0:
        raise ValueError("--resize-spatial dimensions must be positive integers.")
    return height, width


def _contiguous_runs(frames: list[int]):
    if not frames:
        return []
    runs = []
    start = previous = frames[0]
    for frame_idx in frames[1:]:
        if frame_idx == previous + 1:
            previous = frame_idx
            continue
        runs.append((start, previous))
        start = previous = frame_idx
    runs.append((start, previous))
    return runs


def _track_ids_in_frame(frame: np.ndarray):
    ids = np.unique(frame)
    return ids[ids != 0].astype(int)


def _binary_dilation_3x3(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for row_offset in range(3):
        for col_offset in range(3):
            out |= padded[row_offset : row_offset + mask.shape[0], col_offset : col_offset + mask.shape[1]]
    return out


def _infer_parent_id(
    prev_frame: np.ndarray,
    child_mask: np.ndarray,
    child_track_id: int,
    valid_track_ids: set[int],
    min_touch_pixels: int,
    min_touch_ratio: float,
) -> int:
    child_pixels = int(np.count_nonzero(child_mask))
    if child_pixels == 0:
        return 0

    dilated_child = _binary_dilation_3x3(child_mask)
    touching_labels = prev_frame[dilated_child]
    touching_labels = touching_labels[touching_labels != 0]
    if touching_labels.size == 0:
        return 0

    labels, counts = np.unique(touching_labels.astype(np.int64, copy=False), return_counts=True)
    candidates = []
    for label_val, count in zip(labels.tolist(), counts.tolist()):
        parent_id = int(label_val)
        if parent_id == child_track_id or parent_id not in valid_track_ids:
            continue
        candidates.append((int(count), parent_id))

    if not candidates:
        return 0

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_touch_count, best_parent_id = candidates[0]
    if best_touch_count < min_touch_pixels:
        return 0
    if (best_touch_count / child_pixels) < min_touch_ratio:
        return 0
    return best_parent_id


def _has_self_continuity(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    track_id: int,
    min_touch_pixels: int,
    min_touch_ratio: float,
) -> bool:
    previous_mask = prev_frame == track_id
    current_mask = curr_frame == track_id
    current_pixels = int(np.count_nonzero(current_mask))
    if current_pixels == 0 or not np.any(previous_mask):
        return False

    dilated_previous = _binary_dilation_3x3(previous_mask)
    touch_pixels = int(np.count_nonzero(current_mask & dilated_previous))
    if touch_pixels < min_touch_pixels:
        return False
    return (touch_pixels / current_pixels) >= min_touch_ratio


def _fork_existing_daughter_label(
    frames: list[np.ndarray],
    child_id: int,
    frame_idx: int,
    new_child_id: int,
) -> None:
    for time_idx in range(frame_idx, len(frames)):
        frame_t = frames[time_idx]
        child_pixels = frame_t == child_id
        if np.any(child_pixels):
            frame_t[child_pixels] = new_child_id


def _active_cooldown_track_ids(cooldown_until: dict[int, int], frame_idx: int):
    return {track_id for track_id, end_frame in cooldown_until.items() if frame_idx <= end_frame}


def _rescue_cooldown_label_swaps(
    frames: list[np.ndarray],
    frame_idx: int,
    newborn_ids: list[int],
    protected_track_ids: set[int],
):
    prev_frame = frames[frame_idx - 1]
    curr_frame = frames[frame_idx]
    prev_ids = set(_track_ids_in_frame(prev_frame).tolist())
    curr_ids = set(_track_ids_in_frame(curr_frame).tolist())
    rescued_ids = set()

    for protected_id in sorted(protected_track_ids):
        if protected_id not in prev_ids or protected_id in curr_ids:
            continue

        previous_mask = prev_frame == protected_id
        if not np.any(previous_mask):
            continue

        dilated_previous = _binary_dilation_3x3(previous_mask)
        candidates = []
        for newborn_id in newborn_ids:
            if newborn_id in rescued_ids:
                continue
            current_mask = curr_frame == newborn_id
            touch_pixels = int(np.count_nonzero(current_mask & dilated_previous))
            if touch_pixels > 0:
                candidates.append((touch_pixels, newborn_id))

        if len(candidates) != 1:
            continue

        _, rescued_id = candidates[0]
        for time_idx in range(frame_idx, len(frames)):
            frame_t = frames[time_idx]
            rescue_pixels = frame_t == rescued_id
            if np.any(rescue_pixels):
                frame_t[rescue_pixels] = protected_id

        rescued_ids.add(rescued_id)
        curr_ids.discard(rescued_id)
        curr_ids.add(protected_id)

    return sorted(track_id for track_id in newborn_ids if track_id not in rescued_ids)


def _normalize_ctc_divisions(
    frames: list[np.ndarray],
    division_cooldown_frames: int,
    min_touch_pixels: int,
    min_touch_ratio: float,
):
    if division_cooldown_frames < 0:
        raise ValueError("--division-cooldown-frames must be >= 0.")

    max_existing_id = max((int(np.max(frame)) for frame in frames if frame.size), default=0)
    if max_existing_id > MAX_UINT16:
        raise ValueError("Track IDs exceed uint16 capacity required for CTC export.")

    frame_count = len(frames)
    max_track_id = max_existing_id
    parent_map: dict[int, int] = {}
    cooldown_until: dict[int, int] = {}

    for frame_idx in range(1, frame_count):
        if frame_idx % 250 == 0 or frame_idx == frame_count - 1:
            print(f"[EXPORT] division inference frame {frame_idx + 1}/{frame_count}", flush=True)

        prev_frame = frames[frame_idx - 1]
        curr_frame = frames[frame_idx]
        prev_ids = set(_track_ids_in_frame(prev_frame).tolist())
        if not prev_ids:
            continue

        curr_ids = _track_ids_in_frame(curr_frame).tolist()
        newborn_ids = sorted(track_id for track_id in curr_ids if track_id not in prev_ids)
        protected_track_ids = (
            _active_cooldown_track_ids(cooldown_until, frame_idx)
            if division_cooldown_frames > 0
            else set()
        )

        if protected_track_ids and newborn_ids:
            newborn_ids = _rescue_cooldown_label_swaps(
                frames=frames,
                frame_idx=frame_idx,
                newborn_ids=newborn_ids,
                protected_track_ids=protected_track_ids,
            )
            prev_frame = frames[frame_idx - 1]
            curr_frame = frames[frame_idx]
            prev_ids = set(_track_ids_in_frame(prev_frame).tolist())
            curr_ids = _track_ids_in_frame(curr_frame).tolist()
            newborn_ids = sorted(track_id for track_id in curr_ids if track_id not in prev_ids)

        mother_to_children: dict[int, list[int]] = {}
        newborn_id_set = set(newborn_ids)
        valid_parent_ids = prev_ids - protected_track_ids

        for child_id in curr_ids:
            if child_id in prev_ids and _has_self_continuity(
                prev_frame,
                curr_frame,
                child_id,
                min_touch_pixels=min_touch_pixels,
                min_touch_ratio=min_touch_ratio,
            ):
                continue
            child_mask = curr_frame == child_id
            mother_id = _infer_parent_id(
                prev_frame=prev_frame,
                child_mask=child_mask,
                child_track_id=child_id,
                valid_track_ids=valid_parent_ids,
                min_touch_pixels=min_touch_pixels,
                min_touch_ratio=min_touch_ratio,
            )
            if mother_id != 0:
                mother_to_children.setdefault(mother_id, []).append(child_id)

        if not mother_to_children:
            continue

        curr_id_set = set(curr_ids)
        for mother_id in sorted(mother_to_children):
            child_ids = sorted(set(mother_to_children[mother_id]))
            daughter_ids = []

            if mother_id in curr_id_set:
                max_track_id += 1
                if max_track_id > MAX_UINT16:
                    raise ValueError("Division inference would exceed uint16 track ID capacity.")
                continuation_daughter_id = max_track_id

                for time_idx in range(frame_idx, frame_count):
                    frame_t = frames[time_idx]
                    mother_pixels = frame_t == mother_id
                    if np.any(mother_pixels):
                        frame_t[mother_pixels] = continuation_daughter_id

                parent_map[continuation_daughter_id] = mother_id
                daughter_ids.append(continuation_daughter_id)
                curr_id_set.discard(mother_id)
                curr_id_set.add(continuation_daughter_id)
            elif len(child_ids) < 2:
                continue

            for child_id in child_ids:
                if child_id in newborn_id_set:
                    daughter_id = child_id
                else:
                    max_track_id += 1
                    if max_track_id > MAX_UINT16:
                        raise ValueError("Division inference would exceed uint16 track ID capacity.")
                    daughter_id = max_track_id
                    _fork_existing_daughter_label(frames, child_id, frame_idx, daughter_id)
                    curr_id_set.discard(child_id)
                    curr_id_set.add(daughter_id)
                parent_map[daughter_id] = mother_id
                daughter_ids.append(daughter_id)

            if division_cooldown_frames > 0:
                cooldown_end = frame_idx + division_cooldown_frames
                for daughter_id in daughter_ids:
                    cooldown_until[daughter_id] = cooldown_end

    return frames, parent_map


def _scan_label_frames(frames: list[np.ndarray]):
    label_frames: dict[int, list[int]] = {}
    reference_shape = tuple(frames[0].shape)
    for frame_idx, frame in enumerate(frames):
        if tuple(frame.shape) != reference_shape:
            raise ValueError(f"Frame {frame_idx} shape {frame.shape} differs from first frame {reference_shape}.")
        labels = np.unique(frame)
        labels = labels[labels != 0].astype(int, copy=False)
        for label in labels.tolist():
            label_frames.setdefault(label, []).append(frame_idx)
    return label_frames


def _build_track_runs(label_frames: dict[int, list[int]], parent_map: dict[int, int] | None = None):
    parent_map = parent_map or {}
    rows_without_parents: list[TrackRun] = []
    rows_by_old_label: dict[int, list[TrackRun]] = {}
    split_labels = 0
    for old_label in sorted(label_frames):
        runs = _contiguous_runs(label_frames[old_label])
        if len(runs) > 1:
            split_labels += 1
        for begin, end in runs:
            new_label = len(rows_without_parents) + 1
            if new_label > MAX_UINT16:
                raise ValueError("Generated track count exceeds uint16 capacity.")
            row = TrackRun(old_label=old_label, label=new_label, begin=begin, end=end)
            rows_without_parents.append(row)
            rows_by_old_label.setdefault(old_label, []).append(row)

    rows: list[TrackRun] = []
    for row in rows_without_parents:
        parent = 0
        parent_old_label = int(parent_map.get(row.old_label, 0))
        if parent_old_label != 0 and row.begin > 0:
            parent_candidates = [
                candidate
                for candidate in rows_by_old_label.get(parent_old_label, [])
                if candidate.end < row.begin
            ]
            if parent_candidates:
                parent = max(parent_candidates, key=lambda candidate: (candidate.end, candidate.label)).label
                if parent == row.label:
                    parent = 0
        rows.append(
            TrackRun(
                old_label=row.old_label,
                label=row.label,
                begin=row.begin,
                end=row.end,
                parent=parent,
            )
        )
    return rows, split_labels


def _build_frame_maps(rows: list[TrackRun], frame_count: int):
    frame_maps: list[dict[int, int]] = [{} for _ in range(frame_count)]
    for row in rows:
        for frame_idx in range(row.begin, row.end + 1):
            frame_maps[frame_idx][row.old_label] = row.label
    return frame_maps


def _prepare_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists and is not empty. Pass --overwrite to replace it.")
        for path in output_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def export_ctc_result(
    frames: list[np.ndarray],
    output_dir: Path,
    output_digits: str,
    overwrite: bool,
    dry_run: bool,
    infer_divisions: bool = False,
    division_cooldown_frames: int = 20,
    min_touch_pixels: int = 10,
    min_touch_ratio: float = 0.05,
):
    if not frames:
        raise ValueError("No frames to export.")
    frame_count = len(frames)
    digits = _parse_output_digits(output_digits, frame_count)
    parent_map: dict[int, int] = {}
    if infer_divisions:
        print("[EXPORT] inferring CTC-style division parents from tracked masks", flush=True)
        frames, parent_map = _normalize_ctc_divisions(
            frames=frames,
            division_cooldown_frames=division_cooldown_frames,
            min_touch_pixels=min_touch_pixels,
            min_touch_ratio=min_touch_ratio,
        )

    label_frames = _scan_label_frames(frames)
    rows, split_labels = _build_track_runs(label_frames, parent_map=parent_map)
    frame_maps = _build_frame_maps(rows, frame_count)
    parent_rows = sum(1 for row in rows if row.parent != 0)

    print(
        "[EXPORT] "
        f"frames={frame_count} original_labels={len(label_frames)} ctc_tracks={len(rows)} "
        f"parented_tracks={parent_rows} labels_split_for_gaps={split_labels} digits={digits}",
        flush=True,
    )
    if not infer_divisions:
        print("[EXPORT] parent IDs are written as 0 because division inference is disabled.", flush=True)
    elif parent_rows == 0:
        print("[EXPORT] no parent-child division events passed the inference thresholds.", flush=True)

    if dry_run:
        return rows

    _prepare_output_dir(output_dir, overwrite=overwrite)
    with (output_dir / "res_track.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row.label} {row.begin} {row.end} {row.parent}\n")

    for frame_idx, frame in enumerate(frames):
        remapped = np.zeros(frame.shape, dtype=np.uint16)
        for old_label, new_label in frame_maps[frame_idx].items():
            remapped[frame == old_label] = new_label
        tifffile.imwrite(str(output_dir / f"mask{frame_idx:0{digits}d}.tif"), remapped)
        if frame_idx % 250 == 0 or frame_idx == frame_count - 1:
            print(f"[EXPORT] wrote frame {frame_idx + 1}/{frame_count}", flush=True)

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a Cell Tracking Challenge <sequence>_RES folder from tracked label masks "
            "and generate res_track.txt from label lifetimes."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Folder of tracked mask TIFFs, or a .pkl mask stack.")
    parser.add_argument("--output-result-dir", default=None, type=Path, help="Destination folder, e.g. .../02_RES.")
    parser.add_argument("--input-pattern", default="mask*.tif", help="Glob used when --input is a folder.")
    parser.add_argument(
        "--time-axis",
        default="auto",
        help=(
            "Time axis for a 3D pickle array: auto, first, last, 0, 1, or 2. "
            "FILM/Matlab-style stacks are often last."
        ),
    )
    parser.add_argument("--output-digits", default="auto", help="Output mask index width: auto or a positive integer.")
    parser.add_argument(
        "--transpose-spatial",
        action="store_true",
        help="Transpose each output mask frame. Use when RES shape is the reverse of GT shape.",
    )
    parser.add_argument(
        "--resize-spatial",
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        type=int,
        default=None,
        help="Resize each output mask frame with nearest neighbor, e.g. --resize-spatial 1010 1010.",
    )
    parser.add_argument(
        "--infer-divisions",
        action="store_true",
        help="Infer CTC parent IDs from adjacent-frame mask contact and relabel mother continuations as daughters.",
    )
    parser.add_argument(
        "--division-cooldown-frames",
        default=20,
        type=int,
        help="Frames after division where daughter IDs are protected from becoming new mothers.",
    )
    parser.add_argument(
        "--min-touch-pixels",
        default=10,
        type=int,
        help="Minimum dilated-contact pixels needed to assign a parent.",
    )
    parser.add_argument(
        "--min-touch-ratio",
        default=0.05,
        type=float,
        help="Minimum contact pixels divided by child pixels needed to assign a parent.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty output folder.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing output files.")
    parser.add_argument("--inspect-only", action="store_true", help="For pickle input: print structure and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.output_result_dir is None and not args.inspect_only:
            raise ValueError("--output-result-dir is required unless --inspect-only is used.")
        frames = _read_input(
            input_path=args.input,
            pattern=args.input_pattern,
            time_axis=args.time_axis,
            inspect_only=args.inspect_only,
        )
        if args.inspect_only:
            return 0
        frames = _transform_frames(
            frames,
            transpose_spatial=args.transpose_spatial,
            resize_spatial=_parse_resize_spatial(args.resize_spatial),
        )
        export_ctc_result(
            frames=frames,
            output_dir=args.output_result_dir,
            output_digits=args.output_digits,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            infer_divisions=args.infer_divisions,
            division_cooldown_frames=args.division_cooldown_frames,
            min_touch_pixels=args.min_touch_pixels,
            min_touch_ratio=args.min_touch_ratio,
        )
    except Exception as exc:
        print(f"[EXPORT] FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
