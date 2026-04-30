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


def _build_track_runs(label_frames: dict[int, list[int]]):
    rows: list[TrackRun] = []
    split_labels = 0
    for old_label in sorted(label_frames):
        runs = _contiguous_runs(label_frames[old_label])
        if len(runs) > 1:
            split_labels += 1
        for begin, end in runs:
            new_label = len(rows) + 1
            if new_label > MAX_UINT16:
                raise ValueError("Generated track count exceeds uint16 capacity.")
            rows.append(TrackRun(old_label=old_label, label=new_label, begin=begin, end=end))
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


def export_ctc_result(frames: list[np.ndarray], output_dir: Path, output_digits: str, overwrite: bool, dry_run: bool):
    if not frames:
        raise ValueError("No frames to export.")
    frame_count = len(frames)
    digits = _parse_output_digits(output_digits, frame_count)
    label_frames = _scan_label_frames(frames)
    rows, split_labels = _build_track_runs(label_frames)
    frame_maps = _build_frame_maps(rows, frame_count)

    print(
        "[EXPORT] "
        f"frames={frame_count} original_labels={len(label_frames)} ctc_tracks={len(rows)} "
        f"labels_split_for_gaps={split_labels} digits={digits}",
        flush=True,
    )
    print("[EXPORT] parent IDs are written as 0 because no lineage metadata was provided.", flush=True)

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
        export_ctc_result(
            frames=frames,
            output_dir=args.output_result_dir,
            output_digits=args.output_digits,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"[EXPORT] FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
