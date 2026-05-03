#!/usr/bin/env python3

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from skimage.transform import resize


IMAGE_SUFFIXES = {".tif", ".tiff"}
MAX_UINT16 = int(np.iinfo(np.uint16).max)


@dataclass(frozen=True)
class InputTrackRow:
    label: int
    begin: int
    end: int
    parent: int


@dataclass(frozen=True)
class OutputTrackRow:
    old_label: int
    label: int
    begin: int
    end: int
    parent: int


def _natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _normalize_sequence(sequence: str) -> str:
    text = str(sequence)
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def _minimum_digit_width(frame_count: int) -> int:
    highest_index = max(frame_count - 1, 0)
    return max(3, len(str(highest_index)))


def _parse_positive_int(value: str, option_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{option_name} must be a positive integer.") from exc
    if parsed < 1:
        raise ValueError(f"{option_name} must be a positive integer.")
    return parsed


def _resolve_output_digits(output_digits: str, frame_count: int, source_digits: int | None) -> int:
    required = _minimum_digit_width(frame_count)
    if output_digits == "auto":
        digits = max(required, source_digits or 0)
    else:
        digits = _parse_positive_int(output_digits, "--output-digits")

    if frame_count > 10**digits:
        raise ValueError(
            f"Cannot export {frame_count} frames with {digits} digits. "
            f"Use --output-digits {required} or higher."
        )
    return digits


def _indexed_files(folder: Path, prefix: str):
    regex = re.compile(rf"^{re.escape(prefix)}(\d+)\.tiff?$", flags=re.IGNORECASE)
    indexed: dict[int, Path] = {}
    digit_widths = set()
    bad_names = []

    if not folder.is_dir():
        return indexed, digit_widths

    for path in sorted(folder.iterdir(), key=lambda p: _natural_sort_key(p.name)):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not path.name.lower().startswith(prefix.lower()):
            continue
        match = regex.match(path.name)
        if match is None:
            bad_names.append(path.name)
            continue
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"Duplicate {prefix} file for frame {index}: {indexed[index]} and {path}")
        indexed[index] = path
        digit_widths.add(len(match.group(1)))

    if bad_names:
        preview = ", ".join(bad_names[:10])
        raise ValueError(f"{folder} contains malformed {prefix}*.tif files: {preview}")

    return indexed, digit_widths


def _require_contiguous(indexed: dict[int, Path], label: str, require_zero_start: bool = True):
    if not indexed:
        raise ValueError(f"No {label} files found.")
    observed = sorted(indexed)
    expected_start = 0 if require_zero_start else observed[0]
    expected = list(range(expected_start, observed[-1] + 1))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        preview = ", ".join(str(idx) for idx in missing[:20])
        start_text = "frame 0" if require_zero_start else f"frame {expected_start}"
        raise ValueError(f"{label} files are not contiguous from {start_text}. Missing indices: {preview}")


def _source_frame_indices_and_digits(source_root: Path, sequence: str):
    sequence = _normalize_sequence(sequence)
    candidates = [
        (source_root / sequence, "t", False),
        (source_root / f"{sequence}_GT" / "TRA", "man_track", False),
        (source_root / f"{sequence}_GT" / "SEG", "man_seg", False),
    ]

    for folder, prefix, require_zero_start in candidates:
        indexed, digit_widths = _indexed_files(folder, prefix)
        if not indexed:
            continue
        _require_contiguous(indexed, f"{prefix} source", require_zero_start=require_zero_start)
        source_digits = int(next(iter(digit_widths))) if len(digit_widths) == 1 else None
        return sorted(indexed), source_digits

    raise FileNotFoundError(
        f"Could not infer frame count from {source_root / sequence}, "
        f"{source_root / f'{sequence}_GT' / 'TRA'}, or {source_root / f'{sequence}_GT' / 'SEG'}."
    )


def _read_mask(path: Path):
    try:
        mask = np.asarray(tifffile.imread(str(path)))
    except Exception as exc:
        raise ValueError(f"Could not read mask {path}: {exc}") from exc
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"{path} has non-integer dtype {mask.dtype}.")
    if mask.size and int(mask.min()) < 0:
        raise ValueError(f"{path} contains negative labels.")
    return mask


def _spatial_shape(array: np.ndarray):
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr.shape
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        return arr.shape[:2]
    return arr.shape


def _target_shape_from_gt(source_root: Path, sequence: str):
    sequence = _normalize_sequence(sequence)
    candidates = [
        (source_root / f"{sequence}_GT" / "TRA", "man_track"),
        (source_root / f"{sequence}_GT" / "SEG", "man_seg"),
    ]

    for folder, prefix in candidates:
        indexed, _ = _indexed_files(folder, prefix)
        if indexed:
            first = indexed[sorted(indexed)[0]]
            return _spatial_shape(_read_mask(first))

    raise FileNotFoundError(
        f"Could not infer target shape from {source_root / f'{sequence}_GT' / 'TRA'} "
        f"or {source_root / f'{sequence}_GT' / 'SEG'}."
    )


def _resize_label_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]):
    mask_shape = _spatial_shape(mask)
    if mask_shape == target_shape:
        return mask.astype(np.uint16, copy=False)
    if len(mask_shape) != 2 or len(target_shape) != 2:
        raise ValueError(f"Can only resize 2D masks, got {mask_shape} -> {target_shape}.")

    resized = resize(
        mask,
        output_shape=target_shape,
        order=0,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )
    resized = np.rint(resized)
    if resized.size and int(resized.max()) > MAX_UINT16:
        raise ValueError("Resized mask contains labels above uint16 capacity.")
    return resized.astype(np.uint16, copy=False)


def _parse_input_tracks(path: Path):
    rows: dict[int, InputTrackRow] = {}
    if not path.is_file():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_number} must contain four integer columns: L B E P")
            try:
                label, begin, end, parent = [int(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} contains a non-integer value.") from exc
            rows[label] = InputTrackRow(label=label, begin=begin, end=end, parent=parent)
    return rows


def _contiguous_runs(frames: list[int]):
    if not frames:
        return []
    frames = sorted(frames)
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


def _scan_sampled_label_frames(selected_mask_files: list[Path | None], output_frame_indices: list[int]):
    label_frames: dict[int, list[int]] = {}
    reference_shape = None

    for output_frame_idx, mask_path in zip(output_frame_indices, selected_mask_files):
        if mask_path is None:
            continue
        mask = _read_mask(mask_path)
        mask_shape = tuple(mask.shape)
        if reference_shape is None:
            reference_shape = mask_shape
        elif mask_shape != reference_shape:
            raise ValueError(f"{mask_path} shape {mask_shape} differs from first mask shape {reference_shape}.")

        labels = np.unique(mask)
        labels = labels[labels != 0].astype(int, copy=False)
        for label in labels.tolist():
            label_frames.setdefault(label, []).append(output_frame_idx)

    return label_frames, reference_shape


def _build_output_tracks(
    label_frames: dict[int, list[int]],
    input_rows: dict[int, InputTrackRow],
):
    rows_without_parents: list[tuple[int, int, int, int]] = []
    tracks_by_old_label: dict[int, list[tuple[int, int, int, int]]] = {}

    for old_label in sorted(label_frames):
        for begin, end in _contiguous_runs(label_frames[old_label]):
            new_label = len(rows_without_parents) + 1
            if new_label > MAX_UINT16:
                raise ValueError("Downsampled track count exceeds uint16 capacity.")
            row = (old_label, new_label, begin, end)
            rows_without_parents.append(row)
            tracks_by_old_label.setdefault(old_label, []).append(row)

    output_rows: list[OutputTrackRow] = []
    for old_label, new_label, begin, end in rows_without_parents:
        input_row = input_rows.get(old_label)
        parent = 0
        if input_row is not None and input_row.parent != 0 and begin > 0:
            parent_candidates = [
                candidate
                for candidate in tracks_by_old_label.get(input_row.parent, [])
                if candidate[3] < begin
            ]
            if parent_candidates:
                parent = max(parent_candidates, key=lambda candidate: (candidate[3], candidate[1]))[1]
                if parent == new_label:
                    parent = 0

        output_rows.append(
            OutputTrackRow(
                old_label=old_label,
                label=new_label,
                begin=begin,
                end=end,
                parent=parent,
            )
        )

    return output_rows


def _build_frame_label_maps(output_rows: list[OutputTrackRow], output_frame_indices: list[int]):
    frame_maps: dict[int, dict[int, int]] = {frame_idx: {} for frame_idx in output_frame_indices}
    for row in output_rows:
        for frame_idx in range(row.begin, row.end + 1):
            if frame_idx in frame_maps:
                frame_maps[frame_idx][row.old_label] = row.label
    return frame_maps


def _clear_ctc_outputs(output_result_dir: Path):
    output_result_dir.mkdir(parents=True, exist_ok=True)
    for path in output_result_dir.iterdir():
        if path.is_file() and path.name.lower().startswith("mask") and path.suffix.lower() in IMAGE_SUFFIXES:
            path.unlink()
    track_file = output_result_dir / "res_track.txt"
    if track_file.exists():
        track_file.unlink()


def _relabel_mask(mask: np.ndarray, label_map: dict[int, int]):
    relabeled = np.zeros(mask.shape, dtype=np.uint16)
    for old_label, new_label in label_map.items():
        relabeled[mask == old_label] = new_label
    return relabeled


def temporal_downsample_ctc_results(
    input_result_dir: Path,
    output_result_dir: Path,
    source_root: Path | None,
    sequence: str,
    source_frame_count: int | None = None,
    target_shape: tuple[int, int] | None = None,
    pad_missing_with_empty: bool = False,
    factor: int = 16,
    offset: int = 0,
    output_digits: str = "auto",
):
    input_result_dir = input_result_dir.resolve()
    output_result_dir = output_result_dir.resolve()
    source_root = source_root.resolve() if source_root is not None else None
    sequence = _normalize_sequence(sequence)

    if factor < 1:
        raise ValueError("--factor must be >= 1.")
    if offset < 0:
        raise ValueError("--offset must be >= 0.")
    if source_frame_count is not None and source_frame_count < 1:
        raise ValueError("--source-frame-count must be a positive integer.")
    if input_result_dir == output_result_dir:
        raise ValueError("input-result-dir and output-result-dir must be different directories.")
    if not input_result_dir.is_dir():
        raise NotADirectoryError(f"input-result-dir does not exist: {input_result_dir}")
    if source_root is None and source_frame_count is None:
        raise ValueError("Provide either --source-root or --source-frame-count.")
    if source_root is not None and not source_root.is_dir():
        raise NotADirectoryError(f"source-root does not exist: {source_root}")

    if source_frame_count is None:
        output_frame_indices, source_digits = _source_frame_indices_and_digits(source_root, sequence)
    else:
        output_frame_indices = list(range(source_frame_count))
        source_digits = None
    expected_frame_count = len(output_frame_indices)
    if target_shape is None and source_root is not None:
        try:
            target_shape = _target_shape_from_gt(source_root, sequence)
        except FileNotFoundError:
            target_shape = None
    resolved_output_digits = _resolve_output_digits(output_digits, expected_frame_count, source_digits)

    input_masks, _ = _indexed_files(input_result_dir, "mask")
    if not input_masks:
        raise FileNotFoundError(f"No mask*.tif files found in {input_result_dir}.")

    selected_input_indices = [offset + output_idx * factor for output_idx in range(expected_frame_count)]
    missing = [index for index in selected_input_indices if index not in input_masks]
    if missing:
        preview = ", ".join(str(index) for index in missing[:20])
        if not pad_missing_with_empty:
            raise ValueError(
                f"Input result does not contain all selected frames for factor={factor}, offset={offset}. "
                f"Missing input mask indices: {preview}. Pass --pad-missing-with-empty to write blank "
                "prediction masks for unavailable frames."
            )
        print(
            "[TEMPORAL DOWNSAMPLE] warning: "
            f"padding {len(missing)} missing selected frame(s) with empty masks; first missing: {preview}",
            flush=True,
        )
    selected_mask_files = [input_masks.get(index) for index in selected_input_indices]

    first_available_mask = next((path for path in selected_mask_files if path is not None), None)
    blank_shape = target_shape
    if blank_shape is None and first_available_mask is not None:
        blank_shape = tuple(_read_mask(first_available_mask).shape)
    label_frames, reference_shape = _scan_sampled_label_frames(selected_mask_files, output_frame_indices)
    input_rows = _parse_input_tracks(input_result_dir / "res_track.txt")
    output_rows = _build_output_tracks(label_frames, input_rows)
    frame_label_maps = _build_frame_label_maps(output_rows, output_frame_indices)

    _clear_ctc_outputs(output_result_dir)
    for output_frame_idx, input_mask_path in zip(output_frame_indices, selected_mask_files):
        if input_mask_path is None:
            if blank_shape is None:
                raise ValueError("Cannot write empty mask before an available frame establishes the output shape.")
            mask = np.zeros(blank_shape, dtype=np.uint16)
        else:
            mask = _read_mask(input_mask_path)
        relabeled = _relabel_mask(mask, frame_label_maps[output_frame_idx])
        if target_shape is not None:
            relabeled = _resize_label_mask_to_shape(relabeled, target_shape)
        tifffile.imwrite(
            str(output_result_dir / f"mask{output_frame_idx:0{resolved_output_digits}d}.tif"),
            relabeled,
        )

    with (output_result_dir / "res_track.txt").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(f"{row.label} {row.begin} {row.end} {row.parent}\n")

    return {
        "sequence": sequence,
        "frames": expected_frame_count,
        "tracks": len(output_rows),
        "digits": resolved_output_digits,
        "factor": factor,
        "offset": offset,
        "input_result_dir": input_result_dir,
        "output_result_dir": output_result_dir,
        "selected_first": selected_input_indices[0] if selected_input_indices else None,
        "selected_last": selected_input_indices[-1] if selected_input_indices else None,
        "output_first": output_frame_indices[0] if output_frame_indices else None,
        "output_last": output_frame_indices[-1] if output_frame_indices else None,
        "missing_selected_frames": len(missing),
        "shape": reference_shape,
        "target_shape": target_shape,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Temporally downsample interpolated CTC tracking results back to the original "
            "source timeline, rebuilding mask names and res_track.txt."
        )
    )
    parser.add_argument("--input-result-dir", required=True, type=Path, help="Folder containing interpolated mask*.tif files.")
    parser.add_argument("--output-result-dir", required=True, type=Path, help="Destination CTC result folder.")
    parser.add_argument(
        "--source-root",
        default=None,
        type=Path,
        help="Original CTC dataset root. Used to infer frame count unless --source-frame-count is provided.",
    )
    parser.add_argument(
        "--source-frame-count",
        default=None,
        type=int,
        help="Original timeline frame count. Use this to avoid staging/copying raw source images.",
    )
    parser.add_argument(
        "--target-shape",
        default=None,
        help=(
            "Optional output mask shape as HEIGHT,WIDTH. If omitted and --source-root has GT, "
            "the first GT mask shape is used."
        ),
    )
    parser.add_argument(
        "--pad-missing-with-empty",
        action="store_true",
        help="Write blank result masks for selected interpolated frames that are missing from the input result.",
    )
    parser.add_argument("--sequence", required=True, type=str, help="Sequence ID, e.g. 01 or 02.")
    parser.add_argument("--factor", default=16, type=int, help="Temporal downsample factor (default: 16).")
    parser.add_argument("--offset", default=0, type=int, help="First interpolated frame to keep (default: 0).")
    parser.add_argument(
        "--output-digits",
        default="auto",
        help="Digits used for output mask names: auto or a positive integer (default: auto).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        report = temporal_downsample_ctc_results(
            input_result_dir=args.input_result_dir,
            output_result_dir=args.output_result_dir,
            source_root=args.source_root,
            sequence=args.sequence,
            source_frame_count=args.source_frame_count,
            target_shape=(
                None
                if args.target_shape is None
                else tuple(_parse_positive_int(part, "--target-shape") for part in args.target_shape.split(","))
            ),
            pad_missing_with_empty=args.pad_missing_with_empty,
            factor=args.factor,
            offset=args.offset,
            output_digits=args.output_digits,
        )
    except Exception as exc:
        print(f"[TEMPORAL DOWNSAMPLE] FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        "[TEMPORAL DOWNSAMPLE] OK "
        f"sequence={report['sequence']} factor={report['factor']} offset={report['offset']} "
        f"frames={report['frames']} tracks={report['tracks']} digits={report['digits']} "
        f"selected={report['selected_first']}..{report['selected_last']} "
        f"missing_selected_frames={report['missing_selected_frames']} "
        f"output={report['output_result_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
