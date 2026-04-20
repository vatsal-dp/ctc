#!/usr/bin/env python3

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


class ValidationError(Exception):
    pass


@dataclass
class TrackRow:
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


def _time_digits_from_name(filename: str, prefixes: tuple[str, ...]):
    for prefix in prefixes:
        match = re.match(rf"^{re.escape(prefix)}(\d+)\.tiff?$", filename, flags=re.IGNORECASE)
        if match is None:
            continue
        width = len(match.group(1))
        if width in {3, 4}:
            return width
    return None


def infer_digits_from_files(files: list[Path], prefixes: tuple[str, ...]):
    widths = {_time_digits_from_name(path.name, prefixes) for path in files}
    widths.discard(None)
    if len(widths) == 1:
        return next(iter(widths))
    return None


def resolve_digits(digits_arg: str, dataset_root: Path, source_root: Path | None, sequence: str) -> int:
    if digits_arg != "auto":
        return int(digits_arg)

    search_roots = []
    if source_root is not None:
        search_roots.append(source_root)
    search_roots.append(dataset_root)

    candidates: list[tuple[Path, tuple[str, ...]]] = []
    for root in search_roots:
        candidates.extend(
            [
                (root / sequence, ("t",)),
                (root / f"{sequence}_ERR_SEG", ("mask",)),
                (root / f"{sequence}_RES", ("mask",)),
                (root / f"{sequence}_GT" / "TRA", ("man_track",)),
                (root / f"{sequence}_GT" / "SEG", ("man_seg",)),
            ]
        )

    for folder, prefixes in candidates:
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.tif"), key=lambda p: _natural_sort_key(p.name))
        inferred = infer_digits_from_files(files, prefixes)
        if inferred is not None:
            return int(inferred)

    raise ValidationError(
        "Could not infer CTC digit width. Pass --digits 3 or --digits 4 explicitly."
    )


def _read_tiff(path: Path):
    try:
        return tifffile.imread(str(path))
    except Exception as exc:
        raise ValidationError(f"Could not read TIFF file {path}: {exc}") from exc


def _spatial_shape(array: np.ndarray):
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr.shape
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        return arr.shape[:2]
    return arr.shape


def _parse_indexed_files(folder: Path, prefix: str, digits: int):
    regex = re.compile(rf"^{re.escape(prefix)}(\d{{{digits}}})\.tiff?$", flags=re.IGNORECASE)
    indexed: dict[int, Path] = {}
    bad_names = []

    for path in sorted(folder.glob(f"{prefix}*.tif"), key=lambda p: _natural_sort_key(p.name)):
        match = regex.match(path.name)
        if match is None:
            bad_names.append(path.name)
            continue
        index = int(match.group(1))
        if index in indexed:
            raise ValidationError(f"Duplicate {prefix} file for frame {index}: {indexed[index]} and {path}")
        indexed[index] = path

    if bad_names:
        preview = ", ".join(bad_names[:10])
        raise ValidationError(
            f"{folder} contains {prefix}*.tif files that do not match fixed {digits}-digit CTC naming: {preview}"
        )
    return indexed


def _require_contiguous_indices(indexed: dict[int, Path], prefix: str):
    if not indexed:
        raise ValidationError(f"No {prefix}*.tif files found.")
    observed = sorted(indexed)
    expected = list(range(observed[-1] + 1))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        preview = ", ".join(str(idx) for idx in missing[:20])
        raise ValidationError(f"{prefix} files are not contiguous from frame 0. Missing indices: {preview}")


def _parse_res_track(path: Path):
    if not path.is_file():
        raise ValidationError(f"Missing required res_track.txt: {path}")

    rows: dict[int, TrackRow] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValidationError(
                    f"{path}:{line_number} must contain exactly four integer columns: L B E P"
                )
            try:
                label, begin, end, parent = [int(part) for part in parts]
            except ValueError as exc:
                raise ValidationError(f"{path}:{line_number} contains a non-integer value.") from exc

            if label <= 0:
                raise ValidationError(f"{path}:{line_number} has non-positive track label {label}.")
            if label > np.iinfo(np.uint16).max:
                raise ValidationError(f"{path}:{line_number} has label {label}, exceeding uint16 capacity.")
            if begin < 0 or end < 0 or begin > end:
                raise ValidationError(f"{path}:{line_number} has invalid B/E range: {begin}/{end}.")
            if parent < 0:
                raise ValidationError(f"{path}:{line_number} has negative parent ID {parent}.")
            if parent == label:
                raise ValidationError(f"{path}:{line_number} uses its own label as parent ID.")
            if label in rows:
                raise ValidationError(f"{path}:{line_number} duplicates track label {label}.")

            rows[label] = TrackRow(label=label, begin=begin, end=end, parent=parent)

    return rows


def _source_frame_index(source_root: Path | None, dataset_root: Path, sequence: str, digits: int):
    roots = []
    if source_root is not None:
        roots.append(source_root)
    roots.append(dataset_root)

    for root in roots:
        source_dir = root / sequence
        if source_dir.is_dir():
            indexed = _parse_indexed_files(source_dir, "t", digits)
            if indexed:
                return indexed

    return {}


def validate_ctc_result_format(
    dataset_root: Path,
    sequence: str,
    digits_arg: str,
    source_root: Path | None = None,
    allow_non_uint16: bool = False,
):
    sequence = _normalize_sequence(sequence)
    dataset_root = dataset_root.resolve()
    source_root = source_root.resolve() if source_root is not None else None
    result_dir = dataset_root / f"{sequence}_RES"

    if not dataset_root.is_dir():
        raise ValidationError(f"dataset-root does not exist or is not a directory: {dataset_root}")
    if not result_dir.is_dir():
        raise ValidationError(f"Expected result folder does not exist: {result_dir}")

    digits = resolve_digits(digits_arg, dataset_root, source_root, sequence)
    if digits not in {3, 4}:
        raise ValidationError("--digits must resolve to 3 or 4.")

    mask_files = _parse_indexed_files(result_dir, "mask", digits)
    _require_contiguous_indices(mask_files, "mask")

    source_frames = _source_frame_index(source_root, dataset_root, sequence, digits)
    if source_frames and sorted(mask_files) != sorted(source_frames):
        missing = sorted(set(source_frames) - set(mask_files))
        extra = sorted(set(mask_files) - set(source_frames))
        details = []
        if missing:
            details.append(f"missing result frames: {missing[:20]}")
        if extra:
            details.append(f"extra result frames: {extra[:20]}")
        raise ValidationError("Result frames do not match source frames; " + "; ".join(details))

    rows = _parse_res_track(result_dir / "res_track.txt")
    label_frames: dict[int, list[int]] = {}
    reference_shape = None

    for frame_index, mask_path in sorted(mask_files.items()):
        mask = np.asarray(_read_tiff(mask_path))
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValidationError(f"{mask_path} has non-integer dtype {mask.dtype}.")
        if mask.dtype != np.uint16 and not allow_non_uint16:
            raise ValidationError(
                f"{mask_path} has dtype {mask.dtype}; CTC result masks should be uint16. "
                "Pass --allow-non-uint16 only for exploratory validation."
            )
        if mask.size and int(mask.min()) < 0:
            raise ValidationError(f"{mask_path} contains negative labels.")
        if mask.size and int(mask.max()) > np.iinfo(np.uint16).max:
            raise ValidationError(f"{mask_path} contains labels above uint16 capacity.")

        mask_shape = _spatial_shape(mask)
        if reference_shape is None:
            reference_shape = mask_shape
        elif mask_shape != reference_shape:
            raise ValidationError(f"{mask_path} shape {mask_shape} differs from first mask shape {reference_shape}.")

        source_path = source_frames.get(frame_index)
        if source_path is not None:
            source_shape = _spatial_shape(_read_tiff(source_path))
            if source_shape != mask_shape:
                raise ValidationError(
                    f"Shape mismatch at frame {frame_index}: result={mask_shape}, source={source_shape}."
                )

        labels = np.unique(mask)
        labels = labels[labels != 0].astype(int)
        for label in labels.tolist():
            label_frames.setdefault(label, []).append(frame_index)

    observed_labels = set(label_frames)
    row_labels = set(rows)
    if observed_labels != row_labels:
        missing_rows = sorted(observed_labels - row_labels)
        missing_pixels = sorted(row_labels - observed_labels)
        details = []
        if missing_rows:
            details.append(f"labels in masks but missing from res_track.txt: {missing_rows[:20]}")
        if missing_pixels:
            details.append(f"labels in res_track.txt but missing from masks: {missing_pixels[:20]}")
        raise ValidationError("; ".join(details))

    max_frame = max(mask_files)
    for label, row in sorted(rows.items()):
        if row.end > max_frame:
            raise ValidationError(f"Track {label} has E={row.end}, but max result frame is {max_frame}.")

        frames = label_frames.get(label, [])
        if not frames:
            raise ValidationError(f"Track {label} has no pixels in result masks.")
        if row.begin != frames[0] or row.end != frames[-1]:
            raise ValidationError(
                f"Track {label} B/E mismatch: res_track={row.begin}/{row.end}, "
                f"observed={frames[0]}/{frames[-1]}."
            )

        expected_frames = list(range(row.begin, row.end + 1))
        if frames != expected_frames:
            missing = sorted(set(expected_frames) - set(frames))
            raise ValidationError(f"Track {label} has internal frame gaps. Missing frames: {missing[:20]}")

    for label, row in sorted(rows.items()):
        if row.parent == 0:
            continue
        parent = rows.get(row.parent)
        if parent is None:
            raise ValidationError(f"Track {label} has parent {row.parent}, which is not a valid track ID.")
        if parent.end >= row.begin:
            raise ValidationError(
                f"Track {label} begins at {row.begin}, but parent {row.parent} ends at {parent.end}; "
                "a parent track must end before its child track starts."
            )

    return {
        "sequence": sequence,
        "digits": digits,
        "frames": len(mask_files),
        "tracks": len(rows),
        "result_dir": result_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate CTC tracking result folder format.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Root containing <sequence>_RES.")
    parser.add_argument(
        "--source-root",
        default=None,
        type=Path,
        help="Optional original CTC dataset root containing <sequence>/ source frames.",
    )
    parser.add_argument("--sequence", required=True, type=str, help="CTC sequence ID, e.g. 01 or 02.")
    parser.add_argument("--digits", default="auto", choices=["auto", "3", "4"], help="CTC time index digits.")
    parser.add_argument(
        "--allow-non-uint16",
        action="store_true",
        help="Allow non-uint16 integer masks for exploratory checks only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        report = validate_ctc_result_format(
            dataset_root=args.dataset_root,
            sequence=args.sequence,
            digits_arg=args.digits,
            source_root=args.source_root,
            allow_non_uint16=args.allow_non_uint16,
        )
    except ValidationError as exc:
        print(f"[CTC FORMAT] FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        "[CTC FORMAT] OK "
        f"sequence={report['sequence']} digits={report['digits']} "
        f"frames={report['frames']} tracks={report['tracks']} result_dir={report['result_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
