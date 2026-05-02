#!/usr/bin/env python3

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class TrackRow:
    label: int
    begin: int
    end: int
    parent: int


def _natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _normalize_sequence(sequence: str):
    text = str(sequence)
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def _minimum_digit_width(frame_count: int):
    return max(3, len(str(max(frame_count - 1, 0))))


def _parse_output_digits(output_digits: str, frame_count: int):
    if output_digits == "auto":
        return _minimum_digit_width(frame_count)
    try:
        digits = int(output_digits)
    except ValueError as exc:
        raise ValueError("--output-digits must be 'auto' or a positive integer.") from exc
    if digits < 1:
        raise ValueError("--output-digits must be a positive integer.")
    if frame_count > 10**digits:
        raise ValueError(f"Cannot write {frame_count} frames with {digits} digits.")
    return digits


def _indexed_files(folder: Path, prefix: str, strict_names: bool = True):
    regex = re.compile(rf"^{re.escape(prefix)}(\d+)\.tiff?$", flags=re.IGNORECASE)
    indexed: dict[int, Path] = {}

    if not folder.is_dir():
        return indexed

    bad_names = []
    for path in sorted(folder.iterdir(), key=lambda p: _natural_sort_key(p.name)):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not path.name.lower().startswith(prefix.lower()):
            continue
        match = regex.match(path.name)
        if match is None:
            bad_names.append(path.name)
            continue
        frame_idx = int(match.group(1))
        if frame_idx in indexed:
            raise ValueError(f"Duplicate {prefix} frame {frame_idx}: {indexed[frame_idx]} and {path}")
        indexed[frame_idx] = path

    if bad_names and strict_names:
        preview = ", ".join(bad_names[:10])
        raise ValueError(f"{folder} contains malformed {prefix}*.tif files: {preview}")
    return indexed


def _copy_reindexed_files(
    source_files: dict[int, Path],
    output_dir: Path,
    prefix: str,
    start_frame: int,
    end_frame: int,
    output_digits: int,
    required: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = []
    for source_frame in range(start_frame, end_frame + 1):
        source_path = source_files.get(source_frame)
        if source_path is None:
            missing.append(source_frame)
            continue
        output_frame = source_frame - start_frame
        shutil.copy2(source_path, output_dir / f"{prefix}{output_frame:0{output_digits}d}{source_path.suffix.lower()}")
        copied += 1

    if missing and required:
        preview = ", ".join(str(frame) for frame in missing[:20])
        raise FileNotFoundError(f"Missing required {prefix} frames in source range: {preview}")
    return copied, missing


def _parse_track_file(path: Path):
    rows: dict[int, TrackRow] = {}
    if not path.is_file():
        raise FileNotFoundError(f"Missing GT track file: {path}")

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
            if begin < 0 or end < begin or parent < 0:
                raise ValueError(f"{path}:{line_number} has invalid values: {line}")
            if label in rows:
                raise ValueError(f"{path}:{line_number} duplicates track label {label}")
            rows[label] = TrackRow(label=label, begin=begin, end=end, parent=parent)
    return rows


def _clip_track_rows(rows: dict[int, TrackRow], start_frame: int, end_frame: int):
    clipped: dict[int, TrackRow] = {}
    for label, row in sorted(rows.items()):
        if row.end < start_frame or row.begin > end_frame:
            continue
        begin = max(row.begin, start_frame) - start_frame
        end = min(row.end, end_frame) - start_frame
        clipped[label] = TrackRow(label=label, begin=begin, end=end, parent=row.parent)

    fixed: list[TrackRow] = []
    for label, row in sorted(clipped.items()):
        parent = 0
        if row.parent != 0 and row.begin > 0:
            parent_row = clipped.get(row.parent)
            if parent_row is not None and parent_row.end < row.begin:
                parent = row.parent
        fixed.append(TrackRow(label=label, begin=row.begin, end=row.end, parent=parent))
    return fixed


def _write_track_file(path: Path, rows: list[TrackRow]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row.label} {row.begin} {row.end} {row.parent}\n")


def subset_ctc_sequence_range(
    source_root: Path,
    output_root: Path,
    sequence: str,
    start_frame: int,
    end_frame: int,
    output_digits: str = "auto",
    overwrite: bool = False,
):
    if start_frame > end_frame:
        raise ValueError("--start-frame must be <= --end-frame.")
    if not source_root.is_dir():
        raise NotADirectoryError(f"source-root does not exist: {source_root}")

    sequence = _normalize_sequence(sequence)
    frame_count = end_frame - start_frame + 1
    digits = _parse_output_digits(output_digits, frame_count)

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists and is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    image_src = _indexed_files(source_root / sequence, "t", strict_names=False)
    tra_src_dir = source_root / f"{sequence}_GT" / "TRA"
    seg_src_dir = source_root / f"{sequence}_GT" / "SEG"
    tra_src = _indexed_files(tra_src_dir, "man_track")
    seg_src = _indexed_files(seg_src_dir, "man_seg")

    copied_images, missing_images = _copy_reindexed_files(
        image_src,
        output_root / sequence,
        "t",
        start_frame,
        end_frame,
        digits,
        required=False,
    )
    copied_tra, _ = _copy_reindexed_files(
        tra_src,
        output_root / f"{sequence}_GT" / "TRA",
        "man_track",
        start_frame,
        end_frame,
        digits,
        required=True,
    )
    copied_seg, missing_seg = _copy_reindexed_files(
        seg_src,
        output_root / f"{sequence}_GT" / "SEG",
        "man_seg",
        start_frame,
        end_frame,
        digits,
        required=False,
    )

    rows = _parse_track_file(tra_src_dir / "man_track.txt")
    clipped_rows = _clip_track_rows(rows, start_frame, end_frame)
    _write_track_file(output_root / f"{sequence}_GT" / "TRA" / "man_track.txt", clipped_rows)

    return {
        "sequence": sequence,
        "frames": frame_count,
        "digits": digits,
        "source_range": f"{start_frame}..{end_frame}",
        "output_root": output_root,
        "images": copied_images,
        "missing_images": len(missing_images),
        "tra_masks": copied_tra,
        "seg_masks": copied_seg,
        "missing_seg_masks": len(missing_seg),
        "track_rows": len(clipped_rows),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a rebased CTC source/GT subset for a contiguous frame range."
    )
    parser.add_argument("--source-root", required=True, type=Path, help="Full CTC dataset root containing <sequence> and <sequence>_GT.")
    parser.add_argument("--output-root", required=True, type=Path, help="Destination root for the rebased subset.")
    parser.add_argument("--sequence", default="01", type=str, help="CTC sequence, e.g. 01 or 02.")
    parser.add_argument("--start-frame", required=True, type=int, help="First full-dataset frame index, inclusive.")
    parser.add_argument("--end-frame", required=True, type=int, help="Last full-dataset frame index, inclusive.")
    parser.add_argument("--output-digits", default="auto", help="Output frame digits: auto or a positive integer.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output-root if it already has files.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        report = subset_ctc_sequence_range(
            source_root=args.source_root.resolve(),
            output_root=args.output_root.resolve(),
            sequence=args.sequence,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            output_digits=args.output_digits,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"[CTC SUBSET] FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        "[CTC SUBSET] OK "
        f"sequence={report['sequence']} source_range={report['source_range']} "
        f"frames={report['frames']} digits={report['digits']} "
        f"images={report['images']} tra_masks={report['tra_masks']} "
        f"seg_masks={report['seg_masks']} track_rows={report['track_rows']} "
        f"output={report['output_root']}"
    )
    if report["missing_images"]:
        print(f"[CTC SUBSET] warning: missing source images={report['missing_images']}")
    if report["missing_seg_masks"]:
        print(f"[CTC SUBSET] note: missing SEG masks={report['missing_seg_masks']} (SEG GT is often sparse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
