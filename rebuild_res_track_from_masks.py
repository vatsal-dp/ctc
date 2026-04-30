#!/usr/bin/env python3

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass
class TrackSpan:
    begin: int
    end: int
    last_seen: int
    has_gap: bool = False


def _natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _read_existing_parents(path: Path):
    parents: dict[int, int] = {}
    if not path.is_file():
        return parents

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_number} must have four columns: L B E P")
            label, _begin, _end, parent = [int(part) for part in parts]
            parents[label] = parent
    return parents


def _scan_mask_spans(mask_files: list[Path]):
    spans: dict[int, TrackSpan] = {}
    reference_shape = None

    for frame_idx, mask_path in enumerate(mask_files):
        mask = np.asarray(tifffile.imread(str(mask_path)))
        if mask.ndim != 2:
            raise ValueError(f"{mask_path} is not a 2D label mask; shape={mask.shape}")
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"{mask_path} has non-integer dtype {mask.dtype}")

        if reference_shape is None:
            reference_shape = mask.shape
        elif mask.shape != reference_shape:
            raise ValueError(f"{mask_path} shape {mask.shape} differs from first mask shape {reference_shape}")

        labels = np.unique(mask)
        labels = labels[labels != 0].astype(int, copy=False)
        for label in labels.tolist():
            span = spans.get(label)
            if span is None:
                spans[label] = TrackSpan(begin=frame_idx, end=frame_idx, last_seen=frame_idx)
                continue
            if frame_idx != span.last_seen + 1:
                span.has_gap = True
            span.end = frame_idx
            span.last_seen = frame_idx

        if frame_idx % 250 == 0 or frame_idx == len(mask_files) - 1:
            print(f"[REBUILD] scanned frame {frame_idx + 1}/{len(mask_files)}", flush=True)

    return spans, reference_shape


def rebuild_res_track(result_dir: Path, backup: bool = True, dry_run: bool = False):
    result_dir = result_dir.resolve()
    mask_files = sorted(result_dir.glob("mask*.tif"), key=lambda path: _natural_sort_key(path.name))
    if not mask_files:
        raise FileNotFoundError(f"No mask*.tif files found in {result_dir}")

    track_path = result_dir / "res_track.txt"
    existing_parents = _read_existing_parents(track_path)
    spans, reference_shape = _scan_mask_spans(mask_files)

    gaps = sorted(label for label, span in spans.items() if span.has_gap)
    if gaps:
        preview = ", ".join(str(label) for label in gaps[:20])
        raise ValueError(
            f"Cannot rebuild a valid CTC res_track.txt because labels have frame gaps: {preview}. "
            "Re-export with the CTC exporter so gap labels can be split/remapped."
        )

    rows = []
    dropped_parents = 0
    for label in sorted(spans):
        span = spans[label]
        parent = int(existing_parents.get(label, 0))
        if parent != 0:
            parent_span = spans.get(parent)
            if parent_span is None or parent == label or parent_span.end >= span.begin:
                parent = 0
                dropped_parents += 1
        rows.append((label, span.begin, span.end, parent))

    print(
        f"[REBUILD] masks={len(mask_files)} shape={reference_shape} tracks={len(rows)} "
        f"preserved_parents={sum(1 for row in rows if row[3] != 0)} dropped_invalid_parents={dropped_parents}",
        flush=True,
    )

    if dry_run:
        return rows

    if backup and track_path.is_file():
        backup_path = track_path.with_suffix(track_path.suffix + ".bak")
        if backup_path.exists():
            counter = 1
            while True:
                candidate = track_path.with_suffix(track_path.suffix + f".bak{counter}")
                if not candidate.exists():
                    backup_path = candidate
                    break
                counter += 1
        shutil.copy2(track_path, backup_path)
        print(f"[REBUILD] backed up {track_path} -> {backup_path}", flush=True)

    with track_path.open("w", encoding="utf-8") as handle:
        for label, begin, end, parent in rows:
            handle.write(f"{label} {begin} {end} {parent}\n")
    print(f"[REBUILD] wrote {track_path}", flush=True)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild CTC res_track.txt by scanning the actual mask*.tif files in a result folder."
    )
    parser.add_argument("--result-dir", required=True, type=Path, help="CTC result folder, e.g. .../02_RES.")
    parser.add_argument("--no-backup", action="store_true", help="Do not save res_track.txt.bak first.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing res_track.txt.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        rebuild_res_track(
            result_dir=args.result_dir,
            backup=not args.no_backup,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"[REBUILD] FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
