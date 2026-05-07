#!/usr/bin/env python3
"""Run tip tracking in overlapping blocks and stitch the tracked blocks.

This wrapper is intended for very long CTC-style mask sequences where running
the MATLAB-inspired tracker over every frame at once is too slow or too risky.
Each block is tracked independently by ram_run_tiptracking_standalone_optimized.py.
The merge step keeps only the owned frames from each block and uses the
previous overlap to map block-local labels onto already-written global labels.
"""

import argparse
import concurrent.futures
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from export_ctc_res_from_tracked_masks import (
    _build_frame_maps,
    _build_track_runs,
    _parse_output_digits,
    _prepare_output_dir,
    _scan_label_frames,
)
from temporal_downsample_ctc_results import _relabel_mask, temporal_downsample_ctc_results


IMAGE_SUFFIXES = {".tif", ".tiff"}
MAX_UINT16 = int(np.iinfo(np.uint16).max)


@dataclass(frozen=True)
class BlockRange:
    """Global frame ranges for one independently tracked block."""

    index: int
    run_start: int
    run_end: int
    owned_start: int
    owned_end: int

    @property
    def run_frame_count(self) -> int:
        return self.run_end - self.run_start + 1

    @property
    def owned_frame_count(self) -> int:
        return self.owned_end - self.owned_start + 1

    def local_index(self, global_frame: int) -> int:
        return global_frame - self.run_start


@dataclass(frozen=True)
class BlockResult:
    """A tracked result folder associated with its source block range."""

    block: BlockRange
    result_dir: Path


@dataclass(frozen=True)
class TrackRow:
    """Parsed L B E P row from a block-local res_track.txt."""

    label: int
    begin: int
    end: int
    parent: int


def _natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _minimum_digit_width(frame_count: int) -> int:
    return max(3, len(str(max(frame_count - 1, 0))))


def build_block_ranges(frame_count: int, block_size: int = 1000, overlap: int = 100) -> list[BlockRange]:
    """Build contiguous owned blocks with extra overlap in the tracked run."""
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative.")
    if block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= block_size:
        raise ValueError("overlap must be smaller than block_size.")
    if frame_count == 0:
        return []

    blocks: list[BlockRange] = []
    owned_start = 0
    while owned_start < frame_count:
        owned_end = min(frame_count - 1, owned_start + block_size - 1)
        run_start = max(0, owned_start - overlap)
        run_end = min(frame_count - 1, owned_end + overlap)
        blocks.append(
            BlockRange(
                index=len(blocks),
                run_start=run_start,
                run_end=run_end,
                owned_start=owned_start,
                owned_end=owned_end,
            )
        )
        owned_start = owned_end + 1
    return blocks


def _resolve_mask_files(mask_dir: Path, mask_pattern: str | None):
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"mask-dir does not exist: {mask_dir}")

    if mask_pattern:
        if any(char in mask_pattern for char in ["*", "?", "["]):
            files = sorted(mask_dir.glob(mask_pattern), key=lambda path: _natural_sort_key(path.name))
        else:
            files = sorted(
                [path for path in mask_dir.iterdir() if path.is_file() and path.name.endswith(mask_pattern)],
                key=lambda path: _natural_sort_key(path.name),
            )
    else:
        files = []
        for suffix in ["_cp_masks.tif", "_omni5_masks.tif", "_ART_masks.tif", "_masks.tif", "mask*.tif"]:
            if "*" in suffix:
                candidates = sorted(mask_dir.glob(suffix), key=lambda path: _natural_sort_key(path.name))
            else:
                candidates = sorted(
                    [path for path in mask_dir.iterdir() if path.is_file() and path.name.endswith(suffix)],
                    key=lambda path: _natural_sort_key(path.name),
                )
            if candidates:
                files = candidates
                break

    files = [path for path in files if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        pattern_text = mask_pattern if mask_pattern is not None else "default Cellpose mask suffixes"
        raise FileNotFoundError(f"No TIFF masks found in {mask_dir} using {pattern_text}.")
    return files


def _indexed_mask_files(result_dir: Path):
    regex = re.compile(r"^mask(\d+)\.tiff?$", flags=re.IGNORECASE)
    indexed: dict[int, Path] = {}
    bad_names = []
    for path in sorted(result_dir.glob("mask*.tif"), key=lambda item: _natural_sort_key(item.name)):
        match = regex.match(path.name)
        if match is None:
            bad_names.append(path.name)
            continue
        frame_idx = int(match.group(1))
        if frame_idx in indexed:
            raise ValueError(f"Duplicate block mask frame {frame_idx}: {indexed[frame_idx]} and {path}")
        indexed[frame_idx] = path
    if bad_names:
        preview = ", ".join(bad_names[:10])
        raise ValueError(f"{result_dir} contains malformed mask*.tif files: {preview}")
    return indexed


def _read_block_masks(block_result: BlockResult) -> list[np.ndarray]:
    indexed = _indexed_mask_files(block_result.result_dir)
    expected = list(range(block_result.block.run_frame_count))
    observed = sorted(indexed)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        preview = ", ".join(str(frame_idx) for frame_idx in missing[:20])
        raise ValueError(
            f"{block_result.result_dir} does not contain contiguous block masks "
            f"0..{block_result.block.run_frame_count - 1}. Missing: {preview}"
        )
    return [np.asarray(tifffile.imread(str(indexed[frame_idx]))) for frame_idx in expected]


def _parse_track_rows(path: Path) -> dict[int, TrackRow]:
    rows: dict[int, TrackRow] = {}
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
            label, begin, end, parent = [int(part) for part in parts]
            rows[label] = TrackRow(label=label, begin=begin, end=end, parent=parent)
    return rows


def _labels_in_masks(masks: list[np.ndarray]) -> set[int]:
    labels: set[int] = set()
    for mask in masks:
        frame_labels = np.unique(mask)
        labels.update(int(label) for label in frame_labels.tolist() if int(label) != 0)
    return labels


def _accumulate_overlap_match_stats(
    block: BlockRange,
    block_masks: list[np.ndarray],
    global_frames: dict[int, np.ndarray],
):
    local_area: dict[int, int] = {}
    global_area: dict[int, int] = {}
    intersections: dict[tuple[int, int], int] = {}
    intersection_frames: dict[tuple[int, int], set[int]] = {}

    for global_frame in range(block.run_start, block.owned_start):
        global_mask = global_frames.get(global_frame)
        if global_mask is None:
            continue
        local_idx = block.local_index(global_frame)
        if local_idx < 0 or local_idx >= len(block_masks):
            continue
        local_mask = np.asarray(block_masks[local_idx])
        global_mask = np.asarray(global_mask)
        if local_mask.shape != global_mask.shape:
            raise ValueError(
                f"Shape mismatch in block {block.index} overlap frame {global_frame}: "
                f"local={local_mask.shape}, global={global_mask.shape}"
            )

        labels, counts = np.unique(local_mask[local_mask != 0], return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            local_area[int(label)] = local_area.get(int(label), 0) + int(count)

        labels, counts = np.unique(global_mask[global_mask != 0], return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            global_area[int(label)] = global_area.get(int(label), 0) + int(count)

        both = (local_mask != 0) & (global_mask != 0)
        if not np.any(both):
            continue
        global_max = int(global_mask[both].max())
        stride = np.uint64(global_max + 1)
        encoded = local_mask[both].astype(np.uint64, copy=False) * stride + global_mask[both].astype(
            np.uint64,
            copy=False,
        )
        pair_values, pair_counts = np.unique(encoded, return_counts=True)
        for encoded_pair, count in zip(pair_values.tolist(), pair_counts.tolist()):
            local_label = int(encoded_pair // int(stride))
            global_label = int(encoded_pair % int(stride))
            pair = (local_label, global_label)
            intersections[pair] = intersections.get(pair, 0) + int(count)
            intersection_frames.setdefault(pair, set()).add(global_frame)

    return local_area, global_area, intersections, intersection_frames


def _match_overlap_labels(
    block: BlockRange,
    block_masks: list[np.ndarray],
    global_frames: dict[int, np.ndarray],
    min_iou: float,
    min_overlap_frames: int,
) -> dict[int, int]:
    if block.owned_start == 0:
        return {}

    local_area, global_area, intersections, intersection_frames = _accumulate_overlap_match_stats(
        block=block,
        block_masks=block_masks,
        global_frames=global_frames,
    )
    eligible: list[tuple[int, int, float, int, int]] = []
    for pair, intersection in intersections.items():
        local_label, global_label = pair
        union = local_area.get(local_label, 0) + global_area.get(global_label, 0) - intersection
        if union <= 0:
            continue
        iou = intersection / union
        frames = len(intersection_frames.get(pair, set()))
        if iou >= min_iou and frames >= min_overlap_frames:
            eligible.append((local_label, global_label, iou, frames, intersection))

    local_candidates: dict[int, list[tuple[int, int, float, int, int]]] = {}
    global_candidates: dict[int, list[tuple[int, int, float, int, int]]] = {}
    for candidate in eligible:
        local_candidates.setdefault(candidate[0], []).append(candidate)
        global_candidates.setdefault(candidate[1], []).append(candidate)

    mapping: dict[int, int] = {}
    for local_label, candidates in local_candidates.items():
        if len(candidates) != 1:
            continue
        candidate = candidates[0]
        global_label = candidate[1]
        if len(global_candidates.get(global_label, [])) != 1:
            continue
        mapping[local_label] = global_label
    return mapping


def _remap_mask(mask: np.ndarray, label_map: dict[int, int]) -> np.ndarray:
    return _relabel_mask(np.asarray(mask), label_map).astype(np.uint32, copy=False)


def _write_final_result(
    frames_by_index: dict[int, np.ndarray],
    output_result_dir: Path,
    output_digits: str,
    parent_map: dict[int, int],
):
    if not frames_by_index:
        raise ValueError("No merged frames to write.")
    frame_indices = sorted(frames_by_index)
    expected = list(range(frame_indices[0], frame_indices[-1] + 1))
    if frame_indices[0] != 0 or frame_indices != expected:
        missing = sorted(set(expected) - set(frame_indices))
        preview = ", ".join(str(frame_idx) for frame_idx in missing[:20])
        raise ValueError(f"Merged frames must be contiguous from 0. Missing: {preview}")

    frames = [frames_by_index[frame_idx] for frame_idx in frame_indices]
    frame_count = len(frames)
    digits = _parse_output_digits(output_digits, frame_count)
    label_frames = _scan_label_frames(frames)
    rows, split_labels = _build_track_runs(label_frames, parent_map=parent_map)
    frame_maps = _build_frame_maps(rows, frame_count)

    _prepare_output_dir(output_result_dir, overwrite=True)
    with (output_result_dir / "res_track.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row.label} {row.begin} {row.end} {row.parent}\n")

    for frame_idx, frame in enumerate(frames):
        relabeled = _relabel_mask(frame, frame_maps[frame_idx])
        tifffile.imwrite(str(output_result_dir / f"mask{frame_idx:0{digits}d}.tif"), relabeled)

    return {"frames": frame_count, "tracks": len(rows), "digits": digits, "split_labels": split_labels}


def merge_block_results(
    block_results: list[BlockResult],
    output_result_dir: Path,
    output_digits: str = "auto",
    min_iou: float = 0.50,
    min_overlap_frames: int = 3,
):
    """Merge independently tracked block results into one CTC result folder."""
    if min_iou <= 0 or min_iou > 1:
        raise ValueError("min_iou must be in the range (0, 1].")
    if min_overlap_frames < 1:
        raise ValueError("min_overlap_frames must be a positive integer.")

    block_results = sorted(block_results, key=lambda item: item.block.index)
    if not block_results:
        raise ValueError("No block results to merge.")

    expected_owned_start = 0
    for block_result in block_results:
        block = block_result.block
        if block.owned_start != expected_owned_start:
            raise ValueError(
                f"Block {block.index} owned_start={block.owned_start}, expected {expected_owned_start}."
            )
        expected_owned_start = block.owned_end + 1

    global_frames: dict[int, np.ndarray] = {}
    parent_map: dict[int, int] = {}
    next_global_id = 1
    matched_labels = 0

    for block_result in block_results:
        block = block_result.block
        block_masks = _read_block_masks(block_result)
        local_to_global = _match_overlap_labels(
            block=block,
            block_masks=block_masks,
            global_frames=global_frames,
            min_iou=min_iou,
            min_overlap_frames=min_overlap_frames,
        )
        matched_labels += len(local_to_global)

        owned_masks = [
            block_masks[block.local_index(global_frame)]
            for global_frame in range(block.owned_start, block.owned_end + 1)
        ]
        owned_labels = _labels_in_masks(owned_masks)
        newly_allocated: set[int] = set()
        for local_label in sorted(owned_labels):
            if local_label in local_to_global:
                continue
            if next_global_id > MAX_UINT16:
                raise ValueError("Merged track count exceeds uint16 capacity.")
            local_to_global[local_label] = next_global_id
            newly_allocated.add(local_label)
            next_global_id += 1

        rows = _parse_track_rows(block_result.result_dir / "res_track.txt")
        for row in rows.values():
            if row.parent == 0 or row.label not in newly_allocated:
                continue
            child_global = local_to_global.get(row.label)
            parent_global = local_to_global.get(row.parent)
            if child_global is None or parent_global is None or child_global == parent_global:
                continue
            parent_map.setdefault(child_global, parent_global)

        for global_frame in range(block.owned_start, block.owned_end + 1):
            local_idx = block.local_index(global_frame)
            global_frames[global_frame] = _remap_mask(block_masks[local_idx], local_to_global)

        print(
            f"[BLOCKWISE] merged block {block.index + 1}/{len(block_results)} "
            f"owned={block.owned_start}..{block.owned_end} matched_overlap_labels={len(local_to_global) - len(newly_allocated)} "
            f"new_labels={len(newly_allocated)}",
            flush=True,
        )

    report = _write_final_result(
        frames_by_index=global_frames,
        output_result_dir=output_result_dir,
        output_digits=output_digits,
        parent_map=parent_map,
    )
    report.update({"blocks": len(block_results), "matched_overlap_labels": matched_labels})
    return report


def _copy_block_inputs(files: list[Path], block: BlockRange, block_input_dir: Path):
    if block_input_dir.exists():
        shutil.rmtree(block_input_dir)
    block_input_dir.mkdir(parents=True)
    digits = _minimum_digit_width(block.run_frame_count)
    for local_idx, global_idx in enumerate(range(block.run_start, block.run_end + 1)):
        shutil.copy2(files[global_idx], block_input_dir / f"mask{local_idx:0{digits}d}.tif")


def _format_command(command: list[str]) -> str:
    return " ".join(_shell_quote(str(part)) for part in command)


def _shell_quote(text: str) -> str:
    if not text:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:"
    if all(char in safe for char in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _run_tracking_block(
    block: BlockRange,
    files: list[Path],
    block_work_root: Path,
    args,
) -> BlockResult:
    block_name = f"block_{block.index:04d}"
    block_input_dir = block_work_root / "inputs" / block_name
    block_output_root = block_work_root / "tracking"
    block_log_dir = block_work_root / "logs"
    block_log_dir.mkdir(parents=True, exist_ok=True)
    _copy_block_inputs(files, block, block_input_dir)

    command = [
        str(args.python),
        str(args.tracking_script),
        "--mask-dir",
        str(block_input_dir),
        "--mask-pattern",
        "mask*.tif",
        "--output-dir",
        str(block_output_root),
        "--position",
        block_name,
        "--time-series-threshold",
        str(args.time_series_threshold),
        "--output-digits",
        args.output_digits,
        "--io-workers",
        str(args.io_workers),
        "--io-queue-depth",
        str(args.io_queue_depth),
        "--tiff-write-workers",
        str(args.tiff_write_workers),
        "--stack-storage",
        args.stack_storage,
        "--export-mode",
        "full",
    ]
    if args.mmap_dir is not None:
        command.extend(["--mmap-dir", str(args.mmap_dir)])

    log_path = block_log_dir / f"{block_name}.log"
    printable = _format_command(command)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"command={printable}\n")
        log.write(f"global_run={block.run_start}..{block.run_end}\n")
        log.write(f"global_owned={block.owned_start}..{block.owned_end}\n\n")
        print(f"[BLOCKWISE] tracking {block_name} run={block.run_start}..{block.run_end}", flush=True)
        process = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write(process.stdout or "")
        log.write(f"\nreturncode={process.returncode}\n")
    if process.returncode != 0:
        raise RuntimeError(f"Tracking block {block_name} failed with return code {process.returncode}. See {log_path}")

    return BlockResult(block=block, result_dir=block_output_root / f"{block_name}_RES")


def _run_blocks(files: list[Path], blocks: list[BlockRange], block_work_root: Path, args):
    if block_work_root.exists():
        shutil.rmtree(block_work_root)
    block_work_root.mkdir(parents=True)

    if args.jobs == 1:
        return [_run_tracking_block(block, files, block_work_root, args) for block in blocks]

    results: dict[int, BlockResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_block = {
            executor.submit(_run_tracking_block, block, files, block_work_root, args): block
            for block in blocks
        }
        for future in concurrent.futures.as_completed(future_to_block):
            block = future_to_block[future]
            results[block.index] = future.result()
    return [results[index] for index in sorted(results)]


def run_blockwise_tracking(args):
    mask_dir = args.mask_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _resolve_mask_files(mask_dir, args.mask_pattern)
    blocks = build_block_ranges(len(files), block_size=args.block_size, overlap=args.overlap)
    if not blocks:
        raise ValueError("No input frames to track.")

    print(
        f"[BLOCKWISE] START masks={len(files)} blocks={len(blocks)} "
        f"block_size={args.block_size} overlap={args.overlap} jobs={args.jobs}",
        flush=True,
    )
    block_work_root = output_dir / f"{args.position}_blockwise_work"
    block_results = _run_blocks(files, blocks, block_work_root, args)

    merged_result_dir = output_dir / f"{args.position}_RES"
    merge_report = merge_block_results(
        block_results=block_results,
        output_result_dir=merged_result_dir,
        output_digits=args.output_digits,
        min_iou=args.min_iou,
        min_overlap_frames=args.min_overlap_frames,
    )
    print(
        f"[BLOCKWISE] merged full result frames={merge_report['frames']} "
        f"tracks={merge_report['tracks']} output={merged_result_dir}",
        flush=True,
    )

    final_report = None
    if args.export_mode == "final-only":
        final_report = temporal_downsample_ctc_results(
            input_result_dir=merged_result_dir,
            output_result_dir=args.final_output_dir,
            source_root=args.source_root,
            sequence=args.sequence,
            source_frame_count=args.source_frame_count,
            pad_missing_with_empty=args.pad_missing_with_empty,
            factor=args.temporal_downsample_factor,
            offset=args.temporal_downsample_offset,
            output_digits=args.final_output_digits,
        )
        print(
            f"[BLOCKWISE] final-only wrote frames={final_report['frames']} "
            f"tracks={final_report['tracks']} output={args.final_output_dir}",
            flush=True,
        )

    if not args.keep_block_work:
        shutil.rmtree(block_work_root, ignore_errors=True)

    return {"merge": merge_report, "final": final_report}


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run tip tracking in overlapping blocks and stitch into one CTC result."
    )
    parser.add_argument("--mask-dir", required=True, type=Path, help="Folder with input segmentation mask TIFFs.")
    parser.add_argument("--mask-pattern", default=None, help="Input mask glob/suffix, e.g. '*_cp_masks.tif'.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Tracking output root.")
    parser.add_argument("--position", required=True, help="Output prefix; writes <position>_RES.")
    parser.add_argument("--tracking-script", default=script_dir / "ram_run_tiptracking_standalone_optimized.py", type=Path)
    parser.add_argument("--python", default=Path(sys.executable), type=Path, help="Python executable for block tracker runs.")
    parser.add_argument("--block-size", default=1000, type=int, help="Owned frames per block. Default: 1000.")
    parser.add_argument("--overlap", default=100, type=int, help="Extra frames on each side of a block run. Default: 100.")
    parser.add_argument("--jobs", default=1, type=int, help="Number of tracking blocks to run concurrently. Default: 1.")
    parser.add_argument("--min-iou", default=0.50, type=float, help="Minimum cumulative IoU for overlap stitching.")
    parser.add_argument("--min-overlap-frames", default=3, type=int, help="Minimum overlap frames with contact for stitching.")
    parser.add_argument("--time-series-threshold", default=1, type=int)
    parser.add_argument("--output-digits", default="auto")
    parser.add_argument("--io-workers", default=1, type=int)
    parser.add_argument("--io-queue-depth", default=4, type=int)
    parser.add_argument("--tiff-write-workers", default=4, type=int)
    parser.add_argument("--stack-storage", choices=["ram", "mmap"], default="ram")
    parser.add_argument("--mmap-dir", default=None, type=Path)
    parser.add_argument("--export-mode", choices=["full", "final-only"], default="full")
    parser.add_argument("--final-output-dir", default=None, type=Path)
    parser.add_argument("--source-root", default=None, type=Path)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--source-frame-count", default=None, type=int)
    parser.add_argument("--temporal-downsample-factor", default=1, type=int)
    parser.add_argument("--temporal-downsample-offset", default=0, type=int)
    parser.add_argument("--final-output-digits", default="auto")
    parser.add_argument("--pad-missing-with-empty", action="store_true")
    parser.add_argument("--keep-block-work", action="store_true")
    return parser.parse_args()


def _validate_args(args):
    if args.block_size < 1:
        raise ValueError("--block-size must be a positive integer.")
    if args.overlap < 0:
        raise ValueError("--overlap must be >= 0.")
    if args.overlap >= args.block_size:
        raise ValueError("--overlap must be smaller than --block-size.")
    if args.jobs < 1:
        raise ValueError("--jobs must be a positive integer.")
    if args.min_overlap_frames < 1:
        raise ValueError("--min-overlap-frames must be a positive integer.")
    if not (0 < args.min_iou <= 1):
        raise ValueError("--min-iou must be in the range (0, 1].")
    if not args.tracking_script.is_file():
        raise FileNotFoundError(f"tracking-script does not exist: {args.tracking_script}")
    if args.mmap_dir is not None and args.stack_storage != "mmap":
        raise ValueError("--mmap-dir requires --stack-storage mmap.")
    if args.export_mode == "final-only":
        if args.final_output_dir is None:
            raise ValueError("--export-mode final-only requires --final-output-dir.")
        if args.sequence is None:
            raise ValueError("--export-mode final-only requires --sequence.")
        if args.source_root is None and args.source_frame_count is None:
            raise ValueError("--export-mode final-only requires --source-root or --source-frame-count.")
        if args.temporal_downsample_factor < 1:
            raise ValueError("--temporal-downsample-factor must be >= 1.")
        if args.temporal_downsample_offset < 0:
            raise ValueError("--temporal-downsample-offset must be >= 0.")
    args.tracking_script = args.tracking_script.resolve()
    args.python = args.python.resolve()
    args.mmap_dir = args.mmap_dir.resolve() if args.mmap_dir is not None else None
    args.final_output_dir = args.final_output_dir.resolve() if args.final_output_dir is not None else None
    args.source_root = args.source_root.resolve() if args.source_root is not None else None


def main():
    args = parse_args()
    try:
        _validate_args(args)
        run_blockwise_tracking(args)
    except Exception as exc:
        print(f"[BLOCKWISE] FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
