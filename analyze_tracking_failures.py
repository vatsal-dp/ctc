#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

_mpl_config_dir = Path(tempfile.gettempdir()) / "ctc_matplotlib_cache"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from validate_ctc_result_format import _normalize_sequence, _natural_sort_key, resolve_digits


def _indexed_files(folder: Path, prefix: str, digits: int):
    if not folder.is_dir():
        return {}
    regex = re.compile(rf"^{re.escape(prefix)}(\d{{{digits}}})\.tiff?$", flags=re.IGNORECASE)
    indexed = {}
    for path in sorted(folder.glob(f"{prefix}*.tif"), key=lambda p: _natural_sort_key(p.name)):
        match = regex.match(path.name)
        if match is not None:
            indexed[int(match.group(1))] = path
    return indexed


def _read_mask(path: Path):
    arr = np.asarray(tifffile.imread(str(path)))
    if arr.ndim == 2:
        return arr.astype(np.int64, copy=False)
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        return arr[..., 0].astype(np.int64, copy=False)
    return arr.astype(np.int64, copy=False)


def _read_image(path: Path, fallback_shape):
    if path is None or not path.is_file():
        return np.zeros(fallback_shape[:2], dtype=np.float32)
    image = np.asarray(tifffile.imread(str(path)))
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        return image[..., :3]
    if image.ndim == 3:
        return image[0]
    return np.squeeze(image)


def _parse_track_file(path: Path):
    rows = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            label, begin, end, parent = [int(part) for part in line.split()[:4]]
            rows[label] = {"label": label, "begin": begin, "end": end, "parent": parent}
    return rows


def _labels(mask: np.ndarray):
    values = np.unique(mask)
    return values[values != 0].astype(int).tolist()


def _centroid(mask: np.ndarray, label: int):
    coords = np.argwhere(mask == label)
    if coords.size == 0:
        return None
    return tuple(float(value) for value in coords.mean(axis=0))


def _event(sequence, frame, event_type, gt_id="", res_id="", score="", details=""):
    return {
        "sequence": sequence,
        "frame": frame,
        "event_type": event_type,
        "gt_id": gt_id,
        "res_id": res_id,
        "score": score,
        "details": details,
    }


def _best_overlaps(source_mask: np.ndarray, target_mask: np.ndarray):
    matches = {}
    for label in _labels(source_mask):
        pixels = source_mask == label
        area = int(np.count_nonzero(pixels))
        if area == 0:
            continue
        target_values, counts = np.unique(target_mask[pixels], return_counts=True)
        candidates = [(int(target), int(count)) for target, count in zip(target_values, counts) if int(target) != 0]
        if not candidates:
            matches[label] = {"best_label": 0, "overlap": 0, "coverage": 0.0}
            continue
        best_label, overlap = max(candidates, key=lambda item: item[1])
        matches[label] = {
            "best_label": best_label,
            "overlap": overlap,
            "coverage": overlap / area,
        }
    return matches


def _iou_overlaps(gt_mask: np.ndarray, res_mask: np.ndarray):
    gt_labels = _labels(gt_mask)
    res_labels = _labels(res_mask)
    gt_areas = {label: int(np.count_nonzero(gt_mask == label)) for label in gt_labels}
    res_areas = {label: int(np.count_nonzero(res_mask == label)) for label in res_labels}
    pairs = {}

    for gt_label in gt_labels:
        pixels = gt_mask == gt_label
        values, counts = np.unique(res_mask[pixels], return_counts=True)
        for res_label, count in zip(values.tolist(), counts.tolist()):
            res_label = int(res_label)
            if res_label == 0:
                continue
            intersection = int(count)
            union = gt_areas[gt_label] + res_areas[res_label] - intersection
            pairs[(gt_label, res_label)] = intersection / union if union else 0.0

    return pairs, gt_labels, res_labels


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_thumbnail(
    out_path: Path,
    frame_index: int,
    image_path: Path | None,
    gt_mask: np.ndarray | None,
    res_mask: np.ndarray,
    frame_events,
):
    image = _read_image(image_path, res_mask.shape)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax in axes:
        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)
        ax.set_axis_off()

    axes[0].set_title(f"Frame {frame_index}: source")
    if gt_mask is not None:
        axes[1].imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap="Greens", alpha=0.45)
    axes[1].set_title("GT TRA/SEG overlay")
    axes[2].imshow(np.ma.masked_where(res_mask == 0, res_mask), cmap="magma", alpha=0.45)
    axes[2].set_title("Result overlay")

    summary = Counter(event["event_type"] for event in frame_events)
    fig.suptitle(", ".join(f"{key}={value}" for key, value in sorted(summary.items()))[:160])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def analyze_failures(
    dataset_root: Path,
    source_root: Path,
    sequence: str,
    out_dir: Path,
    digits_arg: str,
    coverage_threshold: float,
    iou_threshold: float,
    split_coverage_threshold: float,
    jump_pixels: float,
    jump_factor: float,
    max_thumbnails: int,
):
    sequence = _normalize_sequence(sequence)
    dataset_root = dataset_root.resolve()
    source_root = source_root.resolve()
    out_dir = out_dir.resolve()
    result_dir = dataset_root / f"{sequence}_RES"
    gt_tra_dir = source_root / f"{sequence}_GT" / "TRA"
    gt_seg_dir = source_root / f"{sequence}_GT" / "SEG"

    digits = resolve_digits(digits_arg, dataset_root, source_root, sequence)
    res_files = _indexed_files(result_dir, "mask", digits)
    gt_tra_files = _indexed_files(gt_tra_dir, "man_track", digits)
    gt_seg_files = _indexed_files(gt_seg_dir, "man_seg", digits)
    source_files = _indexed_files(source_root / sequence, "t", digits)

    if not res_files:
        raise FileNotFoundError(f"No result masks found in {result_dir}")
    if not gt_tra_files and not gt_seg_files:
        raise FileNotFoundError(f"No GT TRA or SEG masks found under {source_root / f'{sequence}_GT'}")

    common_tra_frames = sorted(set(res_files) & set(gt_tra_files))
    if not common_tra_frames:
        common_tra_frames = sorted(set(res_files) & set(gt_seg_files))
    if not common_tra_frames:
        raise RuntimeError("No common frames between result masks and GT annotations.")

    gt_track_rows = _parse_track_file(gt_tra_dir / "man_track.txt")
    res_track_rows = _parse_track_file(result_dir / "res_track.txt")

    events = []
    matched_res_by_gt_frame = {}
    matched_res_by_gt = defaultdict(list)
    res_centroids = defaultdict(list)
    frame_cache = {}

    for frame_index in common_tra_frames:
        res_mask = _read_mask(res_files[frame_index])
        gt_mask_path = gt_tra_files.get(frame_index) or gt_seg_files.get(frame_index)
        gt_mask = _read_mask(gt_mask_path)
        frame_cache[frame_index] = (gt_mask, res_mask)

        gt_matches = _best_overlaps(gt_mask, res_mask)
        res_matches = _best_overlaps(res_mask, gt_mask)

        gt_to_res = {}
        res_to_gts = defaultdict(list)
        for gt_label, match in gt_matches.items():
            best_res = int(match["best_label"])
            coverage = float(match["coverage"])
            if best_res == 0:
                events.append(
                    _event(sequence, frame_index, "missed_gt_object", gt_id=gt_label, score="0")
                )
                continue
            if coverage < coverage_threshold:
                events.append(
                    _event(
                        sequence,
                        frame_index,
                        "low_gt_marker_coverage",
                        gt_id=gt_label,
                        res_id=best_res,
                        score=f"{coverage:.4f}",
                    )
                )
                continue

            gt_to_res[gt_label] = best_res
            res_to_gts[best_res].append(gt_label)
            matched_res_by_gt_frame[(gt_label, frame_index)] = best_res
            matched_res_by_gt[gt_label].append((frame_index, best_res))

        for res_label, match in res_matches.items():
            if int(match["best_label"]) == 0:
                events.append(
                    _event(
                        sequence,
                        frame_index,
                        "extra_result_no_gt_marker",
                        res_id=res_label,
                        score="0",
                        details="Marker-based check; confirm visually if GT TRA markers are sparse.",
                    )
                )

        for res_label, gt_labels in res_to_gts.items():
            if len(gt_labels) > 1:
                events.append(
                    _event(
                        sequence,
                        frame_index,
                        "many_gt_to_one_result_merge",
                        gt_id=",".join(str(label) for label in sorted(gt_labels)),
                        res_id=res_label,
                        details=f"{len(gt_labels)} GT labels matched one result label.",
                    )
                )

        for gt_label in _labels(gt_mask):
            pixels = gt_mask == gt_label
            values, counts = np.unique(res_mask[pixels], return_counts=True)
            matched_result_parts = [
                int(value)
                for value, count in zip(values.tolist(), counts.tolist())
                if int(value) != 0 and (count / max(1, int(np.count_nonzero(pixels)))) >= split_coverage_threshold
            ]
            if len(matched_result_parts) > 1:
                events.append(
                    _event(
                        sequence,
                        frame_index,
                        "one_gt_to_many_results_split",
                        gt_id=gt_label,
                        res_id=",".join(str(label) for label in sorted(matched_result_parts)),
                        details=f"{len(matched_result_parts)} result labels cover one GT marker/segment.",
                    )
                )

        if frame_index in gt_seg_files:
            gt_seg = _read_mask(gt_seg_files[frame_index])
            iou_pairs, gt_seg_labels, res_labels = _iou_overlaps(gt_seg, res_mask)
            for gt_label in gt_seg_labels:
                best = max((iou for (candidate_gt, _), iou in iou_pairs.items() if candidate_gt == gt_label), default=0.0)
                if best < iou_threshold:
                    events.append(
                        _event(
                            sequence,
                            frame_index,
                            "low_segmentation_iou",
                            gt_id=gt_label,
                            score=f"{best:.4f}",
                            details="Computed against GT SEG frame.",
                        )
                    )
            for res_label in res_labels:
                best = max((iou for (_, candidate_res), iou in iou_pairs.items() if candidate_res == res_label), default=0.0)
                if best < iou_threshold:
                    events.append(
                        _event(
                            sequence,
                            frame_index,
                            "extra_or_low_iou_result_object",
                            res_id=res_label,
                            score=f"{best:.4f}",
                            details="Computed against GT SEG frame.",
                        )
                    )

        for res_label in _labels(res_mask):
            center = _centroid(res_mask, res_label)
            if center is not None:
                res_centroids[res_label].append((frame_index, center))

    for gt_label, assignments in sorted(matched_res_by_gt.items()):
        assignments = sorted(assignments)
        previous_frame = None
        previous_res = None
        seen_res = []
        for frame_index, res_label in assignments:
            if previous_frame is not None and frame_index == previous_frame + 1 and previous_res != res_label:
                events.append(
                    _event(
                        sequence,
                        frame_index,
                        "id_switch",
                        gt_id=gt_label,
                        res_id=res_label,
                        details=f"Previous result label was {previous_res} at frame {previous_frame}.",
                    )
                )
            previous_frame = frame_index
            previous_res = res_label
            seen_res.append(res_label)
        if len(set(seen_res)) > 1:
            events.append(
                _event(
                    sequence,
                    assignments[0][0],
                    "fragmented_gt_track",
                    gt_id=gt_label,
                    res_id=",".join(str(label) for label in sorted(set(seen_res))),
                    details=f"GT track matched {len(set(seen_res))} result IDs over time.",
                )
            )

    for res_label, centers in sorted(res_centroids.items()):
        centers = sorted(centers)
        steps = []
        for (frame_a, center_a), (frame_b, center_b) in zip(centers, centers[1:]):
            if frame_b != frame_a + 1:
                continue
            distance = math.dist(center_a, center_b)
            steps.append((frame_b, distance))
        positive_distances = [distance for _, distance in steps if distance > 0]
        median_step = float(np.median(positive_distances)) if positive_distances else 0.0
        threshold = max(jump_pixels, median_step * jump_factor)
        for frame_index, distance in steps:
            if distance > threshold:
                events.append(
                    _event(
                        sequence,
                        frame_index,
                        "suspicious_centroid_jump",
                        res_id=res_label,
                        score=f"{distance:.2f}",
                        details=f"Threshold={threshold:.2f}; median_step={median_step:.2f}.",
                    )
                )

    for gt_label, row in sorted(gt_track_rows.items()):
        parent_id = int(row["parent"])
        if parent_id == 0:
            continue
        parent_row = gt_track_rows.get(parent_id)
        if parent_row is None:
            continue
        child_frame = int(row["begin"])
        parent_frame = int(parent_row["end"])
        res_child = matched_res_by_gt_frame.get((gt_label, child_frame))
        res_parent = matched_res_by_gt_frame.get((parent_id, parent_frame))
        if res_child is None or res_parent is None:
            events.append(
                _event(
                    sequence,
                    child_frame,
                    "division_missing_result_mapping",
                    gt_id=gt_label,
                    details=f"GT parent={parent_id}; res_child={res_child}; res_parent={res_parent}.",
                )
            )
            continue
        res_child_row = res_track_rows.get(res_child)
        observed_parent = None if res_child_row is None else int(res_child_row["parent"])
        if observed_parent != res_parent:
            events.append(
                _event(
                    sequence,
                    child_frame,
                    "wrong_division_parent",
                    gt_id=gt_label,
                    res_id=res_child,
                    details=f"Expected result parent {res_parent}, observed {observed_parent}.",
                )
            )

    event_fields = ["sequence", "frame", "event_type", "gt_id", "res_id", "score", "details"]
    _write_csv(out_dir / "failure_events.csv", events, event_fields)

    frame_counts = Counter((event["frame"], event["event_type"]) for event in events)
    frame_rows = [
        {"sequence": sequence, "frame": frame, "event_type": event_type, "count": count}
        for (frame, event_type), count in sorted(frame_counts.items())
    ]
    _write_csv(out_dir / "failure_summary_by_frame.csv", frame_rows, ["sequence", "frame", "event_type", "count"])

    track_counts = Counter(
        (event["gt_id"] if event["gt_id"] != "" else f"res:{event['res_id']}", event["event_type"])
        for event in events
    )
    track_rows = [
        {"sequence": sequence, "track": track, "event_type": event_type, "count": count}
        for (track, event_type), count in sorted(track_counts.items(), key=lambda item: (str(item[0][0]), item[0][1]))
    ]
    _write_csv(out_dir / "failure_summary_by_track.csv", track_rows, ["sequence", "track", "event_type", "count"])

    top_frames = [
        frame for frame, _ in Counter(event["frame"] for event in events).most_common(max_thumbnails)
        if isinstance(frame, int)
    ]
    events_by_frame = defaultdict(list)
    for event in events:
        events_by_frame[event["frame"]].append(event)

    for frame_index in top_frames:
        gt_mask, res_mask = frame_cache.get(frame_index, (None, None))
        if res_mask is None:
            res_mask = _read_mask(res_files[frame_index])
        if gt_mask is None:
            gt_path = gt_tra_files.get(frame_index) or gt_seg_files.get(frame_index)
            gt_mask = _read_mask(gt_path) if gt_path is not None else None
        image_path = source_files.get(frame_index)
        _save_thumbnail(
            out_dir / "thumbnails" / f"frame_{frame_index:0{digits}d}.png",
            frame_index,
            image_path,
            gt_mask,
            res_mask,
            events_by_frame[frame_index],
        )

    return {
        "digits": digits,
        "frames_analyzed": len(common_tra_frames),
        "events": len(events),
        "out_dir": out_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a local failure report for CTC tracking results.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Root containing <sequence>_RES.")
    parser.add_argument("--source-root", required=True, type=Path, help="Original CTC dataset root containing GT.")
    parser.add_argument("--sequence", required=True, type=str, help="CTC sequence ID, e.g. 01 or 02.")
    parser.add_argument("--out", required=True, type=Path, help="Output folder for CSVs and thumbnails.")
    parser.add_argument("--digits", default="auto", choices=["auto", "3", "4"], help="CTC time index digits.")
    parser.add_argument("--coverage-threshold", default=0.5, type=float, help="GT marker/segment coverage threshold.")
    parser.add_argument("--iou-threshold", default=0.5, type=float, help="GT SEG IoU threshold.")
    parser.add_argument("--split-coverage-threshold", default=0.2, type=float)
    parser.add_argument("--jump-pixels", default=50.0, type=float)
    parser.add_argument("--jump-factor", default=5.0, type=float)
    parser.add_argument("--max-thumbnails", default=25, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    report = analyze_failures(
        dataset_root=args.dataset_root,
        source_root=args.source_root,
        sequence=args.sequence,
        out_dir=args.out,
        digits_arg=args.digits,
        coverage_threshold=args.coverage_threshold,
        iou_threshold=args.iou_threshold,
        split_coverage_threshold=args.split_coverage_threshold,
        jump_pixels=args.jump_pixels,
        jump_factor=args.jump_factor,
        max_thumbnails=args.max_thumbnails,
    )
    print(
        "[FAILURE ANALYSIS] wrote "
        f"{report['events']} events across {report['frames_analyzed']} frames to {report['out_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
