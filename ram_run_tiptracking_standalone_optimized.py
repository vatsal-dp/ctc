#!/usr/bin/env python3

import argparse
import concurrent.futures
import gc
import os
import queue
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import convolve, rotate
from scipy.stats import mode
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, erosion, opening, skeletonize
from skimage.transform import resize


_MAX_CTC_TRACK_ID = int(np.iinfo(np.uint16).max)
_TRACKING_DTYPE = np.uint32


# ---------------------------------------------------------------------------
# Prefetch I/O queue  (OPT-1)
# Reads the next N mask TIFFs from disk on a background thread so the main
# tracking loop never waits for disk.  Uses a bounded queue so memory stays
# under control regardless of dataset size.
# ---------------------------------------------------------------------------

class _PrefetchQueue:
    """Background-thread TIFF prefetcher.

    Yields frames in order via ``__iter__``.  Stops cleanly even if the
    consumer exits early.  ``queue_depth`` controls how many decoded numpy
    arrays are held in RAM at once (default 4 – tune up if disk is the
    bottleneck, down on memory-constrained machines).
    """

    _SENTINEL = object()

    def __init__(
        self,
        files: list[Path],
        resiz_factor: float,
        queue_depth: int = 4,
        io_workers: int = 1,
    ):
        self._files = files
        self._resiz_factor = resiz_factor
        self._q: queue.Queue = queue.Queue(maxsize=queue_depth)
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        if io_workers > 1:
            # Parallel I/O with a thread pool; results re-ordered via per-slot
            # futures list so frame order is always preserved.
            self._threads.append(
                threading.Thread(target=self._producer_parallel,
                                 args=(io_workers,), daemon=True)
            )
        else:
            self._threads.append(
                threading.Thread(target=self._producer_serial, daemon=True)
            )
        for t in self._threads:
            t.start()

    def _read_one(self, path: Path) -> np.ndarray:
        return _read_mask_frame(path, self._resiz_factor)

    def _producer_serial(self) -> None:
        try:
            for path in self._files:
                if self._stop_event.is_set():
                    break
                frame = self._read_one(path)
                self._q.put(frame)
        finally:
            self._q.put(self._SENTINEL)

    def _producer_parallel(self, workers: int) -> None:
        """Submit reads to a thread pool; put results in order into the queue."""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                # Use a sliding window of futures to limit in-flight reads.
                window = workers * 2
                futures = []
                file_iter = iter(self._files)

                def _submit_next():
                    try:
                        p = next(file_iter)
                        futures.append(ex.submit(self._read_one, p))
                    except StopIteration:
                        pass

                # Seed the window.
                for _ in range(min(window, len(self._files))):
                    _submit_next()

                while futures:
                    if self._stop_event.is_set():
                        break
                    fut = futures.pop(0)
                    frame = fut.result()
                    _submit_next()
                    self._q.put(frame)
        finally:
            self._q.put(self._SENTINEL)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                break
            yield item

    def stop(self) -> None:
        self._stop_event.set()
        # Drain so the producer thread can unblock and exit.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        for t in self._threads:
            t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Vectorised mode helper  (OPT-2)
# scipy.stats.mode is O(N log N) and allocates a new object each call.
# For the common case of an integer array we can use np.bincount which is O(N).
# ---------------------------------------------------------------------------

def _fast_mode_uint(arr: np.ndarray) -> int:
    """Return the most frequent value in a non-empty 1-D integer array."""
    if arr.size == 0:
        raise ValueError("Empty array has no mode")
    bc = np.bincount(arr.ravel().astype(np.intp))
    return int(bc.argmax())


# ---------------------------------------------------------------------------
# Vectorised label compaction  (OPT-3)
# _compact_labels_in_place processes one frame at a time with a Python loop.
# This version builds a global LUT over the full stack in one pass and then
# applies it with numpy fancy indexing, cutting frame-by-frame overhead.
# ---------------------------------------------------------------------------

def _compact_labels_in_place_fast(mask_stack: np.ndarray) -> tuple[int, int]:
    """In-place compaction using a single global LUT – faster for large stacks."""
    labels = np.unique(mask_stack)
    labels = labels[labels != 0].astype(np.int64, copy=False)
    if labels.size == 0:
        return 0, 0

    max_label_before = int(labels[-1])

    # Build LUT: old_id → new_id (0 stays 0)
    lut = np.zeros(max_label_before + 1, dtype=mask_stack.dtype)
    for new_id, old_id in enumerate(labels, start=1):
        lut[old_id] = new_id

    n_frames = int(mask_stack.shape[2])
    for frame_idx in range(n_frames):
        frame = mask_stack[:, :, frame_idx]
        if not np.any(frame):
            continue
        # Clip to LUT bounds (guard against any stray IDs above max_label_before)
        clipped = np.clip(frame, 0, max_label_before)
        mask_stack[:, :, frame_idx] = lut[clipped]

    return int(labels.size), max_label_before


# ---------------------------------------------------------------------------
# Vectorised temporal-presence table  (OPT-4)
# Building tp_im one frame at a time with np.bincount already is decent, but
# we can avoid repeated Python-level loops by writing directly into tp_im via
# slices and reduce function-call overhead.
# ---------------------------------------------------------------------------

def _build_tp_im(mask_stack: np.ndarray, max_id: int) -> np.ndarray:
    """Build the (max_id, n_frames) temporal-presence pixel-count table."""
    n_frames = int(mask_stack.shape[2])
    tp_im = np.zeros((max_id, n_frames), dtype=np.uint32)
    for ih in range(n_frames):
        if ih % 1000 == 0 or ih == n_frames - 1:
            print(f"[TRACKING] post frame {ih + 1}/{n_frames}", flush=True)
        counts = np.bincount(mask_stack[:, :, ih].ravel(), minlength=max_id + 1)
        tp_im[:, ih] = counts[1:max_id + 1].astype(np.uint32)
    return tp_im


# ---------------------------------------------------------------------------
# Parallel TIFF writer  (OPT-5)
# Writing 27 k TIFFs one-at-a-time with a Python loop leaves the disk idle
# between calls.  A thread pool lets several writes fly concurrently.
# ---------------------------------------------------------------------------

def _write_tiff_parallel(
    result_dir: Path,
    tensor: np.ndarray,
    output_digits: int,
    n_workers: int = 4,
) -> None:
    """Write all frames as uint16 TIFFs using a thread-pool."""
    total = int(tensor.shape[2])

    def _write_one(args):
        frame_idx, path, data = args
        tifffile.imwrite(str(path), data)

    tasks = []
    for frame_idx in range(total):
        path = result_dir / f"mask{frame_idx:0{output_digits}d}.tif"
        data = tensor[:, :, frame_idx].astype(np.uint16, copy=False)
        tasks.append((frame_idx, path, data))

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(_write_one, tasks))


# ---------------------------------------------------------------------------
# Bounding-box cache for active cells  (OPT-6)
# The inner per-cell loop repeatedly scans the full frame to extract `is5`
# (mask == it1).  On a 2 k × 2 k frame with 200 cells that is 200 full-frame
# scans per frame.  Caching bounding boxes cuts the scan region dramatically.
# ---------------------------------------------------------------------------

def _build_bbox_cache(frame: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    """Return {label: (r_min, r_max, c_min, c_max)} for all nonzero labels.

    Uses regionprops which is a single pass over the frame.
    """
    props = regionprops(frame.astype(np.int32))
    return {p.label: p.bbox for p in props}  # bbox = (min_row, min_col, max_row, max_col)


# ---------------------------------------------------------------------------
# Vectorised overlap scoring  (OPT-7)
# The inner loop over `uniq` candidates builds pix_si row-by-row with
# np.sum(is6a == iu) – one full-array scan per candidate.  We replace it with
# a single np.bincount call.
# ---------------------------------------------------------------------------

def _overlap_scores(is6a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique_labels, fraction_of_total) for all nonzero values in is6a."""
    flat = is6a.ravel()
    nz = flat[flat != 0]
    if nz.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    bc = np.bincount(nz.astype(np.intp))
    ids = np.where(bc > 0)[0]
    ids = ids[ids != 0]
    counts = bc[ids].astype(float)
    fracs = counts / counts.sum()
    return ids.astype(np.int64), fracs


# ---------------------------------------------------------------------------
# Vectorised track-id remap  (OPT-8)
# _reindex_tracks_compact loops over every old_id with frame[frame==old_id]=new.
# For N ids and F frames that is N×F full-array scans.  One LUT + fancy index
# is O(pixels) regardless of N.
# ---------------------------------------------------------------------------

def _remap_stack_with_lut(
    mask_stack: np.ndarray,
    lut: np.ndarray,
) -> None:
    """Apply `lut[pixel] → new_pixel` in-place across all frames.

    lut must be indexed by old pixel value (0 → 0 by convention).
    """
    n_frames = int(mask_stack.shape[2])
    lut_max = len(lut) - 1
    for frame_idx in range(n_frames):
        frame = mask_stack[:, :, frame_idx]
        if not np.any(frame):
            continue
        clipped = np.clip(frame, 0, lut_max)
        mask_stack[:, :, frame_idx] = lut[clipped]


def _looks_like_network_path(path: Path) -> bool:
    text = str(path)
    return text.startswith("\\\\") or text.startswith("//")


def _format_gib(byte_count: int | float) -> str:
    return f"{byte_count / (1024**3):.2f} GiB"


def _default_mmap_candidates(output_dir: Path) -> list[Path]:
    if not _looks_like_network_path(output_dir):
        return [output_dir]

    candidates = [Path(tempfile.gettempdir()) / "tiptracking_mmap"]
    if os.name == "nt":
        temp_drive = Path(tempfile.gettempdir()).drive.upper()
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZC":
            drive = f"{letter}:"
            if drive == temp_drive:
                continue
            candidates.append(Path(f"{drive}\\tiptracking_mmap"))

    return candidates


def _probe_mmap_dir(candidate: Path) -> tuple[int | None, OSError | None]:
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(candidate).free, None
    except OSError as exc:
        return None, exc


def _choose_mmap_dir(output_dir: Path, mmap_dir: Path | None, estimated_stack_bytes: int) -> Path:
    candidates = [mmap_dir] if mmap_dir is not None else _default_mmap_candidates(output_dir)
    rejected: list[str] = []

    for candidate in candidates:
        free_bytes, error = _probe_mmap_dir(candidate)
        if error is not None:
            rejected.append(f"{candidate} unavailable ({error})")
            continue
        if free_bytes is not None and free_bytes < estimated_stack_bytes:
            rejected.append(
                f"{candidate} has only {_format_gib(free_bytes)} free"
            )
            continue

        mmap_dir = candidate
        break
    else:
        details = "; ".join(rejected) if rejected else "no candidates were usable"
        raise RuntimeError(
            f"No temporary mmap directory has enough free space for the "
            f"{_format_gib(estimated_stack_bytes)} tracking stack: {details}. "
            "Pass --mmap-dir pointing to a local scratch disk with enough space, "
            "for example --mmap-dir D:\\tiptracking_mmap."
        )

    if mmap_dir != output_dir:
        free_text = f" free={_format_gib(free_bytes)}" if free_bytes is not None else ""
        print(
            f"[TRACKING] low-memory mode: using mmap scratch dir={mmap_dir}{free_text}",
            flush=True,
        )

    return mmap_dir


def _natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _ctc_time_digits_from_name(filename: str):
    match = re.match(r"^(?:mask|t|man_track|man_seg)(\d+)\.tiff?$", filename, flags=re.IGNORECASE)
    if match is None:
        return None

    return len(match.group(1))


def _minimum_ctc_digit_width(frame_count: int) -> int:
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative.")
    highest_index = max(frame_count - 1, 0)
    return max(3, len(str(highest_index)))


def _parse_digit_arg(digits_arg: str, option_name: str) -> int:
    try:
        digits = int(digits_arg)
    except ValueError as exc:
        raise ValueError(f"{option_name} must be 'auto' or a positive integer.") from exc
    if digits < 1:
        raise ValueError(f"{option_name} must be a positive integer.")
    return digits


def _resolve_output_digits(output_digits: str, input_files: list[Path], frame_count: int) -> int:
    required_digits = _minimum_ctc_digit_width(frame_count)

    if output_digits != "auto":
        digits = _parse_digit_arg(output_digits, "--output-digits")
    else:
        inferred = {_ctc_time_digits_from_name(path.name) for path in input_files}
        inferred.discard(None)
        if len(inferred) == 1:
            digits = max(required_digits, int(next(iter(inferred))))
        else:
            digits = required_digits

    if frame_count > 10**digits:
        raise ValueError(
            f"Cannot export {frame_count} frames with {digits} digits. "
            f"Use --output-digits {required_digits} or higher."
        )
    return digits


def binar(mask):
    # OPT: avoid full copy; boolean cast then view as uint avoids an allocation
    return (mask != 0).view(np.uint8)


def imopen_binary(mask, selem):
    return opening(mask.astype(bool), selem)


def imopen_labels(mask, selem):
    if mask.dtype == bool or mask.max() <= 1:
        return opening(mask.astype(bool), selem)

    result = np.zeros_like(mask)
    labels = np.unique(mask[mask != 0])
    for label_val in labels:
        binary_mask = mask == label_val
        opened = opening(binary_mask, selem)
        result[opened] = label_val
    return result


def find_endpoints(skel):
    skel = skel.astype(bool)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return skel & (neighbor_count == 1)


def find_branchpoints(skel):
    skel = skel.astype(bool)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return skel & (neighbor_count > 2)


def _infer_parent_id(
    prev_frame: np.ndarray,
    child_mask: np.ndarray,
    child_track_id: int,
    valid_track_ids: set[int],
    min_touch_pixels: int = 10,
    min_touch_ratio: float = 0.05,
) -> int:
    child_pixels = int(np.count_nonzero(child_mask))
    if child_pixels == 0:
        return 0

    dilated_child = binary_dilation(child_mask.astype(bool), np.ones((3, 3), dtype=bool))
    touching_labels = prev_frame[dilated_child]
    touching_labels = touching_labels[touching_labels != 0]
    if touching_labels.size == 0:
        return 0

    labels, counts = np.unique(touching_labels.astype(np.int64), return_counts=True)
    candidates = []
    for label_val, count in zip(labels.tolist(), counts.tolist()):
        parent_id = int(label_val)
        touch_count = int(count)
        if parent_id == child_track_id:
            continue
        if parent_id not in valid_track_ids:
            continue
        candidates.append((touch_count, parent_id))

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
    min_touch_pixels: int = 10,
    min_touch_ratio: float = 0.05,
) -> bool:
    previous_mask = prev_frame == track_id
    current_mask = curr_frame == track_id
    current_pixels = int(np.count_nonzero(current_mask))
    if current_pixels == 0 or not np.any(previous_mask):
        return False

    dilated_previous = binary_dilation(previous_mask.astype(bool), np.ones((3, 3), dtype=bool))
    touch_pixels = int(np.count_nonzero(current_mask & dilated_previous))
    if touch_pixels < min_touch_pixels:
        return False
    return (touch_pixels / current_pixels) >= min_touch_ratio


def _fork_existing_daughter_label(
    normalized_tensor: np.ndarray,
    child_id: int,
    frame_idx: int,
    new_child_id: int,
) -> None:
    for time_idx in range(frame_idx, int(normalized_tensor.shape[2])):
        frame_t = normalized_tensor[:, :, time_idx]
        child_pixels = frame_t == child_id
        if np.any(child_pixels):
            frame_t[child_pixels] = new_child_id


def _track_ids_in_frame(frame: np.ndarray):
    ids = np.unique(frame)
    return ids[ids != 0].astype(int)


def _active_cooldown_track_ids(cooldown_until: dict[int, int], frame_idx: int):
    return {track_id for track_id, end_frame in cooldown_until.items() if frame_idx <= end_frame}


# ---------------------------------------------------------------------------
# Identity-continuity rescue  (general, multi-signal, confidence-based)
# ---------------------------------------------------------------------------

def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return (row, col) centroid of nonzero pixels, or (-1, -1) if empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (-1.0, -1.0)
    return (float(ys.mean()), float(xs.mean()))


def _compute_rescue_confidence(
    ref_mask: np.ndarray,
    cand_mask: np.ndarray,
    gap_frames: int,
    is_under_division_cooldown: bool,
    max_centroid_dist_px: float = 50.0,
) -> float:
    """Return a [0, 1] confidence that *cand_mask* is the same cell as *ref_mask*.

    Signals used:
      - overlap / touch (dilated contact pixels)
      - IoU
      - centroid distance (penalised by gap length)
      - area similarity
      - division-cooldown prior (soft boost)

    Returns 0.0 immediately when hard exclusion criteria fail.
    """
    ref_px = int(np.count_nonzero(ref_mask))
    cand_px = int(np.count_nonzero(cand_mask))
    if ref_px == 0 or cand_px == 0:
        return 0.0

    # --- overlap / touch ---
    dilated_ref = binary_dilation(ref_mask.astype(bool), np.ones((3, 3), dtype=bool))
    touch_px = int(np.count_nonzero(cand_mask.astype(bool) & dilated_ref))
    touch_score = min(touch_px / max(ref_px, cand_px), 1.0)  # 0 → 1

    # Direct overlap
    intersection = int(np.count_nonzero(ref_mask.astype(bool) & cand_mask.astype(bool)))
    union = ref_px + cand_px - intersection
    iou = intersection / union if union > 0 else 0.0

    # --- centroid distance (normalised; penalised for multi-frame gaps) ---
    r_cy, r_cx = _mask_centroid(ref_mask)
    c_cy, c_cx = _mask_centroid(cand_mask)
    dist = float(np.hypot(c_cy - r_cy, c_cx - r_cx))
    # Allow proportionally more drift when there is a gap
    effective_max = max_centroid_dist_px * (1 + 0.5 * (gap_frames - 1))
    dist_score = max(0.0, 1.0 - dist / effective_max)

    # --- area similarity ---
    area_ratio = min(ref_px, cand_px) / max(ref_px, cand_px)  # (0, 1]
    area_score = area_ratio  # large difference → low score

    # --- weighted combination ---
    # Touch/IoU dominate; distance and area are supporting evidence.
    score = (
        0.35 * touch_score
        + 0.30 * iou
        + 0.20 * dist_score
        + 0.15 * area_score
    )

    # Soft boost for post-division daughters (they're more likely to be mislabelled)
    if is_under_division_cooldown:
        score = min(score * 1.15, 1.0)

    return float(score)


def _rescue_identity_continuity(
    normalized_tensor: np.ndarray,
    cooldown_until: dict[int, int],
    max_rescue_gap: int = 1,
    rescue_confidence_threshold: float = 0.30,
    max_centroid_dist_px: float = 50.0,
    excluded_lost_ids: set[int] | None = None,
    audit_log: list | None = None,
) -> None:
    """General identity-continuity rescue pass (operates in-place).

    For every track that disappears, look for a new label in the next
    ``max_rescue_gap`` frames.  Score each candidate with
    ``_compute_rescue_confidence``.  If exactly one candidate clears
    ``rescue_confidence_threshold`` and no competing candidate comes
    within 80 % of its score, relabel the candidate back to the
    original track ID for all remaining frames.

    This handles:
    - post-division daughter label swaps (aided by cooldown soft prior)
    - general tracker relabelling unrelated to division

    Parameters
    ----------
    normalized_tensor:
        H × W × T uint array, modified **in-place**.
    cooldown_until:
        Maps track_id → last frame of division-cooldown protection.
    max_rescue_gap:
        How many frames ahead to search for a replacement label.
        1 = only the immediately following frame; 3 is a reasonable upper bound.
    rescue_confidence_threshold:
        Minimum confidence to accept a rescue candidate.
    max_centroid_dist_px:
        Hard spatial limit on centroid distance (passed to scorer).
    audit_log:
        If a list is supplied, dicts describing each rescue/skip event are
        appended so the caller can inspect or persist them.
    """
    frame_count = int(normalized_tensor.shape[2])
    if audit_log is None:
        audit_log = []
    if excluded_lost_ids is None:
        excluded_lost_ids = set()

    # Track which IDs have already been used as a rescue target in this pass
    # to prevent one candidate label from being claimed by two lost tracks.
    claimed_newborn_ids: set[int] = set()

    for frame_idx in range(1, frame_count):
        prev_frame = normalized_tensor[:, :, frame_idx - 1]
        curr_frame = normalized_tensor[:, :, frame_idx]

        prev_ids = set(_track_ids_in_frame(prev_frame).tolist())
        curr_ids = set(_track_ids_in_frame(curr_frame).tolist())

        # Tracks present in prev but gone in curr are candidates for rescue.
        disappeared = prev_ids - curr_ids
        if not disappeared:
            continue

        for lost_id in sorted(disappeared):
            if lost_id in excluded_lost_ids:
                audit_log.append({
                    "event": "skip_excluded_parent",
                    "lost_id": lost_id,
                    "frame": frame_idx,
                })
                continue

            ref_mask = prev_frame == lost_id
            if not np.any(ref_mask):
                continue

            is_cooldown = lost_id in cooldown_until and frame_idx <= cooldown_until[lost_id]

            best_score = 0.0
            best_newborn_id = None
            second_score = 0.0

            for gap in range(1, max_rescue_gap + 1):
                search_frame_idx = frame_idx + gap - 1
                if search_frame_idx >= frame_count:
                    break

                search_frame = normalized_tensor[:, :, search_frame_idx]
                search_ids = set(_track_ids_in_frame(search_frame).tolist())

                # Newborn = present in search frame but not in prev_frame
                newborn_ids_here = [
                    nid for nid in search_ids
                    if nid not in prev_ids and nid not in claimed_newborn_ids
                ]

                for nid in newborn_ids_here:
                    cand_mask = search_frame == nid
                    score = _compute_rescue_confidence(
                        ref_mask=ref_mask,
                        cand_mask=cand_mask,
                        gap_frames=gap,
                        is_under_division_cooldown=is_cooldown,
                        max_centroid_dist_px=max_centroid_dist_px,
                    )
                    if score > best_score:
                        second_score = best_score
                        best_score = score
                        best_newborn_id = nid
                    elif score > second_score:
                        second_score = score

            if best_newborn_id is None or best_score < rescue_confidence_threshold:
                audit_log.append({
                    "event": "skip_no_candidate",
                    "lost_id": lost_id,
                    "frame": frame_idx,
                    "best_score": round(best_score, 4),
                    "threshold": rescue_confidence_threshold,
                    "cooldown": is_cooldown,
                })
                continue

            # Ambiguity check: reject if runner-up is within 80 % of best score
            ambiguous = second_score >= 0.80 * best_score and second_score >= rescue_confidence_threshold
            if ambiguous:
                audit_log.append({
                    "event": "skip_ambiguous",
                    "lost_id": lost_id,
                    "frame": frame_idx,
                    "best_candidate": best_newborn_id,
                    "best_score": round(best_score, 4),
                    "second_score": round(second_score, 4),
                    "cooldown": is_cooldown,
                })
                continue

            # Accept the rescue: relabel best_newborn_id → lost_id for all
            # remaining frames so the track is continuous.
            frames_updated = 0
            for time_idx in range(frame_idx, frame_count):
                frame_t = normalized_tensor[:, :, time_idx]
                rescue_pixels = frame_t == best_newborn_id
                if np.any(rescue_pixels):
                    frame_t[rescue_pixels] = lost_id
                    frames_updated += 1

            claimed_newborn_ids.add(best_newborn_id)
            audit_log.append({
                "event": "rescued",
                "lost_id": lost_id,
                "rescued_from": best_newborn_id,
                "frame": frame_idx,
                "best_score": round(best_score, 4),
                "second_score": round(second_score, 4),
                "cooldown": is_cooldown,
                "frames_updated": frames_updated,
            })


def _log_rescue_summary(audit_log: list, label: str = "identity-continuity rescue") -> None:
    """Print a compact human-readable summary of an audit log."""
    rescued = [e for e in audit_log if e["event"] == "rescued"]
    ambiguous = [e for e in audit_log if e["event"] == "skip_ambiguous"]
    no_cand = [e for e in audit_log if e["event"] == "skip_no_candidate"]
    excluded = [e for e in audit_log if e["event"] == "skip_excluded_parent"]
    print(
        f"[TRACKING] {label}: rescued={len(rescued)} "
        f"ambiguous_skipped={len(ambiguous)} no_candidate_skipped={len(no_cand)} "
        f"division_parent_skipped={len(excluded)}",
        flush=True,
    )
    for e in rescued:
        cd_tag = " [cooldown]" if e.get("cooldown") else ""
        print(
            f"[TRACKING]   RESCUED  frame={e['frame']} "
            f"lost_id={e['lost_id']} ← {e['rescued_from']} "
            f"score={e['best_score']:.3f}{cd_tag} "
            f"frames_updated={e['frames_updated']}",
            flush=True,
        )
    for e in ambiguous:
        print(
            f"[TRACKING]   AMBIGUOUS frame={e['frame']} "
            f"lost_id={e['lost_id']} best={e['best_candidate']} "
            f"score={e['best_score']:.3f} second={e['second_score']:.3f}",
            flush=True,
        )


def _normalize_ctc_divisions(
    final_tracked_tensor: np.ndarray,
    division_cooldown_frames: int = 20,
    identity_rescue_gap: int = 1,
    rescue_confidence_threshold: float = 0.30,
    max_centroid_dist_px: float = 50.0,
):
    if division_cooldown_frames < 0:
        raise ValueError("division_cooldown_frames must be >= 0.")
    if identity_rescue_gap < 0:
        raise ValueError("identity_rescue_gap must be >= 0.")

    max_uint16 = int(np.iinfo(np.uint16).max)
    max_input_id = int(np.max(final_tracked_tensor)) if final_tracked_tensor.size else 0
    if max_input_id > max_uint16:
        raise ValueError("Track IDs exceed uint16 capacity required for challenge mask export.")

    normalized_tensor = final_tracked_tensor
    frame_count = int(normalized_tensor.shape[2])
    max_track_id = int(np.max(normalized_tensor)) if normalized_tensor.size else 0
    parent_map = {}
    cooldown_until: dict[int, int] = {}

    # -----------------------------------------------------------------------
    # Stage 1 (optional): General identity-continuity rescue BEFORE division
    # normalisation.  This handles label swaps unrelated to cell divisions as
    # well as post-division swaps that the cooldown mechanism would miss.
    # It runs first so that the division detection in Stage 2 sees a cleaner
    # label field.
    # -----------------------------------------------------------------------
    if identity_rescue_gap > 0:
        print("[TRACKING] export: running identity-continuity rescue (pre-division pass)", flush=True)
        pre_audit: list = []
        _rescue_identity_continuity(
            normalized_tensor=normalized_tensor,
            # Pass an empty cooldown map on the first pass; divisions have not
            # been detected yet so all tracks are treated equally.
            cooldown_until={},
            max_rescue_gap=identity_rescue_gap,
            rescue_confidence_threshold=rescue_confidence_threshold,
            max_centroid_dist_px=max_centroid_dist_px,
            audit_log=pre_audit,
        )
        _log_rescue_summary(pre_audit, label="pre-division identity rescue")

    # -----------------------------------------------------------------------
    # Stage 2: CTC-style division normalisation (unchanged logic).
    # The cooldown map built here is passed to the post-division rescue pass.
    # -----------------------------------------------------------------------
    for frame_idx in range(1, frame_count):
        prev_frame = normalized_tensor[:, :, frame_idx - 1]
        curr_frame = normalized_tensor[:, :, frame_idx]
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

        mother_to_children = {}
        newborn_id_set = set(newborn_ids)
        valid_parent_ids = prev_ids - protected_track_ids
        for child_id in curr_ids:
            if child_id in prev_ids and _has_self_continuity(prev_frame, curr_frame, child_id):
                continue
            child_mask = curr_frame == child_id
            mother_id = _infer_parent_id(
                prev_frame=prev_frame,
                child_mask=child_mask,
                child_track_id=child_id,
                valid_track_ids=valid_parent_ids,
            )
            if mother_id == 0:
                continue
            mother_to_children.setdefault(mother_id, []).append(child_id)

        if not mother_to_children:
            continue

        curr_id_set = set(curr_ids)
        for mother_id in sorted(mother_to_children):
            child_ids = sorted(set(mother_to_children[mother_id]))
            daughter_ids = []

            if mother_id in curr_id_set:
                max_track_id += 1
                if max_track_id > max_uint16:
                    raise ValueError("Division normalization would exceed uint16 track ID capacity.")
                continuation_daughter_id = max_track_id

                # Mother must end at f-1; relabel continuation branch from frame f onward.
                for time_idx in range(frame_idx, frame_count):
                    frame_t = normalized_tensor[:, :, time_idx]
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
                    if max_track_id > max_uint16:
                        raise ValueError("Division normalization would exceed uint16 track ID capacity.")
                    daughter_id = max_track_id
                    _fork_existing_daughter_label(normalized_tensor, child_id, frame_idx, daughter_id)
                    curr_id_set.discard(child_id)
                    curr_id_set.add(daughter_id)
                parent_map[daughter_id] = mother_id
                daughter_ids.append(daughter_id)

            if division_cooldown_frames > 0:
                cooldown_end = frame_idx + division_cooldown_frames
                for daughter_id in daughter_ids:
                    cooldown_until[daughter_id] = cooldown_end

    # -----------------------------------------------------------------------
    # Stage 3 (optional): Second identity-continuity rescue pass AFTER
    # division normalisation.  Now the cooldown map is populated, so the
    # scorer can apply a soft boost to post-division daughters.  This catches
    # any remaining label swaps that the pre-division pass could not see
    # (because they were caused by the division renaming itself).
    # -----------------------------------------------------------------------
    if identity_rescue_gap > 0 and cooldown_until:
        print("[TRACKING] export: running identity-continuity rescue (post-division pass)", flush=True)
        post_audit: list = []
        _rescue_identity_continuity(
            normalized_tensor=normalized_tensor,
            cooldown_until=cooldown_until,
            max_rescue_gap=identity_rescue_gap,
            rescue_confidence_threshold=rescue_confidence_threshold,
            max_centroid_dist_px=max_centroid_dist_px,
            excluded_lost_ids=set(parent_map.values()),
            audit_log=post_audit,
        )
        _log_rescue_summary(post_audit, label="post-division identity rescue")

    return normalized_tensor, parent_map


def _reindex_tracks_compact(tracked_tensor: np.ndarray, parent_map: dict[int, int]):
    track_ids = np.unique(tracked_tensor)
    track_ids = track_ids[track_ids != 0].astype(int)
    id_map = {int(old_id): new_id for new_id, old_id in enumerate(track_ids.tolist(), start=1)}

    if len(id_map) > np.iinfo(np.uint16).max:
        raise ValueError("Track count exceeds uint16 capacity required for challenge mask export.")

    reindexed_tensor = np.zeros_like(tracked_tensor, dtype=np.uint16)
    for old_id, new_id in id_map.items():
        reindexed_tensor[tracked_tensor == old_id] = new_id

    remapped_parent_map = {}
    for child_old_id, parent_old_id in parent_map.items():
        child_new_id = id_map.get(int(child_old_id))
        parent_new_id = id_map.get(int(parent_old_id))
        if child_new_id is None or parent_new_id is None:
            continue
        if child_new_id == parent_new_id:
            continue
        remapped_parent_map[child_new_id] = parent_new_id

    return reindexed_tensor, id_map, remapped_parent_map


def _scan_track_frame_ranges(final_tracked_tensor: np.ndarray):
    frame_ranges = {}
    for frame_idx in range(int(final_tracked_tensor.shape[2])):
        frame_ids = _track_ids_in_frame(final_tracked_tensor[:, :, frame_idx])
        for track_id in frame_ids.tolist():
            if track_id not in frame_ranges:
                frame_ranges[track_id] = [frame_idx, frame_idx]
            else:
                frame_ranges[track_id][1] = frame_idx
    return {track_id: (bounds[0], bounds[1]) for track_id, bounds in frame_ranges.items()}


def _compact_labels_in_place(mask_stack: np.ndarray):
    labels = np.unique(mask_stack)
    labels = labels[labels != 0].astype(np.int64, copy=False)
    if labels.size == 0:
        return 0, 0

    max_label_before = int(labels[-1])
    compact_labels = np.arange(1, labels.size + 1, dtype=mask_stack.dtype)

    for frame_idx in range(int(mask_stack.shape[2])):
        frame = mask_stack[:, :, frame_idx]
        nonzero = frame != 0
        if not np.any(nonzero):
            continue
        frame_values = frame[nonzero].astype(np.int64, copy=False)
        frame[nonzero] = compact_labels[np.searchsorted(labels, frame_values)]

    return int(labels.size), max_label_before


def _drop_track_fragment(mask_stack: np.ndarray, tp_im: np.ndarray, tp1: np.ndarray, track_id: int, timepoints):
    for frame_idx in timepoints:
        mask_2_change = mask_stack[:, :, frame_idx]
        pixo = mask_2_change == track_id
        if np.any(pixo):
            mask_2_change[pixo] = 0
        tp_im[track_id - 1, frame_idx] = 0
        tp1[track_id - 1, frame_idx] = False


def _contiguous_true_runs(present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return half-open [start, end) runs for a 1-D boolean presence row."""
    if present.size == 0 or not np.any(present):
        empty = np.array([], dtype=np.intp)
        return empty, empty

    padded = np.empty(present.size + 2, dtype=np.int8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = present.astype(np.int8, copy=False)
    edges = np.flatnonzero(np.diff(padded))
    return edges[0::2].astype(np.intp, copy=False), edges[1::2].astype(np.intp, copy=False)


def _queue_frame_relabels(
    frame_relabels: list[list[tuple[int, int]] | None],
    start_frame: int,
    end_frame: int,
    old_label: int,
    new_label: int,
) -> None:
    for frame_idx in range(start_frame, end_frame):
        actions = frame_relabels[frame_idx]
        if actions is None:
            frame_relabels[frame_idx] = [(old_label, new_label)]
        else:
            actions.append((old_label, new_label))


def _apply_frame_relabels(
    mask_stack: np.ndarray,
    frame_relabels: list[list[tuple[int, int]] | None],
    lut_size: int,
) -> None:
    affected_frames = [frame_idx for frame_idx, actions in enumerate(frame_relabels) if actions]
    if not affected_frames:
        return

    total_actions = sum(len(frame_relabels[frame_idx]) for frame_idx in affected_frames)
    print(
        f"[TRACKING] post: applying {total_actions} fragment label maps "
        f"across {len(affected_frames)} affected frames",
        flush=True,
    )

    base_lut = np.arange(max(lut_size, 1), dtype=mask_stack.dtype)
    for done, frame_idx in enumerate(affected_frames, start=1):
        if done % 1000 == 0 or done == len(affected_frames):
            print(
                f"[TRACKING] post discontinuity remap frame {done}/{len(affected_frames)}",
                flush=True,
            )

        frame = mask_stack[:, :, frame_idx]
        frame_max = int(np.max(frame)) if frame.size > 0 else 0
        if frame_max >= base_lut.size:
            lut = np.arange(frame_max + 1, dtype=mask_stack.dtype)
            lut[:base_lut.size] = base_lut
        else:
            lut = base_lut.copy()

        actions = frame_relabels[frame_idx]
        if actions is None:
            continue

        for old_label, new_label in actions:
            if old_label >= lut.size:
                expanded_lut = np.arange(old_label + 1, dtype=mask_stack.dtype)
                expanded_lut[:lut.size] = lut
                lut = expanded_lut
            lut[old_label] = new_label

        mask_stack[:, :, frame_idx] = lut[frame]


def _split_discontinuous_tracks(
    mask_stack: np.ndarray,
    tp_im: np.ndarray,
    tp1: np.ndarray,
    time_series_threshold: int,
    max_track_id: int = _MAX_CTC_TRACK_ID,
):
    obj = np.where(np.any(tp_im != 0, axis=1))[0] + 1
    numb_m1 = int(mask_stack.shape[2])
    free_label_ids = np.where(~np.any(tp_im != 0, axis=1))[0] + 1
    free_label_ids = free_label_ids[free_label_ids <= max_track_id]
    free_label_cursor = 0
    next_appended_label = int(tp_im.shape[0]) + 1
    pruned_short_fragments = 0
    pruned_capacity_fragments = 0
    frame_relabels: list[list[tuple[int, int]] | None] = [None] * numb_m1
    assigned_fragments: list[tuple[int, int, int, int]] = []
    dropped_fragments: list[tuple[int, int, int]] = []
    max_assigned_label = int(tp_im.shape[0])

    for cel1 in obj:
        if cel1 <= 0 or cel1 > tp1.shape[0]:
            continue
        starts, ends = _contiguous_true_runs(tp1[cel1 - 1, :])
        if starts.size <= 1:
            continue

        for start_frame, end_frame in zip(starts[1:].tolist(), ends[1:].tolist()):
            fragment_len = end_frame - start_frame
            if fragment_len < time_series_threshold:
                # Drop tiny disconnected fragments early so they do not consume extra track IDs.
                dropped_fragments.append((int(cel1), start_frame, end_frame))
                _queue_frame_relabels(frame_relabels, start_frame, end_frame, int(cel1), 0)
                pruned_short_fragments += 1
                continue

            if free_label_cursor < free_label_ids.size:
                new_label = int(free_label_ids[free_label_cursor])
                free_label_cursor += 1
            elif next_appended_label <= max_track_id:
                new_label = next_appended_label
                next_appended_label += 1
            else:
                dropped_fragments.append((int(cel1), start_frame, end_frame))
                _queue_frame_relabels(frame_relabels, start_frame, end_frame, int(cel1), 0)
                pruned_capacity_fragments += 1
                continue

            max_assigned_label = max(max_assigned_label, new_label)
            assigned_fragments.append((int(cel1), new_label, start_frame, end_frame))
            _queue_frame_relabels(frame_relabels, start_frame, end_frame, int(cel1), new_label)

    if max_assigned_label > tp_im.shape[0]:
        add_rows = max_assigned_label - tp_im.shape[0]
        tp_im = np.vstack([tp_im, np.zeros((add_rows, numb_m1), dtype=tp_im.dtype)])
        tp1 = np.vstack([tp1, np.zeros((add_rows, numb_m1), dtype=tp1.dtype)])

    for cel1, new_label, start_frame, end_frame in assigned_fragments:
        frames = slice(start_frame, end_frame)
        tp_im[new_label - 1, frames] = tp_im[cel1 - 1, frames]
        tp_im[cel1 - 1, frames] = 0
        tp1[new_label - 1, frames] = True
        tp1[cel1 - 1, frames] = False

    for cel1, start_frame, end_frame in dropped_fragments:
        frames = slice(start_frame, end_frame)
        tp_im[cel1 - 1, frames] = 0
        tp1[cel1 - 1, frames] = False

    _apply_frame_relabels(mask_stack, frame_relabels, max(max_assigned_label, int(tp_im.shape[0])) + 1)

    return tp_im, tp1, pruned_short_fragments, pruned_capacity_fragments


def _build_res_track_from_parent_map(
    final_tracked_tensor: np.ndarray,
    parent_map: dict[int, int],
    frame_ranges: dict[int, tuple[int, int]] | None = None,
):
    if frame_ranges is None:
        frame_ranges = _scan_track_frame_ranges(final_tracked_tensor)
    rows = []

    for track_id in sorted(frame_ranges):
        begin_frame, end_frame = frame_ranges[track_id]
        parent_id = int(parent_map.get(track_id, 0))
        if begin_frame == 0:
            parent_id = 0

        rows.append((track_id, begin_frame, end_frame, parent_id))

    return rows


def _validate_res_track_rows(final_tracked_tensor: np.ndarray, rows, frame_ranges=None):
    frame_count = int(final_tracked_tensor.shape[2])
    if frame_ranges is None:
        frame_ranges = _scan_track_frame_ranges(final_tracked_tensor)
    valid_track_ids = set(frame_ranges)
    row_track_ids = {int(row[0]) for row in rows}

    if row_track_ids != valid_track_ids:
        raise ValueError("res_track rows and tracked mask IDs do not match.")

    for track_id, begin_frame, end_frame, parent_id in rows:
        if begin_frame < 0 or end_frame < 0 or begin_frame > end_frame:
            raise ValueError(f"Invalid frame range for track {track_id}: B={begin_frame}, E={end_frame}")
        if end_frame >= frame_count:
            raise ValueError(f"Track {track_id} has E={end_frame}, but max frame is {frame_count - 1}")

        observed_begin, observed_end = frame_ranges[track_id]
        if begin_frame != observed_begin or end_frame != observed_end:
            raise ValueError(
                f"Track {track_id} B/E mismatch: expected {observed_begin}/{observed_end}, "
                f"got {begin_frame}/{end_frame}"
            )
        if parent_id != 0 and (parent_id == track_id or parent_id not in valid_track_ids):
            raise ValueError(f"Track {track_id} has invalid parent ID {parent_id}.")


def _write_challenge_outputs(
    result_dir: Path,
    final_tracked_tensor: np.ndarray,
    output_digits: int,
    division_cooldown_frames: int,
    identity_rescue_gap: int = 1,
    rescue_confidence_threshold: float = 0.30,
    max_centroid_dist_px: float = 50.0,
    tiff_write_workers: int = 4,
):
    result_dir.mkdir(parents=True, exist_ok=True)

    print("[TRACKING] export: normalizing CTC divisions + identity rescue", flush=True)
    normalized_tensor, division_parent_map = _normalize_ctc_divisions(
        final_tracked_tensor,
        division_cooldown_frames=division_cooldown_frames,
        identity_rescue_gap=identity_rescue_gap,
        rescue_confidence_threshold=rescue_confidence_threshold,
        max_centroid_dist_px=max_centroid_dist_px,
    )

    print("[TRACKING] export: scanning track frame ranges", flush=True)
    frame_ranges = _scan_track_frame_ranges(normalized_tensor)
    print("[TRACKING] export: building res_track rows", flush=True)
    rows = _build_res_track_from_parent_map(normalized_tensor, division_parent_map, frame_ranges)
    print("[TRACKING] export: validating res_track rows", flush=True)
    _validate_res_track_rows(normalized_tensor, rows, frame_ranges)

    print("[TRACKING] export: writing res_track.txt", flush=True)
    with (result_dir / "res_track.txt").open("w", encoding="utf-8") as f:
        for track_id, begin_frame, end_frame, parent_id in rows:
            f.write(f"{track_id} {begin_frame} {end_frame} {parent_id}\n")

    for old_mask in result_dir.glob("mask*.tif"):
        old_mask.unlink()

    total_frames = int(normalized_tensor.shape[2])
    if total_frames > 10**output_digits:
        raise ValueError(f"Cannot write {total_frames} frames with {output_digits}-digit CTC indices.")

    print(f"[TRACKING] export: writing {total_frames} mask files", flush=True)
    # OPT-5: parallel TIFF write – several OS write calls in flight at once
    _write_tiff_parallel(result_dir, normalized_tensor, output_digits,
                         n_workers=tiff_write_workers)

    final_object_count = max(frame_ranges) if frame_ranges else 0
    return len(rows), final_object_count


def _resolve_mask_files(mask_dir: Path, mask_pattern: str | None):
    if mask_pattern:
        if any(ch in mask_pattern for ch in ["*", "?", "["]):
            files = sorted(mask_dir.glob(mask_pattern), key=lambda p: _natural_sort_key(p.name))
        else:
            files = sorted(
                [p for p in mask_dir.iterdir() if p.is_file() and p.name.endswith(mask_pattern)],
                key=lambda p: _natural_sort_key(p.name),
            )
        if not files:
            raise FileNotFoundError(f"No masks found in {mask_dir} using pattern '{mask_pattern}'")
        matched_pattern = mask_pattern
    else:
        files = []
        matched_pattern = None
        for suffix in ["_cp_masks.tif", "_omni5_masks.tif", "_ART_masks.tif", "_masks.tif"]:
            files = sorted(
                [p for p in mask_dir.iterdir() if p.is_file() and p.name.endswith(suffix)],
                key=lambda p: _natural_sort_key(p.name),
            )
            if files:
                matched_pattern = suffix
                break
        if not files:
            raise FileNotFoundError(
                "No mask files found. Searched for *_cp_masks.tif, *_omni5_masks.tif, *_ART_masks.tif, *_masks.tif"
            )

    return files, matched_pattern


def _read_mask_frame(path: Path, resiz_factor: float):
    mask = tifffile.imread(str(path)).astype(np.uint16)
    if resiz_factor != 1.0:
        mask = resize(
            mask,
            (int(round(mask.shape[0] * resiz_factor)), int(round(mask.shape[1] * resiz_factor))),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint16)
    return mask


def run_tracking(
    mask_dir: Path,
    output_dir: Path,
    pos: str,
    mask_pattern: str | None,
    time_series_threshold: int,
    down_factor: int,
    resiz_factor: float,
    strict_matlab_id_matching: bool,
    output_digits: str,
    division_cooldown_frames: int,
    identity_rescue_gap: int = 1,
    rescue_confidence_threshold: float = 0.30,
    max_centroid_dist_px: float = 50.0,
    io_workers: int = 1,
    io_queue_depth: int = 4,
    tiff_write_workers: int = 4,
    mmap_dir: Path | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if down_factor != 1:
        raise ValueError("Challenge export requires down_factor=1 to preserve exact frame indexing.")
    if not np.isclose(resiz_factor, 1.0):
        raise ValueError("Challenge export requires resiz_factor=1.0 to preserve original image geometry.")
    if division_cooldown_frames < 0:
        raise ValueError("division_cooldown_frames must be >= 0.")

    result_dir = output_dir / f"{pos}_RES"
    result_dir.mkdir(parents=True, exist_ok=True)

    files, matched_pattern = _resolve_mask_files(mask_dir, mask_pattern)
    im_no = len(files)
    if im_no == 0:
        raise RuntimeError("No mask files loaded.")
    resolved_output_digits = _resolve_output_digits(output_digits, files, im_no)

    manifest_path = output_dir / f"{pos}_tracking_input_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"mask_dir={mask_dir}\n")
        f.write(f"output_dir={output_dir}\n")
        f.write(f"result_dir={result_dir}\n")
        f.write(f"matched_pattern={matched_pattern}\n")
        f.write(f"mask_count={len(files)}\n")
        f.write(f"output_digits={resolved_output_digits}\n")
        f.write(f"division_cooldown_frames={division_cooldown_frames}\n")
        f.write(f"identity_rescue_gap={identity_rescue_gap}\n")
        f.write(f"rescue_confidence_threshold={rescue_confidence_threshold}\n")
        f.write(f"max_centroid_dist_px={max_centroid_dist_px}\n")
        f.write(f"io_workers={io_workers}\n")
        f.write(f"io_queue_depth={io_queue_depth}\n")
        f.write(f"tiff_write_workers={tiff_write_workers}\n")
        f.write(f"mmap_dir={mmap_dir if mmap_dir is not None else 'auto'}\n")
        f.write("mask_files:\n")
        for p in files:
            f.write(f"  {p.name}\n")

    print(f"[TRACKING] START mask_dir={mask_dir} output_dir={output_dir} result_dir={result_dir}")
    print(
        f"[TRACKING] loaded_masks={im_no} matched_pattern={matched_pattern} "
        f"output_digits={resolved_output_digits}"
    )

    if down_factor <= 0:
        raise ValueError("down_factor must be >= 1")

    first_mask = _read_mask_frame(files[0], resiz_factor)
    frame_shape = first_mask.shape

    bytes_per_frame = frame_shape[0] * frame_shape[1] * np.dtype(_TRACKING_DTYPE).itemsize
    estimated_stack_bytes = bytes_per_frame * im_no
    mmap_root = _choose_mmap_dir(output_dir, mmap_dir, estimated_stack_bytes)
    tracked_stack_path = mmap_root / f"{pos}_tracked_stack_{time.time_ns()}.{np.dtype(_TRACKING_DTYPE).name}.mmap"
    print(
        f"[TRACKING] low-memory mode: disk-backed stack={tracked_stack_path} "
        f"dtype={np.dtype(_TRACKING_DTYPE).name} shape={frame_shape} "
        f"frames={im_no} estimated_size={_format_gib(estimated_stack_bytes)}"
    )

    tracked_stack = None
    mask0 = None
    final_tracked_tensor = None
    mask_2_change = None
    try:
        tracked_stack = np.memmap(
            tracked_stack_path,
            mode="w+",
            dtype=_TRACKING_DTYPE,
            shape=(frame_shape[0], frame_shape[1], im_no),
        )

        is1 = np.copy(first_mask).astype(_TRACKING_DTYPE)
        iblank_g = np.zeros(is1.shape, dtype=_TRACKING_DTYPE)
        max_id_ever = int(np.max(is1)) if np.max(is1) > 0 else 0

        tic = time.time()

        # OPT-1: Prefetch mask TIFFs from disk on a background thread so I/O
        # and computation are overlapped.  files[0] already decoded into
        # first_mask, so pass the full list and skip the first yielded frame.
        prefetcher = _PrefetchQueue(
            files,
            resiz_factor=resiz_factor,
            queue_depth=io_queue_depth,
            io_workers=io_workers,
        )
        frame_iter = iter(prefetcher)

        for it0 in range(im_no):
            if it0 % 25 == 0 or it0 == im_no - 1:
                print(f"[TRACKING] frame {it0 + 1}/{im_no}")

            # Consume next decoded frame from the prefetch queue.
            raw_frame = next(frame_iter)
            if it0 == 0:
                is2 = np.copy(first_mask).astype(np.uint16)
            else:
                is2 = raw_frame.astype(np.uint16)
                if is2.shape != frame_shape:
                    prefetcher.stop()
                    raise ValueError(
                        f"Mask shape mismatch for {files[it0].name}: got {is2.shape}, expected {frame_shape}."
                    )
            is2c = np.copy(is2)
            is1b = binar(is1)
            is3 = is1b.astype(np.uint16) * is2

            tr_cells = np.unique(is1[is1 != 0])
            gap_cells = np.unique(iblank_g[iblank_g != 0])
            cells_tr = np.concatenate((tr_cells, gap_cells))
            iblank0 = np.zeros_like(is1)

            if cells_tr.size != 0:
                for it1 in np.sort(cells_tr):
                    is5 = (is1 == it1).copy()
                    is_gap_cell = False
                    if not np.any(is5):
                        is5 = (iblank_g == it1).copy()
                        is_gap_cell = True

                    if is_gap_cell:
                        is6a = np.uint16(is5) * is2
                        if strict_matlab_id_matching:
                            iblank_g[iblank_g == it1] = 0
                    else:
                        is6a = np.uint16(is5) * is3

                    if np.any(is6a):
                        if strict_matlab_id_matching:
                            nz = is6a[is6a != 0]
                            if nz.size == 0:
                                continue
                            # OPT-2: bincount mode is ~3-5× faster than scipy.stats.mode
                            is2ind = _fast_mode_uint(nz.astype(np.intp))

                            iblank0[is2 == is2ind] = it1
                            is3[is3 == is2ind] = 0
                            is2c[is2 == is2ind] = 0
                            continue

                        valid_match = True
                        if is_gap_cell:
                            overlap_area = np.sum(is6a > 0)
                            cell_area = np.sum(is5)
                            ratio = overlap_area / cell_area if cell_area > 0 else 0
                            if ratio < 0.25:
                                valid_match = False

                        if valid_match:
                            # OPT-7: single bincount replaces len(uniq) full-frame scans
                            uniq, fracs = _overlap_scores(is6a)
                            if uniq.size == 0:
                                continue

                            pixi = uniq[fracs >= 0.35].astype(int)
                            if pixi.size == 0:
                                pixi = uniq[fracs >= 0.10].astype(int)

                            pix2 = np.array([], dtype=int)
                            for iu2 in pixi:
                                ixx = imopen_binary(is2c == iu2, np.ones((3, 3)))
                                pix2 = np.append(pix2, np.where(ixx.flatten())[0])

                            iblank0.flat[pix2] = it1

                            if is_gap_cell:
                                iblank_g[iblank_g == it1] = 0

                            i2 = iblank0 == it1
                            ratio_a = np.sum(is5) / np.sum(i2) if np.sum(i2) > 0 else 0
                        else:
                            continue

                        if ratio_a <= 0.80 and np.sum(erosion(is5, np.ones((9, 9)))) > 0:
                            b = erosion(is5, np.ones((9, 9)))
                            ba = skeletonize(b)
                            b_endpoints = find_endpoints(ba)
                            b1 = binary_dilation(b_endpoints, np.ones((27, 27)))
                            b2 = b1 * ba
                            b3 = label(b2)

                            iab = np.zeros_like(is5, dtype=float)
                            for it2 in range(1, int(b3.max()) + 1):
                                b4 = b3 == it2
                                props = regionprops(b4.astype(int))
                                if len(props) > 0:
                                    orient = props[0].orientation
                                    se = np.zeros((81, 81))
                                    se[40, :] = 1
                                    se = rotate(se, np.degrees(orient), reshape=False, order=0)
                                    ab = binary_dilation(b4, se > 0.5)
                                    iab = iab + ab.astype(float)

                            ia = i2.astype(int) - is5.astype(int)
                            ib = imopen_binary(ia == 1, np.ones((3, 3)))
                            ic = label(ib)
                            iab2 = iab * ic
                            pix1 = np.unique(iab2[iab2 != 0]).astype(int)

                            if pix1.size > 0:
                                for it3 in pix1:
                                    if np.sum(ic == it3) <= 1000:
                                        ic[ic == it3] = 0

                                iblank0[ic != 0] = 0
                                id_mask = (ia == 1) * (ic == 0)
                                id_labeled = label(id_mask)
                                id2 = np.zeros_like(id_mask, dtype=bool)
                                for lbl in np.unique(id_labeled[id_labeled != 0]):
                                    size = np.sum(id_labeled == lbl)
                                    if 80 <= size <= 10000:
                                        id2[id_labeled == lbl] = True

                                ig = is5 + id2
                                pix2 = np.where(ig.flatten() != 0)[0]
                            else:
                                er_counts = np.array([np.sum(ic == i) for i in range(1, int(ic.max()) + 1)])
                                er_pix = np.where(er_counts >= 1000)[0] + 1

                                if er_pix.size > 0:
                                    for it3 in er_pix:
                                        iblank0[ic == it3] = 0
                                        ic[ic == it3] = 0

                                    id_mask = (ia == 1) * (ic == 0)
                                    id0 = imopen_binary(id_mask, np.ones((3, 3)))
                                    id2 = id_mask * (id0 == 0)
                                    ig = is5 + id2 + ic

                                    ig_labeled = label(ig.astype(bool))
                                    ig_filtered = np.zeros_like(ig, dtype=bool)
                                    for lbl in np.unique(ig_labeled[ig_labeled != 0]):
                                        size = np.sum(ig_labeled == lbl)
                                        if 20 <= size <= 10000000:
                                            ig_filtered[ig_labeled == lbl] = True
                                    pix2 = np.where(ig_filtered.flatten())[0]
                                else:
                                    iblank0.flat[pix2] = it1
                                    is3.flat[pix2] = 0
                                    is2c.flat[pix2] = 0

                            iblank0.flat[pix2] = it1
                            is3.flat[pix2] = 0
                            is2c.flat[pix2] = 0

                        elif ratio_a >= 1.1:
                            is6a2 = is6a.astype(bool)
                            pix3 = np.sum((is6a2 + is5) == 2)
                            pix4 = np.sum(is5 != 0)

                            if pix4 > 0 and (pix3 / pix4) > 0.9:
                                is6b = skeletonize(erosion(is5, np.ones((3, 3)))).astype(np.uint16) * is3
                                if not np.any(is6b):
                                    is6b = erosion(is5, np.ones((3, 3))).astype(np.uint16) * is3

                                pix5 = np.unique(is6b[is6b != 0])
                                pix51 = np.sum(is5 != 0)
                                pix6 = np.array([0])

                                for iu3 in pix5:
                                    pix0 = np.where(imopen_binary(is2c == iu3, np.ones((3, 3))).flatten())[0]
                                    pix52 = np.where((is6a == iu3).flatten())[0]
                                    pix52b = len(pix52)

                                    if pix51 > 0 and pix52b > 0:
                                        if (len(pix0) / pix51 < 0.9) and (len(pix0) / pix52b < 1.5):
                                            pix6 = np.append(pix6, pix0)
                                        elif len(pix0) / pix52b > 1.5:
                                            pix6 = np.append(pix6, pix52)

                                if np.any(pix6 != 0):
                                    pix6 = pix6[pix6 != 0]
                                    ib = np.zeros_like(is5)
                                    ib.flat[pix6.astype(int)] = 1
                                    ib1 = skeletonize(ib.astype(bool))
                                    ib_branchpoints = find_branchpoints(ib1)
                                    ib2 = binary_dilation(ib_branchpoints, np.ones((3, 3)))
                                    ib3 = np.unique((is6a * ib2.astype(np.uint16))[is6a * ib2.astype(np.uint16) != 0])

                                    ic1 = skeletonize(is5.astype(bool))
                                    ic_branchpoints = find_branchpoints(ic1)
                                    ic2 = binary_dilation(ic_branchpoints, np.ones((3, 3)))
                                    ic3 = np.unique((is6a * ic2.astype(np.uint16))[is6a * ic2.astype(np.uint16) != 0])

                                    pixi = np.setdiff1d(ib3, ic3)
                                    if pixi.size > 0:
                                        for ij in pixi:
                                            erasepix = np.where((is6a == ij).flatten())[0]
                                            ib.flat[erasepix] = 0

                                    ib = imopen_binary(ib, np.ones((3, 3)))
                                    pix7 = np.where(ib.flatten() != 0)[0]
                                else:
                                    pix7 = np.where(is5.flatten() != 0)[0]
                                    ibb = np.zeros_like(is5)
                                    ibb.flat[pix7] = 1
                                    ix2 = ibb * iblank0
                                    ix2 = imopen_binary(ix2, np.ones((3, 3)))
                                    pixt = np.where((ix2 != it1).flatten())[0]
                                    ibb.flat[pixt] = 0
                                    pix7 = np.where(ibb.flatten() != 0)[0]

                                iblank0.flat[pix7] = it1
                                is3.flat[pix7] = 0
                                is2c.flat[pix7] = 0
                        else:
                            is3.flat[pix2] = 0
                            is2c.flat[pix2] = 0

                seg_gap = np.setdiff1d(tr_cells, np.unique(iblank0))
                if seg_gap.size > 0:
                    for itg in seg_gap:
                        iblank_g[is1 == itg] = itg

                iblank0b = iblank0.copy()
                iblank0b[iblank0 != 0] = 1
                isb = is2 * np.uint16(~iblank0b.astype(bool))
                isb = imopen_labels(isb, np.ones((3, 3)))
                newcells = np.unique(isb[isb != 0])
                iblank = iblank0.copy()

                if newcells.size > 0:
                    for a, it2 in enumerate(newcells, start=1):
                        assigned_id = max_id_ever + a
                        iblank[isb == it2] = assigned_id
                    max_id_ever += len(newcells)

                tracked_stack[:, :, it0] = iblank.astype(_TRACKING_DTYPE, copy=False)
                is1 = tracked_stack[:, :, it0].copy()
            else:
                tracked_stack[:, :, it0] = is2.copy()
                is1 = tracked_stack[:, :, it0].copy()
                if it0 == 0 or max_id_ever == 0:
                    max_id_ever = int(np.max(is2)) if np.max(is2) > 0 else 0

        elapsed_time = time.time() - tic
        print(f"[TRACKING] main_loop_elapsed={elapsed_time:.2f}s")
        prefetcher.stop()
        tracked_stack.flush()

        print("[TRACKING] post: compacting raw tracker labels", flush=True)
        mask0 = tracked_stack
        # OPT-3: LUT-based compaction – one global pass instead of per-frame label loops
        raw_track_count, raw_max_label = _compact_labels_in_place_fast(mask0)
        if raw_track_count > 0:
            print(
                f"[TRACKING] post: raw tracker labels used={raw_track_count} "
                f"max_label_before_compaction={raw_max_label}",
                flush=True,
            )

        print("[TRACKING] post: preparing downsampled mask stack", flush=True)
        dz = int(mask0.shape[2])
        max_id = raw_track_count
        numb_m1 = int(mask0.shape[2])
        if max_id > 0:
            print(f"[TRACKING] post: building temporal presence table for {numb_m1} frames", flush=True)
            # OPT-4: consolidated tp_im builder
            tp_im = _build_tp_im(mask0, max_id)
        else:
            tp_im = np.zeros((0, numb_m1), dtype=np.uint32)

        tp1 = tp_im != 0

        object_count = int(np.count_nonzero(np.any(tp_im != 0, axis=1)))
        print(f"[TRACKING] post: resolving discontinuous tracks across {object_count} objects", flush=True)
        tp_im, tp1, pruned_short_fragments, pruned_capacity_fragments = _split_discontinuous_tracks(
            mask_stack=mask0,
            tp_im=tp_im,
            tp1=tp1,
            time_series_threshold=time_series_threshold,
        )
        if pruned_short_fragments > 0:
            print(
                f"[TRACKING] post: pruned {pruned_short_fragments} disconnected fragments "
                f"shorter than time-series-threshold={time_series_threshold}",
                flush=True,
            )
        if pruned_capacity_fragments > 0:
            print(
                f"[TRACKING] post: pruned {pruned_capacity_fragments} disconnected fragments "
                f"after reaching the uint16 CTC label limit ({_MAX_CTC_TRACK_ID}); "
                "increase --time-series-threshold to prefer longer surviving tracks",
                flush=True,
            )

        obj_cor = (np.where(np.sum(tp1, axis=1) >= time_series_threshold)[0] + 1).astype(int)
        no_obj = int(len(obj_cor))
        if no_obj > np.iinfo(np.uint16).max:
            raise ValueError("Track count exceeds uint16 capacity required for challenge mask export.")

        print(f"[TRACKING] post: compacting surviving tracks to {no_obj} objects", flush=True)
        track_id_map = np.zeros(tp_im.shape[0] + 1, dtype=np.uint16)
        for seq_id, raw_id in enumerate(obj_cor, start=1):
            track_id_map[raw_id] = seq_id

        # OPT-8: single LUT remap replaces F × len(obj_cor) individual scans
        if numb_m1 % 250 == 0 or numb_m1 > 0:
            print(f"[TRACKING] post remap {numb_m1} frames via LUT", flush=True)
        _remap_stack_with_lut(mask0, track_id_map)

        final_number_frames = int(mask0.shape[2])
        final_tracked_tensor = mask0
        del tp_im, tp1, track_id_map

        track_count, final_number_objects = _write_challenge_outputs(
            result_dir=result_dir,
            final_tracked_tensor=final_tracked_tensor,
            output_digits=resolved_output_digits,
            division_cooldown_frames=division_cooldown_frames,
            identity_rescue_gap=identity_rescue_gap,
            rescue_confidence_threshold=rescue_confidence_threshold,
            max_centroid_dist_px=max_centroid_dist_px,
            tiff_write_workers=tiff_write_workers,
        )

        print(
            f"[TRACKING] END final_objects={final_number_objects} "
            f"frames={final_number_frames} down_factor={down_factor} dz={dz} "
            f"res_track_rows={track_count} result_dir={result_dir}"
        )
    finally:
        if tracked_stack is not None:
            tracked_stack.flush()

        # Ensure all memmap references are dropped before attempting to remove the backing file.
        mask_2_change = None
        final_tracked_tensor = None
        mask0 = None
        tracked_stack = None
        gc.collect()

        if tracked_stack_path.exists():
            removed = False
            for retry_idx in range(5):
                try:
                    tracked_stack_path.unlink()
                    removed = True
                    break
                except PermissionError:
                    time.sleep(0.2 * (retry_idx + 1))
            if not removed and tracked_stack_path.exists():
                print(
                    f"[TRACKING] warning: could not remove temporary memmap file: {tracked_stack_path}",
                    flush=True,
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone tip-tracking runner with CTC/challenge-format export."
    )
    parser.add_argument(
        "--mask-dir",
        required=False,
        type=Path,
        default=None,
        help="Folder with mask TIFF files (default: <script_dir>/input)",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        type=Path,
        default=None,
        help="Root folder for outputs (default: <script_dir>/output_<timestamp>); writes <sequence>_RES inside it",
    )
    parser.add_argument("--position", type=str, default=None, help="Output prefix (default: mask-dir basename)")
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default=None,
        help="Optional mask filename glob/suffix (e.g. '*_cp_masks.tif' or '_cp_masks.tif')",
    )
    parser.add_argument("--time-series-threshold", type=int, default=1)
    parser.add_argument("--down-factor", type=int, default=1)
    parser.add_argument("--resiz-factor", type=float, default=1.0)
    parser.add_argument(
        "--output-digits",
        default="auto",
        help="Digits used for CTC maskT.tif output names: auto or a positive integer (default: auto).",
    )
    parser.add_argument(
        "--division-cooldown-frames",
        default=20,
        type=int,
        help="Frames after a detected division where daughter IDs are protected from immediate relabeling (default: 20; 0 disables).",
    )
    parser.add_argument(
        "--identity-rescue-gap",
        default=1,
        type=int,
        help=(
            "Maximum frame gap over which the identity-continuity rescue will search for a "
            "replacement label for a disappearing track (default: 1; 0 disables rescue)."
        ),
    )
    parser.add_argument(
        "--rescue-confidence-threshold",
        default=0.30,
        type=float,
        help=(
            "Minimum confidence score [0–1] required before a new label is relabelled back to a "
            "lost track ID (default: 0.30). Raise to reduce false rescues; lower to catch more swaps."
        ),
    )
    parser.add_argument(
        "--max-centroid-dist-px",
        default=50.0,
        type=float,
        help=(
            "Hard spatial limit in pixels on centroid-to-centroid distance when scoring rescue "
            "candidates (default: 50.0). Increase for fast-moving or large cells."
        ),
    )
    parser.add_argument("--strict-matlab-id-matching", dest="strict_matlab_id_matching", action="store_true")
    parser.add_argument("--no-strict-matlab-id-matching", dest="strict_matlab_id_matching", action="store_false")
    parser.set_defaults(strict_matlab_id_matching=True)

    # ------------------------------------------------------------------
    # Performance-tuning options
    # ------------------------------------------------------------------
    parser.add_argument(
        "--io-workers",
        default=1,
        type=int,
        help=(
            "Number of background threads for prefetching mask TIFFs from disk "
            "(default: 1). Increase to 2-4 on NVMe or network-attached storage "
            "where parallel reads are faster. 1 = serial prefetch on one thread."
        ),
    )
    parser.add_argument(
        "--io-queue-depth",
        default=4,
        type=int,
        help=(
            "Number of decoded mask frames to keep in the prefetch queue "
            "(default: 4). Each slot costs roughly frame_h × frame_w × 2 bytes. "
            "Raise for faster throughput; lower on memory-constrained machines."
        ),
    )
    parser.add_argument(
        "--tiff-write-workers",
        default=4,
        type=int,
        help=(
            "Number of threads for parallel TIFF output writing (default: 4). "
            "Increase on fast storage; 1 disables parallelism."
        ),
    )
    parser.add_argument(
        "--mmap-dir",
        default=None,
        type=Path,
        help=(
            "Directory for the temporary disk-backed tracking stack. By default, UNC/network "
            "output paths use a local temp scratch folder and local output paths use output-dir. "
            "Set this to a fast local SSD folder when processing masks from a network share."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    mask_dir = (args.mask_dir if args.mask_dir is not None else script_dir / "input").resolve()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else script_dir / f"output_{time.strftime('%Y%m%d_%H%M%S')}"
    ).resolve()
    mmap_dir = args.mmap_dir.expanduser().resolve() if args.mmap_dir is not None else None

    if not mask_dir.exists() or not mask_dir.is_dir():
        raise NotADirectoryError(
            f"mask-dir does not exist or is not a directory: {mask_dir}. "
            "Use --mask-dir to override the default."
        )

    pos = args.position or mask_dir.name

    run_tracking(
        mask_dir=mask_dir,
        output_dir=output_dir,
        pos=pos,
        mask_pattern=args.mask_pattern,
        time_series_threshold=args.time_series_threshold,
        down_factor=args.down_factor,
        resiz_factor=args.resiz_factor,
        strict_matlab_id_matching=args.strict_matlab_id_matching,
        output_digits=args.output_digits,
        division_cooldown_frames=args.division_cooldown_frames,
        identity_rescue_gap=args.identity_rescue_gap,
        rescue_confidence_threshold=args.rescue_confidence_threshold,
        max_centroid_dist_px=args.max_centroid_dist_px,
        io_workers=args.io_workers,
        io_queue_depth=args.io_queue_depth,
        tiff_write_workers=args.tiff_write_workers,
        mmap_dir=mmap_dir,
    )


if __name__ == "__main__":
    main()
