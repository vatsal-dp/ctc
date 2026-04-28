#!/usr/bin/env python3

import argparse
import gc
import re
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import convolve, rotate
from scipy.ndimage import label as bwlabel
from scipy.stats import mode
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, erosion, opening, skeletonize
from skimage.transform import resize


_MAX_CTC_TRACK_ID = int(np.iinfo(np.uint16).max)
_TRACKING_DTYPE = np.uint32


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
    mask_bin = mask.copy()
    mask_bin[mask != 0] = 1
    return mask_bin


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


def _track_ids_in_frame(frame: np.ndarray):
    ids = np.unique(frame)
    return ids[ids != 0].astype(int)


def _active_cooldown_track_ids(cooldown_until: dict[int, int], frame_idx: int):
    return {track_id for track_id, end_frame in cooldown_until.items() if frame_idx <= end_frame}


def _rescue_cooldown_label_swaps(
    normalized_tensor: np.ndarray,
    frame_idx: int,
    newborn_ids: list[int],
    protected_track_ids: set[int],
):
    prev_frame = normalized_tensor[:, :, frame_idx - 1]
    curr_frame = normalized_tensor[:, :, frame_idx]
    prev_ids = set(_track_ids_in_frame(prev_frame).tolist())
    curr_ids = set(_track_ids_in_frame(curr_frame).tolist())
    rescued_ids = set()

    for protected_id in sorted(protected_track_ids):
        if protected_id not in prev_ids:
            continue
        if protected_id in curr_ids:
            continue

        previous_mask = prev_frame == protected_id
        if not np.any(previous_mask):
            continue

        dilated_previous = binary_dilation(previous_mask, np.ones((3, 3), dtype=bool))
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
        for time_idx in range(frame_idx, int(normalized_tensor.shape[2])):
            frame_t = normalized_tensor[:, :, time_idx]
            rescue_pixels = frame_t == rescued_id
            if np.any(rescue_pixels):
                frame_t[rescue_pixels] = protected_id

        rescued_ids.add(rescued_id)
        curr_ids.discard(rescued_id)
        curr_ids.add(protected_id)

    return sorted(track_id for track_id in newborn_ids if track_id not in rescued_ids)


def _normalize_ctc_divisions(final_tracked_tensor: np.ndarray, division_cooldown_frames: int = 20):
    if division_cooldown_frames < 0:
        raise ValueError("division_cooldown_frames must be >= 0.")

    max_uint16 = int(np.iinfo(np.uint16).max)
    max_input_id = int(np.max(final_tracked_tensor)) if final_tracked_tensor.size else 0
    if max_input_id > max_uint16:
        raise ValueError("Track IDs exceed uint16 capacity required for challenge mask export.")

    normalized_tensor = final_tracked_tensor
    frame_count = int(normalized_tensor.shape[2])
    max_track_id = int(np.max(normalized_tensor)) if normalized_tensor.size else 0
    parent_map = {}
    cooldown_until: dict[int, int] = {}

    # Enforce CTC-style division semantics frame by frame.
    for frame_idx in range(1, frame_count):
        prev_frame = normalized_tensor[:, :, frame_idx - 1]
        curr_frame = normalized_tensor[:, :, frame_idx]
        prev_ids = set(_track_ids_in_frame(prev_frame).tolist())
        if not prev_ids:
            continue

        curr_ids = _track_ids_in_frame(curr_frame).tolist()
        newborn_ids = sorted(track_id for track_id in curr_ids if track_id not in prev_ids)
        if not newborn_ids:
            continue

        protected_track_ids = (
            _active_cooldown_track_ids(cooldown_until, frame_idx)
            if division_cooldown_frames > 0
            else set()
        )
        if protected_track_ids:
            newborn_ids = _rescue_cooldown_label_swaps(
                normalized_tensor=normalized_tensor,
                frame_idx=frame_idx,
                newborn_ids=newborn_ids,
                protected_track_ids=protected_track_ids,
            )
            prev_frame = normalized_tensor[:, :, frame_idx - 1]
            curr_frame = normalized_tensor[:, :, frame_idx]
            prev_ids = set(_track_ids_in_frame(prev_frame).tolist())
            curr_ids = _track_ids_in_frame(curr_frame).tolist()
            newborn_ids = sorted(track_id for track_id in curr_ids if track_id not in prev_ids)
            if not newborn_ids:
                continue

        mother_to_newborns = {}
        valid_parent_ids = prev_ids - protected_track_ids
        for newborn_id in newborn_ids:
            child_mask = curr_frame == newborn_id
            mother_id = _infer_parent_id(
                prev_frame=prev_frame,
                child_mask=child_mask,
                child_track_id=newborn_id,
                valid_track_ids=valid_parent_ids,
            )
            if mother_id == 0:
                continue
            mother_to_newborns.setdefault(mother_id, []).append(newborn_id)

        if not mother_to_newborns:
            continue

        curr_id_set = set(curr_ids)
        for mother_id in sorted(mother_to_newborns):
            newborns = sorted(set(mother_to_newborns[mother_id]))
            daughter_ids = list(newborns)

            if mother_id in curr_id_set:
                max_track_id += 1
                if max_track_id > max_uint16:
                    raise ValueError("Division normalization would exceed uint16 track ID capacity.")
                continuation_daughter_id = max_track_id

                # Mother must end at f-1, so relabel continuation branch from frame f onward.
                for time_idx in range(frame_idx, frame_count):
                    frame_t = normalized_tensor[:, :, time_idx]
                    mother_pixels = frame_t == mother_id
                    if np.any(mother_pixels):
                        frame_t[mother_pixels] = continuation_daughter_id

                parent_map[continuation_daughter_id] = mother_id
                daughter_ids.append(continuation_daughter_id)
                curr_id_set.discard(mother_id)
                curr_id_set.add(continuation_daughter_id)
            elif len(newborns) < 2:
                continue

            for newborn_id in newborns:
                parent_map[newborn_id] = mother_id

            if division_cooldown_frames > 0:
                cooldown_end = frame_idx + division_cooldown_frames
                for daughter_id in daughter_ids:
                    cooldown_until[daughter_id] = cooldown_end

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
    pruned_short_fragments = 0
    pruned_capacity_fragments = 0

    for cel1 in obj:
        if cel1 <= 0 or cel1 > tp1.shape[0]:
            continue
        tpz, num_components = bwlabel(tp1[cel1 - 1, :].astype(np.uint8))
        if num_components <= 1:
            continue

        for it in range(2, num_components + 1):
            timepoints_to_change = np.where(tpz == it)[0]
            if timepoints_to_change.size < time_series_threshold:
                # Drop tiny disconnected fragments early so they do not consume extra track IDs.
                _drop_track_fragment(mask_stack, tp_im, tp1, int(cel1), timepoints_to_change)
                pruned_short_fragments += 1
                continue

            if free_label_cursor < free_label_ids.size:
                new_label = int(free_label_ids[free_label_cursor])
                free_label_cursor += 1
            elif tp_im.shape[0] < max_track_id:
                new_label = tp_im.shape[0] + 1
                tp_im = np.vstack([tp_im, np.zeros((1, numb_m1), dtype=tp_im.dtype)])
                tp1 = np.vstack([tp1, np.zeros((1, numb_m1), dtype=tp1.dtype)])
            else:
                _drop_track_fragment(mask_stack, tp_im, tp1, int(cel1), timepoints_to_change)
                pruned_capacity_fragments += 1
                continue

            for it2 in timepoints_to_change:
                mask_2_change = mask_stack[:, :, it2]
                pixo = mask_2_change == cel1
                if np.any(pixo):
                    mask_2_change[pixo] = new_label

                tp_im[new_label - 1, it2] = tp_im[cel1 - 1, it2]
                tp_im[cel1 - 1, it2] = 0
                tp1[new_label - 1, it2] = True
                tp1[cel1 - 1, it2] = False

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
):
    result_dir.mkdir(parents=True, exist_ok=True)

    print("[TRACKING] export: normalizing CTC divisions", flush=True)
    normalized_tensor, division_parent_map = _normalize_ctc_divisions(
        final_tracked_tensor,
        division_cooldown_frames=division_cooldown_frames,
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
    for frame_idx in range(normalized_tensor.shape[2]):
        if frame_idx % 250 == 0 or frame_idx == total_frames - 1:
            print(f"[TRACKING] export frame {frame_idx + 1}/{total_frames}", flush=True)
        tifffile.imwrite(
            str(result_dir / f"mask{frame_idx:0{output_digits}d}.tif"),
            normalized_tensor[:, :, frame_idx].astype(np.uint16, copy=False),
        )

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

    tracked_stack_path = output_dir / f"{pos}_tracked_stack_{time.time_ns()}.{np.dtype(_TRACKING_DTYPE).name}.mmap"

    bytes_per_frame = frame_shape[0] * frame_shape[1] * np.dtype(_TRACKING_DTYPE).itemsize
    estimated_stack_gib = (bytes_per_frame * im_no) / (1024**3)
    print(
        f"[TRACKING] low-memory mode: disk-backed stack={tracked_stack_path} "
        f"dtype={np.dtype(_TRACKING_DTYPE).name} shape={frame_shape} "
        f"frames={im_no} estimated_size={estimated_stack_gib:.2f} GiB"
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

        for it0 in range(im_no):
            if it0 % 25 == 0 or it0 == im_no - 1:
                print(f"[TRACKING] frame {it0 + 1}/{im_no}")

            if it0 == 0:
                is2 = np.copy(first_mask).astype(np.uint16)
            else:
                is2 = np.copy(_read_mask_frame(files[it0], resiz_factor)).astype(np.uint16)
                if is2.shape != frame_shape:
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
                            mode_res = mode(nz, keepdims=False)
                            try:
                                is2ind = int(mode_res.mode)
                            except Exception:
                                is2ind = int(np.ravel(mode_res.mode)[0])

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
                            uniq = np.unique(is6a[is6a != 0])
                            pix_si = np.zeros((len(uniq), 3))
                            for iu in range(len(uniq)):
                                pix_si[iu, 0] = uniq[iu]
                                pix_si[iu, 1] = np.sum(is6a == uniq[iu])
                            pix_si[:, 2] = pix_si[:, 1] / np.sum(pix_si[:, 1])

                            pixi = pix_si[pix_si[:, 2] >= 0.35, 0].astype(int)
                            if pixi.size == 0:
                                pixi = pix_si[pix_si[:, 2] >= 0.10, 0].astype(int)

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
        tracked_stack.flush()

        print("[TRACKING] post: compacting raw tracker labels", flush=True)
        mask0 = tracked_stack
        raw_track_count, raw_max_label = _compact_labels_in_place(mask0)
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
        tp_im = np.zeros((max_id, numb_m1), dtype=np.uint32)
        if max_id > 0:
            print(f"[TRACKING] post: building temporal presence table for {numb_m1} frames", flush=True)
            for ih in range(numb_m1):
                if ih % 250 == 0 or ih == numb_m1 - 1:
                    print(f"[TRACKING] post frame {ih + 1}/{numb_m1}", flush=True)
                counts = np.bincount(mask0[:, :, ih].ravel(), minlength=max_id + 1)
                tp_im[:, ih] = counts[1:].astype(np.uint32, copy=False)

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

        for frame_idx in range(numb_m1):
            if frame_idx % 250 == 0 or frame_idx == numb_m1 - 1:
                print(f"[TRACKING] post remap frame {frame_idx + 1}/{numb_m1}", flush=True)
            mask0[:, :, frame_idx] = track_id_map[mask0[:, :, frame_idx]]

        final_number_frames = int(mask0.shape[2])
        final_tracked_tensor = mask0
        del tp_im, tp1, track_id_map

        track_count, final_number_objects = _write_challenge_outputs(
            result_dir=result_dir,
            final_tracked_tensor=final_tracked_tensor,
            output_digits=resolved_output_digits,
            division_cooldown_frames=division_cooldown_frames,
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
    parser.add_argument("--strict-matlab-id-matching", dest="strict_matlab_id_matching", action="store_true")
    parser.add_argument("--no-strict-matlab-id-matching", dest="strict_matlab_id_matching", action="store_false")
    parser.set_defaults(strict_matlab_id_matching=True)
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
    )


if __name__ == "__main__":
    main()
