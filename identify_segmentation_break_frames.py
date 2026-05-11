#!/usr/bin/env python3
"""Find cell tracks whose object-size row drops to zero before the movie ends.

The script looks for an ``all_ob``-style matrix where rows are tracks/cells and
columns are time points. It writes two files:

* ``segmentation_break_events.csv``: one row per nonzero-to-zero transition.
* ``segmentation_break_review_frames.csv``: unique frames to inspect.
"""

import argparse
import csv
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PREFERRED_MATRIX_NAMES = (
    "all_ob",
    "all_obj_sizes",
    "all_obj_size",
    "all_object_sizes",
    "all_object_size",
    "object_sizes",
    "obj_sizes",
    "cell_sizes",
)


@dataclass(frozen=True)
class BreakEvent:
    track_row_0_based: int
    break_frame_0_based: int
    last_nonzero_frame_0_based: int
    zero_run_end_0_based: int
    next_nonzero_frame_0_based: int | None
    size_before_break: int | float
    event_type: str


@dataclass(frozen=True)
class LoadedMatrix:
    matrix: np.ndarray
    name: str
    source_path: Path


def _native_number(value: Any) -> int | float:
    scalar = np.asarray(value).item()
    if isinstance(scalar, (np.integer, int)):
        return int(scalar)
    if isinstance(scalar, (np.floating, float)):
        as_float = float(scalar)
        return int(as_float) if as_float.is_integer() else as_float
    return scalar


def _as_numeric_2d_matrix(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        return None
    return arr


def _iter_named_arrays(obj: Any, prefix: str = ""):
    matrix = _as_numeric_2d_matrix(obj)
    if matrix is not None and prefix:
        yield prefix, matrix

    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).startswith("__"):
                continue
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_named_arrays(value, name)
        return

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            name = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_named_arrays(value, name)
        return

    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size <= 100:
        for idx, value in np.ndenumerate(obj):
            name = f"{prefix}{idx}" if prefix else str(idx)
            yield from _iter_named_arrays(value, name)
        return

    if hasattr(obj, "_fieldnames"):
        for field in getattr(obj, "_fieldnames", []) or []:
            name = f"{prefix}.{field}" if prefix else field
            yield from _iter_named_arrays(getattr(obj, field), name)


def _candidate_score(name: str, variable: str | None) -> tuple[int, str]:
    lowered = name.lower()
    leaf = lowered.rsplit(".", 1)[-1].rsplit("/", 1)[-1]
    if variable:
        wanted = variable.lower()
        if leaf == wanted or lowered == wanted:
            return (0, name)
        if lowered.endswith("." + wanted) or lowered.endswith("/" + wanted):
            return (1, name)
        return (1000, name)

    for idx, preferred in enumerate(PREFERRED_MATRIX_NAMES):
        if leaf == preferred:
            return (idx, name)
    if "all_ob" in lowered:
        return (50, name)
    if ("obj" in lowered or "object" in lowered or "cell" in lowered) and "size" in lowered:
        return (75, name)
    return (1000, name)


def _choose_matrix(candidates: list[tuple[str, np.ndarray]], variable: str | None, source_path: Path) -> LoadedMatrix:
    usable = [
        (score, name, matrix)
        for name, matrix in candidates
        if (score := _candidate_score(name, variable))[0] < 1000
    ]
    if not usable:
        available = ", ".join(name for name, _ in candidates[:20]) or "none"
        wanted = variable or "one of " + ", ".join(PREFERRED_MATRIX_NAMES)
        raise ValueError(f"Could not find {wanted} in {source_path}. Available 2D numeric arrays: {available}")

    usable.sort(key=lambda item: item[0])
    _, name, matrix = usable[0]
    return LoadedMatrix(matrix=np.asarray(matrix), name=name, source_path=source_path)


def _file_is_all_zero(path: Path) -> bool:
    if path.stat().st_size == 0:
        return True
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return True
            if any(chunk):
                return False


def _load_mat(path: Path, variable: str | None) -> LoadedMatrix:
    if _file_is_all_zero(path):
        raise ValueError(f"{path} contains only zero bytes; it is not a valid MATLAB file.")
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise RuntimeError("Reading .mat files requires scipy. Install this repo's requirements first.") from exc

    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return _load_h5(path, variable)
    return _choose_matrix(list(_iter_named_arrays(data)), variable, path)


def _load_h5(path: Path, variable: str | None) -> LoadedMatrix:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Reading .h5 or MATLAB v7.3 files requires h5py. "
            "Install it on the lab PC with: python -m pip install h5py"
        ) from exc

    candidates: list[tuple[str, np.ndarray]] = []
    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: Any):
            if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
                return
            if name.startswith("#refs#"):
                return
            if len(obj.shape) != 2:
                return
            if not np.issubdtype(obj.dtype, np.number):
                return
            candidates.append((name, np.asarray(obj[()])))

        handle.visititems(visit)
    return _choose_matrix(candidates, variable, path)


def _load_pickle(path: Path, variable: str | None) -> LoadedMatrix:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return _choose_matrix(list(_iter_named_arrays(data)), variable, path)


def _candidate_files(folder: Path) -> list[Path]:
    patterns = (
        "*Tracks_MATLAB*.mat",
        "*Tracks*.h5",
        "*.mat",
        "*.h5",
        "*.pkl",
        "*.pickle",
    )
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        for path in sorted(folder.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    return files


def load_all_ob_matrix(source: Path | str, variable: str | None = None) -> LoadedMatrix:
    source_path = Path(source)
    if source_path.is_dir():
        failures: list[str] = []
        for candidate in _candidate_files(source_path):
            try:
                return load_all_ob_matrix(candidate, variable=variable)
            except Exception as exc:
                failures.append(f"{candidate.name}: {exc}")
        details = "\n  ".join(failures) if failures else "No .mat, .h5, .pkl, or .pickle files found."
        raise ValueError(f"Could not load an all_ob matrix from {source_path}.\n  {details}")

    suffix = source_path.suffix.lower()
    if suffix == ".mat":
        return _load_mat(source_path, variable)
    if suffix in {".h5", ".hdf5"}:
        return _load_h5(source_path, variable)
    if suffix in {".pkl", ".pickle"}:
        return _load_pickle(source_path, variable)
    raise ValueError(f"Unsupported input file type: {source_path}")


def find_break_events(
    matrix: np.ndarray,
    *,
    include_last_frame: bool = False,
    mode: str = "all",
    first_per_row: bool = False,
) -> list[BreakEvent]:
    all_ob = np.asarray(matrix)
    if all_ob.ndim != 2:
        raise ValueError(f"all_ob must be 2D with rows=tracks and columns=frames; got shape={all_ob.shape}")
    if mode not in {"all", "final-stops", "temporary-gaps"}:
        raise ValueError("mode must be one of: all, final-stops, temporary-gaps")

    frame_count = all_ob.shape[1]
    events: list[BreakEvent] = []
    for row_idx in range(all_ob.shape[0]):
        row = all_ob[row_idx]
        present = np.asarray(row > 0, dtype=bool)
        if not present.any() or frame_count < 2:
            continue

        zero_starts = np.flatnonzero(present[:-1] & ~present[1:]) + 1
        for zero_start in zero_starts:
            if zero_start == frame_count - 1 and not include_last_frame:
                continue

            zero_end = int(zero_start)
            while zero_end + 1 < frame_count and not present[zero_end + 1]:
                zero_end += 1

            next_nonzero = zero_end + 1 if zero_end + 1 < frame_count and present[zero_end + 1] else None
            event_type = "temporary_gap" if next_nonzero is not None else "final_stop"
            if mode == "final-stops" and event_type != "final_stop":
                continue
            if mode == "temporary-gaps" and event_type != "temporary_gap":
                continue

            events.append(
                BreakEvent(
                    track_row_0_based=int(row_idx),
                    break_frame_0_based=int(zero_start),
                    last_nonzero_frame_0_based=int(zero_start - 1),
                    zero_run_end_0_based=int(zero_end),
                    next_nonzero_frame_0_based=None if next_nonzero is None else int(next_nonzero),
                    size_before_break=_native_number(row[zero_start - 1]),
                    event_type=event_type,
                )
            )
            if first_per_row:
                break

    return events


def unique_review_frames(events: list[BreakEvent]) -> list[int]:
    return sorted({event.break_frame_0_based for event in events})


def _one_based(value: int | None) -> int | str:
    return "" if value is None else value + 1


def write_break_events_csv(path: Path, loaded: LoadedMatrix, events: list[BreakEvent]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_file",
                "matrix_name",
                "matrix_rows_tracks",
                "matrix_cols_frames",
                "track_row_0_based",
                "track_row_1_based",
                "break_frame_0_based",
                "break_frame_1_based",
                "last_nonzero_frame_0_based",
                "last_nonzero_frame_1_based",
                "zero_run_end_0_based",
                "zero_run_end_1_based",
                "zero_run_length",
                "reappears_after_break",
                "next_nonzero_frame_0_based",
                "next_nonzero_frame_1_based",
                "size_before_break",
                "event_type",
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "source_file": str(loaded.source_path),
                    "matrix_name": loaded.name,
                    "matrix_rows_tracks": loaded.matrix.shape[0],
                    "matrix_cols_frames": loaded.matrix.shape[1],
                    "track_row_0_based": event.track_row_0_based,
                    "track_row_1_based": event.track_row_0_based + 1,
                    "break_frame_0_based": event.break_frame_0_based,
                    "break_frame_1_based": event.break_frame_0_based + 1,
                    "last_nonzero_frame_0_based": event.last_nonzero_frame_0_based,
                    "last_nonzero_frame_1_based": event.last_nonzero_frame_0_based + 1,
                    "zero_run_end_0_based": event.zero_run_end_0_based,
                    "zero_run_end_1_based": event.zero_run_end_0_based + 1,
                    "zero_run_length": event.zero_run_end_0_based - event.break_frame_0_based + 1,
                    "reappears_after_break": event.next_nonzero_frame_0_based is not None,
                    "next_nonzero_frame_0_based": "" if event.next_nonzero_frame_0_based is None else event.next_nonzero_frame_0_based,
                    "next_nonzero_frame_1_based": _one_based(event.next_nonzero_frame_0_based),
                    "size_before_break": event.size_before_break,
                    "event_type": event.event_type,
                }
            )


def write_review_frames_csv(path: Path, events: list[BreakEvent]) -> None:
    by_frame: dict[int, list[BreakEvent]] = {}
    for event in events:
        by_frame.setdefault(event.break_frame_0_based, []).append(event)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_frame_0_based",
                "review_frame_1_based",
                "event_count",
                "final_stop_count",
                "temporary_gap_count",
                "track_rows_1_based",
            ],
        )
        writer.writeheader()
        for frame in sorted(by_frame):
            frame_events = by_frame[frame]
            writer.writerow(
                {
                    "review_frame_0_based": frame,
                    "review_frame_1_based": frame + 1,
                    "event_count": len(frame_events),
                    "final_stop_count": sum(event.event_type == "final_stop" for event in frame_events),
                    "temporary_gap_count": sum(event.event_type == "temporary_gap" for event in frame_events),
                    "track_rows_1_based": ";".join(str(event.track_row_0_based + 1) for event in frame_events),
                }
            )


def write_review_frames_txt(path: Path, events: list[BreakEvent]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 0-based frame numbers, matching mask000.tif naming\n")
        for frame in unique_review_frames(events):
            handle.write(f"{frame}\n")


def _default_output_dir(source: Path) -> Path:
    return (source if source.is_dir() else source.parent) / "segmentation_break_review"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an all_ob/object-size matrix and list frames where a track's "
            "cell size drops from nonzero to zero."
        )
    )
    parser.add_argument("source", type=Path, help="Folder or .mat/.h5/.pkl file containing all_ob.")
    parser.add_argument("--variable", default=None, help="Matrix variable/dataset name. Default: prefer all_ob.")
    parser.add_argument(
        "--mode",
        choices=["all", "final-stops", "temporary-gaps"],
        default="all",
        help="Which nonzero-to-zero events to report. Default: all.",
    )
    parser.add_argument(
        "--first-per-row",
        action="store_true",
        help="Only report the first qualifying break for each track row.",
    )
    parser.add_argument(
        "--include-last-frame",
        action="store_true",
        help="Also report drops to zero on the final frame. Default skips them.",
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose the loaded matrix if your file stores frames as rows and tracks as columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write CSV/TXT outputs. Default: SOURCE/segmentation_break_review.",
    )
    parser.add_argument("--prefix", default="segmentation_break", help="Output filename prefix.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        loaded = load_all_ob_matrix(args.source, variable=args.variable)
        if args.transpose:
            loaded = LoadedMatrix(matrix=loaded.matrix.T, name=loaded.name + ".T", source_path=loaded.source_path)

        events = find_break_events(
            loaded.matrix,
            include_last_frame=args.include_last_frame,
            mode=args.mode,
            first_per_row=args.first_per_row,
        )

        output_dir = args.output_dir or _default_output_dir(args.source)
        output_dir.mkdir(parents=True, exist_ok=True)
        events_csv = output_dir / f"{args.prefix}_events.csv"
        frames_csv = output_dir / f"{args.prefix}_review_frames.csv"
        frames_txt = output_dir / f"{args.prefix}_review_frames_0_based.txt"

        write_break_events_csv(events_csv, loaded, events)
        write_review_frames_csv(frames_csv, events)
        write_review_frames_txt(frames_txt, events)

        final_stops = sum(event.event_type == "final_stop" for event in events)
        temporary_gaps = sum(event.event_type == "temporary_gap" for event in events)
        print(f"Loaded matrix: {loaded.name} from {loaded.source_path}")
        print(f"Matrix shape: rows/tracks={loaded.matrix.shape[0]} cols/frames={loaded.matrix.shape[1]}")
        print(
            f"Found {len(events)} break events across {len(unique_review_frames(events))} unique frames "
            f"(final_stops={final_stops}, temporary_gaps={temporary_gaps})."
        )
        print(f"Wrote: {events_csv}")
        print(f"Wrote: {frames_csv}")
        print(f"Wrote: {frames_txt}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
