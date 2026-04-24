#!/usr/bin/env python3

import argparse
import csv
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

METRICS = ("TRA", "SEG", "DET")

TRA_DET_SECTIONS = {
    "TRA": (
        "Splitting Operations",
        "False Negative Vertices",
        "False Positive Vertices",
        "Redundant Edges To Be Deleted",
        "Edges To Be Added",
        "Edges with Wrong Semantics",
    ),
    "DET": (
        "Splitting Operations",
        "False Negative Vertices",
        "False Positive Vertices",
    ),
}

SECTION_SLUGS = {
    "Splitting Operations": "splitting_operations",
    "False Negative Vertices": "false_negative_vertices",
    "False Positive Vertices": "false_positive_vertices",
    "Redundant Edges To Be Deleted": "redundant_edges_to_be_deleted",
    "Edges To Be Added": "edges_to_be_added",
    "Edges with Wrong Semantics": "edges_with_wrong_semantics",
}

SECTION_RE = re.compile(r"^-+(?P<title>.*?)(?:\s*\(Penalty=(?P<penalty>[^)]+)\))?-+\s*$")
SCORE_RE = re.compile(
    r"\b(?P<metric>TRA|DET|SEG)\s+measure:\s*(?P<score>[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
SEG_FRAME_RE = re.compile(r"^-+T=(?P<t>\d+)(?:\s+Z=(?P<z>-?\d+))?-+\s*$")
SEG_OBJECT_RE = re.compile(
    r"^GT_label=(?P<gt_label>\d+)\s+J=(?P<jaccard>[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)$"
)


def _normalize_sequence(sequence: str) -> str:
    text = str(sequence)
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def _platform_folder() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "Mac"
    if system == "windows":
        return "Win"
    return "Linux"


def _candidate_executable_names(metric: str):
    metric = metric.upper()
    names = [
        f"{metric}Measure",
        f"{metric}Measure.exe",
        f"{metric}measure",
        f"{metric}measure.exe",
        metric,
        f"{metric}.exe",
    ]
    return names


def _is_probable_executable(path: Path):
    if not path.is_file():
        return False
    if platform.system().lower() == "windows":
        return path.suffix.lower() == ".exe" or "." not in path.name
    return os.access(path, os.X_OK) or "." not in path.name


def find_metric_executable(metric: str, software_dir: Path | None):
    metric = metric.upper()
    candidates = _candidate_executable_names(metric)

    if software_dir is not None:
        roots = [software_dir / _platform_folder(), software_dir]
        for root in roots:
            if not root.exists():
                continue
            for name in candidates:
                direct = root / name
                if _is_probable_executable(direct):
                    return direct
            for path in root.rglob("*"):
                name = path.name.lower()
                if metric.lower() in name and "measure" in name and _is_probable_executable(path):
                    return path

    for name in candidates:
        found = shutil.which(name)
        if found:
            return Path(found)

    search_hint = f" under {software_dir}" if software_dir is not None else " on PATH"
    raise FileNotFoundError(f"Could not find executable for metric {metric}{search_hint}.")


def parse_score(stdout: str):
    numbers = re.findall(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?", stdout)
    if not numbers:
        return None
    return float(numbers[-1])


def parse_official_score(text: str, metric: str | None = None):
    metric = metric.upper() if metric is not None else None
    matches = list(SCORE_RE.finditer(text))
    if metric is not None:
        matches = [match for match in matches if match.group("metric").upper() == metric]
    if not matches:
        return None
    return float(matches[-1].group("score"))


def _sequence_from_result_dir(result_dir: Path) -> str:
    return result_dir.name[: -len("_RES")] if result_dir.name.endswith("_RES") else result_dir.name


def _format_float(value):
    if value is None:
        return ""
    return f"{value:.12g}"


def _parse_penalty(value: str | None):
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_tra_det_log(log_path: Path, metric: str):
    metric = metric.upper()
    if metric not in TRA_DET_SECTIONS:
        raise ValueError(f"TRA/DET log parser does not support metric {metric!r}.")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    score = parse_official_score(text, metric)
    sequence = _sequence_from_result_dir(log_path.parent)
    expected_sections = TRA_DET_SECTIONS[metric]
    counts = {section: 0 for section in expected_sections}
    penalties = {section: None for section in expected_sections}

    current_section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("="):
            current_section = None
            continue

        header = SECTION_RE.match(line)
        if header:
            title = header.group("title").strip()
            if title in counts:
                current_section = title
                penalties[title] = _parse_penalty(header.group("penalty"))
            else:
                current_section = None
            continue

        if current_section is not None:
            counts[current_section] += 1

    rows = []
    for section in expected_sections:
        penalty = penalties[section]
        count = counts[section]
        rows.append(
            {
                "sequence": sequence,
                "metric": metric,
                "official_score": _format_float(score),
                "penalty_type": SECTION_SLUGS[section],
                "penalty_label": section,
                "penalty": _format_float(penalty),
                "count": count,
                "weighted_penalty": _format_float(None if penalty is None else penalty * count),
                "log_path": str(log_path),
            }
        )
    return rows


def parse_seg_log(log_path: Path, low_jaccard_threshold: float = 0.5):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    score = parse_official_score(text, "SEG")
    sequence = _sequence_from_result_dir(log_path.parent)

    current_t = ""
    current_z = ""
    low_rows = []
    object_count = 0
    low_count = 0
    zero_count = 0
    min_jaccard = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        frame = SEG_FRAME_RE.match(line)
        if frame:
            current_t = frame.group("t")
            current_z = frame.group("z") or ""
            continue

        match = SEG_OBJECT_RE.match(line)
        if not match:
            continue

        object_count += 1
        jaccard = float(match.group("jaccard"))
        min_jaccard = jaccard if min_jaccard is None else min(min_jaccard, jaccard)
        if jaccard == 0:
            zero_count += 1
        if jaccard < low_jaccard_threshold:
            low_count += 1
            low_rows.append(
                {
                    "sequence": sequence,
                    "official_score": _format_float(score),
                    "t": current_t,
                    "z": current_z,
                    "gt_label": match.group("gt_label"),
                    "jaccard": _format_float(jaccard),
                    "threshold": _format_float(low_jaccard_threshold),
                    "log_path": str(log_path),
                }
            )

    summary = {
        "sequence": sequence,
        "metric": "SEG",
        "official_score": _format_float(score),
        "objects": object_count,
        "low_jaccard_objects": low_count,
        "zero_jaccard_objects": zero_count,
        "min_jaccard": _format_float(min_jaccard),
        "threshold": _format_float(low_jaccard_threshold),
        "log_path": str(log_path),
    }
    return low_rows, summary


def find_result_dirs(dataset_root: Path, sequences: list[str] | None = None):
    if sequences is None:
        result_dirs = sorted(
            path for path in dataset_root.iterdir() if path.is_dir() and path.name.endswith("_RES")
        )
    else:
        result_dirs = [dataset_root / f"{_normalize_sequence(sequence)}_RES" for sequence in sequences]
    return [path for path in result_dirs if path.is_dir()]


def summarize_official_logs(
    dataset_root: Path,
    sequences: list[str] | None = None,
    low_jaccard_threshold: float = 0.5,
):
    result_dirs = find_result_dirs(dataset_root, sequences)
    tra_rows = []
    det_rows = []
    seg_low_rows = []
    seg_summary_rows = []

    for result_dir in result_dirs:
        tra_log = result_dir / "TRA_log.txt"
        if tra_log.is_file():
            tra_rows.extend(parse_tra_det_log(tra_log, "TRA"))

        det_log = result_dir / "DET_log.txt"
        if det_log.is_file():
            det_rows.extend(parse_tra_det_log(det_log, "DET"))

        seg_log = result_dir / "SEG_log.txt"
        if seg_log.is_file():
            low_rows, summary = parse_seg_log(seg_log, low_jaccard_threshold=low_jaccard_threshold)
            seg_low_rows.extend(low_rows)
            seg_summary_rows.append(summary)

    outputs = {
        "tra_penalties": dataset_root / "ctc_TRA_penalty_counts.csv",
        "det_penalties": dataset_root / "ctc_DET_penalty_counts.csv",
        "seg_low_jaccard": dataset_root / "ctc_SEG_low_jaccard_objects.csv",
        "seg_summary": dataset_root / "ctc_SEG_summary.csv",
    }

    _write_csv(
        outputs["tra_penalties"],
        [
            "sequence",
            "metric",
            "official_score",
            "penalty_type",
            "penalty_label",
            "penalty",
            "count",
            "weighted_penalty",
            "log_path",
        ],
        tra_rows,
    )
    _write_csv(
        outputs["det_penalties"],
        [
            "sequence",
            "metric",
            "official_score",
            "penalty_type",
            "penalty_label",
            "penalty",
            "count",
            "weighted_penalty",
            "log_path",
        ],
        det_rows,
    )
    _write_csv(
        outputs["seg_low_jaccard"],
        ["sequence", "official_score", "t", "z", "gt_label", "jaccard", "threshold", "log_path"],
        seg_low_rows,
    )
    _write_csv(
        outputs["seg_summary"],
        [
            "sequence",
            "metric",
            "official_score",
            "objects",
            "low_jaccard_objects",
            "zero_jaccard_objects",
            "min_jaccard",
            "threshold",
            "log_path",
        ],
        seg_summary_rows,
    )

    return {
        "result_dirs": result_dirs,
        "outputs": outputs,
        "tra_rows": len(tra_rows),
        "det_rows": len(det_rows),
        "seg_low_rows": len(seg_low_rows),
        "seg_summary_rows": len(seg_summary_rows),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _symlink(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError as exc:
        raise RuntimeError(
            f"Could not create symlink {dst} -> {src}. Put GT/RES folders under the same dataset root "
            "or run this on a system where symlinks are allowed."
        ) from exc


def prepare_eval_root(dataset_root: Path, source_root: Path | None, sequence: str):
    result_dir = dataset_root / f"{sequence}_RES"
    gt_dir = dataset_root / f"{sequence}_GT"
    source_gt_dir = source_root / f"{sequence}_GT" if source_root is not None else None

    if not result_dir.is_dir():
        raise FileNotFoundError(f"Missing result folder: {result_dir}")
    if gt_dir.is_dir():
        return dataset_root, None
    if source_gt_dir is None or not source_gt_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {sequence}_GT beside results and no usable --source-root was provided."
        )

    tmp = tempfile.TemporaryDirectory(prefix=f"ctc_eval_{sequence}_")
    tmp_root = Path(tmp.name)
    _symlink(result_dir.resolve(), tmp_root / f"{sequence}_RES")
    _symlink(source_gt_dir.resolve(), tmp_root / f"{sequence}_GT")

    source_img_dir = source_root / sequence if source_root is not None else None
    if source_img_dir is not None and source_img_dir.is_dir():
        _symlink(source_img_dir.resolve(), tmp_root / sequence)

    return tmp_root, tmp


def run_metric(
    metric: str,
    executable: Path,
    eval_root: Path,
    sequence: str,
    digits: int,
    result_dir: Path,
    det_penalize_extra_detections: int | None,
):
    metric = metric.upper()
    command = [str(executable), str(eval_root), sequence, str(digits)]
    if metric == "DET" and det_penalize_extra_detections is not None:
        command.append(str(det_penalize_extra_detections))

    completed = subprocess.run(command, capture_output=True, text=True)
    output_path = result_dir / f"{metric}_runner_output.txt"
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("command:\n")
        handle.write(" ".join(command) + "\n\n")
        handle.write(f"returncode: {completed.returncode}\n\n")
        handle.write("stdout:\n")
        handle.write(completed.stdout)
        handle.write("\n\nstderr:\n")
        handle.write(completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"{metric} evaluation failed with return code {completed.returncode}. See {output_path}."
        )

    score = parse_score(completed.stdout)
    log_name = f"{metric}_log.txt"
    log_path = result_dir / log_name
    archived_log = ""
    if log_path.is_file():
        official_score = parse_official_score(
            log_path.read_text(encoding="utf-8", errors="replace"),
            metric,
        )
        if official_score is not None:
            score = official_score
        archive_dir = result_dir / "ctc_metric_logs"
        archive_dir.mkdir(exist_ok=True)
        archived = archive_dir / log_name
        shutil.copy2(log_path, archived)
        archived_log = str(archived)

    return {
        "metric": metric,
        "score": "" if score is None else f"{score:.12g}",
        "stdout_log": str(output_path),
        "ctc_log": archived_log,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run official CTC metrics on an existing result folder.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Root containing <sequence>_RES.")
    parser.add_argument(
        "--source-root",
        default=None,
        type=Path,
        help="Optional original CTC root containing <sequence>_GT if GT is not beside results.",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        type=str,
        help="CTC sequence ID, e.g. 01 or 02. Required unless --parse-logs-only is used.",
    )
    parser.add_argument("--digits", default="auto", choices=["auto", "3", "4"], help="CTC time index digits.")
    parser.add_argument(
        "--ctc-software-dir",
        default=None,
        type=Path,
        help="Root of downloaded CTC evaluation software, containing Mac/Linux/Win subfolders.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=list(METRICS),
        help="Metrics to run.",
    )
    parser.add_argument(
        "--det-penalize-extra-detections",
        choices=["0", "1"],
        default=None,
        help="Optional fourth DET argument. Leave unset for the official default.",
    )
    parser.add_argument(
        "--skip-format-validation",
        action="store_true",
        help="Skip validate_ctc_result_format.py before running metrics.",
    )
    parser.add_argument(
        "--parse-logs-only",
        action="store_true",
        help="Skip metric execution and only write CSV summaries from existing official logs.",
    )
    parser.add_argument(
        "--seg-low-jaccard-threshold",
        default=0.5,
        type=float,
        help="SEG objects below this official Jaccard value are written to ctc_SEG_low_jaccard_objects.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    source_root = args.source_root.resolve() if args.source_root is not None else None
    sequence = _normalize_sequence(args.sequence) if args.sequence is not None else None

    if args.parse_logs_only:
        log_summary = summarize_official_logs(
            dataset_root=dataset_root,
            sequences=[sequence] if sequence is not None else None,
            low_jaccard_threshold=args.seg_low_jaccard_threshold,
        )
        print(f"[CTC EVAL] parsed official logs from {len(log_summary['result_dirs'])} result folder(s)")
        for path in log_summary["outputs"].values():
            print(f"[CTC EVAL] wrote {path}")
        return 0

    if sequence is None:
        raise SystemExit("[CTC EVAL] --sequence is required unless --parse-logs-only is used.")
    result_dir = dataset_root / f"{sequence}_RES"

    from validate_ctc_result_format import ValidationError, resolve_digits, validate_ctc_result_format

    digits = resolve_digits(args.digits, dataset_root, source_root, sequence)
    if not args.skip_format_validation:
        try:
            report = validate_ctc_result_format(
                dataset_root=dataset_root,
                source_root=source_root,
                sequence=sequence,
                digits_arg=str(digits),
            )
        except ValidationError as exc:
            raise SystemExit(f"[CTC EVAL] format validation failed: {exc}")
        print(
            "[CTC EVAL] format OK "
            f"sequence={report['sequence']} digits={report['digits']} frames={report['frames']} tracks={report['tracks']}"
        )

    eval_root, temp_root = prepare_eval_root(dataset_root, source_root, sequence)
    try:
        rows = []
        for metric in args.metrics:
            executable = find_metric_executable(metric, args.ctc_software_dir)
            print(f"[CTC EVAL] running {metric} with {executable}")
            rows.append(
                run_metric(
                    metric=metric,
                    executable=executable,
                    eval_root=eval_root,
                    sequence=sequence,
                    digits=digits,
                    result_dir=result_dir,
                    det_penalize_extra_detections=(
                        None
                        if args.det_penalize_extra_detections is None
                        else int(args.det_penalize_extra_detections)
                    ),
                )
            )
    finally:
        if temp_root is not None:
            temp_root.cleanup()

    summary_path = result_dir / "ctc_metrics_summary.csv"
    _write_csv(summary_path, ["metric", "score", "stdout_log", "ctc_log"], rows)

    log_summary = summarize_official_logs(
        dataset_root=dataset_root,
        low_jaccard_threshold=args.seg_low_jaccard_threshold,
    )

    for row in rows:
        score = row["score"] if row["score"] else "unparsed"
        print(f"[CTC EVAL] {row['metric']}={score}")
    print(f"[CTC EVAL] wrote {summary_path}")
    print(f"[CTC EVAL] parsed official logs from {len(log_summary['result_dirs'])} result folder(s)")
    for path in log_summary["outputs"].values():
        print(f"[CTC EVAL] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
