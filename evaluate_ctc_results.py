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

from validate_ctc_result_format import ValidationError, _normalize_sequence, resolve_digits, validate_ctc_result_format


METRICS = ("TRA", "SEG", "DET")


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
    parser.add_argument("--sequence", required=True, type=str, help="CTC sequence ID, e.g. 01 or 02.")
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
    return parser.parse_args()


def main():
    args = parse_args()
    sequence = _normalize_sequence(args.sequence)
    dataset_root = args.dataset_root.resolve()
    source_root = args.source_root.resolve() if args.source_root is not None else None
    result_dir = dataset_root / f"{sequence}_RES"

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
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "score", "stdout_log", "ctc_log"])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        score = row["score"] if row["score"] else "unparsed"
        print(f"[CTC EVAL] {row['metric']}={score}")
    print(f"[CTC EVAL] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
