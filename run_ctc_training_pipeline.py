#!/usr/bin/env python3

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


METRICS = ("TRA", "SEG", "DET")


def _normalize_sequence(sequence: str) -> str:
    text = str(sequence)
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def _format_command(command: list[str]) -> str:
    if platform.system().lower() == "windows":
        return subprocess.list2cmdline([str(part) for part in command])
    return " ".join(_shell_quote(str(part)) for part in command)


def _shell_quote(text: str) -> str:
    if not text:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:"
    if all(char in safe for char in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _run_command(command: list[str], log_path: Path, dry_run: bool):
    printable = _format_command(command)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[PIPELINE] {printable}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"command={printable}\n\n")

        if dry_run:
            log.write("[DRY RUN] Command was not executed.\n")
            return

        process = subprocess.Popen(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)

        return_code = process.wait()
        log.write(f"\nreturncode={return_code}\n")
        if return_code != 0:
            raise RuntimeError(f"Command failed with return code {return_code}. See {log_path}.")


def _copy_tree_if_needed(src: Path, dst: Path, dry_run: bool):
    if not src.is_dir():
        raise NotADirectoryError(f"Missing required folder: {src}")
    print(f"[PIPELINE] staging GT: {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _script_path(script_dir: Path, name: str):
    path = script_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing pipeline script: {path}")
    return path


def _mask_dir_for_sequence(source_root: Path, sequence: str, mask_dir_template: str | None):
    if mask_dir_template:
        return Path(mask_dir_template.format(sequence=sequence, seq=sequence)).expanduser()
    return source_root / f"{sequence}_ERR_SEG"


def _run_sequence(args, script_dir: Path, sequence: str):
    python_exe = Path(args.python).expanduser()
    source_root = args.source_root.resolve()
    eval_root = args.eval_root.resolve()
    log_dir = args.log_dir.resolve() if args.log_dir is not None else eval_root / "pipeline_logs"
    result_dir = eval_root / f"{sequence}_RES"

    if args.stage_gt != "none":
        gt_src = source_root / f"{sequence}_GT"
        gt_dst = eval_root / f"{sequence}_GT"
        if args.stage_gt == "copy":
            _copy_tree_if_needed(gt_src, gt_dst, args.dry_run)
        elif args.stage_gt == "symlink":
            print(f"[PIPELINE] staging GT symlink: {gt_src} -> {gt_dst}")
            if not args.dry_run:
                gt_dst.parent.mkdir(parents=True, exist_ok=True)
                if not gt_dst.exists():
                    try:
                        gt_dst.symlink_to(gt_src.resolve(), target_is_directory=True)
                    except OSError as exc:
                        raise RuntimeError(
                            "Could not create GT symlink. On Windows, use --stage-gt copy "
                            "unless Developer Mode/admin symlinks are enabled."
                        ) from exc

    if not args.skip_tracking:
        mask_dir = _mask_dir_for_sequence(source_root, sequence, args.mask_dir_template)
        command = [
            str(python_exe),
            str(_script_path(script_dir, "run_tiptracking_standalone.py")),
            "--mask-dir",
            str(mask_dir),
            "--mask-pattern",
            args.mask_pattern,
            "--output-dir",
            str(eval_root),
            "--position",
            sequence,
            "--time-series-threshold",
            str(args.time_series_threshold),
            "--output-digits",
            args.output_digits,
        ]
        if not args.strict_matlab_id_matching:
            command.append("--no-strict-matlab-id-matching")
        _run_command(command, log_dir / f"{sequence}_tracking.log", args.dry_run)

    if not args.skip_validation:
        command = [
            str(python_exe),
            str(_script_path(script_dir, "validate_ctc_result_format.py")),
            "--dataset-root",
            str(eval_root),
            "--source-root",
            str(source_root),
            "--sequence",
            sequence,
            "--digits",
            args.digits,
        ]
        _run_command(command, log_dir / f"{sequence}_validate.log", args.dry_run)

    if not args.skip_evaluation:
        if args.ctc_software_dir is None:
            raise ValueError("--ctc-software-dir is required unless --skip-evaluation is set.")
        command = [
            str(python_exe),
            str(_script_path(script_dir, "evaluate_ctc_results.py")),
            "--dataset-root",
            str(eval_root),
            "--source-root",
            str(source_root),
            "--sequence",
            sequence,
            "--digits",
            args.digits,
            "--ctc-software-dir",
            str(args.ctc_software_dir.resolve()),
            "--metrics",
            *args.metrics,
        ]
        if args.det_penalize_extra_detections is not None:
            command.extend(["--det-penalize-extra-detections", args.det_penalize_extra_detections])
        _run_command(command, log_dir / f"{sequence}_evaluate.log", args.dry_run)

    if not args.skip_failure_analysis:
        command = [
            str(python_exe),
            str(_script_path(script_dir, "analyze_tracking_failures.py")),
            "--dataset-root",
            str(eval_root),
            "--source-root",
            str(source_root),
            "--sequence",
            sequence,
            "--out",
            str(result_dir.parent / f"{sequence}_FAILURES"),
            "--digits",
            args.digits,
        ]
        _run_command(command, log_dir / f"{sequence}_failures.log", args.dry_run)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the local tracking-only CTC training pipeline: track, validate format, "
            "run official CTC metrics, and generate failure reports."
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Original CTC dataset root containing 01_ERR_SEG, 01_GT, 01, etc.",
    )
    parser.add_argument(
        "--eval-root",
        required=True,
        type=Path,
        help="Evaluation workspace root where 01_RES/02_RES and reports are written.",
    )
    parser.add_argument(
        "--ctc-software-dir",
        default=None,
        type=Path,
        help="Root EvaluationSoftware folder containing Win/Linux/Mac subfolders.",
    )
    parser.add_argument("--sequences", nargs="+", default=["01", "02"], help="Sequences to process.")
    parser.add_argument("--mask-pattern", default="mask*.tif", help="Input segmentation mask glob.")
    parser.add_argument(
        "--mask-dir-template",
        default=None,
        help=(
            "Optional mask-dir template with {sequence}, e.g. "
            "'D:/data/BF-C2DL-HSC/{sequence}_ERR_SEG'. Defaults to source-root/{sequence}_ERR_SEG."
        ),
    )
    parser.add_argument("--time-series-threshold", default=1, type=int)
    parser.add_argument("--output-digits", choices=["auto", "3", "4"], default="auto")
    parser.add_argument("--digits", choices=["auto", "3", "4"], default="auto", help="CTC evaluation digit width.")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument(
        "--det-penalize-extra-detections",
        choices=["0", "1"],
        default=None,
        help="Optional DET fourth argument. Leave unset for official default.",
    )
    parser.add_argument(
        "--stage-gt",
        choices=["copy", "symlink", "none"],
        default="copy",
        help="Stage <sequence>_GT beside results before evaluation. Default is copy for Windows friendliness.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run child scripts.")
    parser.add_argument("--log-dir", default=None, type=Path, help="Folder for pipeline step logs.")
    parser.add_argument("--skip-tracking", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-failure-analysis", action="store_true")
    parser.add_argument(
        "--no-strict-matlab-id-matching",
        dest="strict_matlab_id_matching",
        action="store_false",
        help="Pass through to run_tiptracking_standalone.py.",
    )
    parser.set_defaults(strict_matlab_id_matching=True)
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with later sequences after a failure.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write logs without executing.")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    args.source_root = args.source_root.resolve()
    args.eval_root = args.eval_root.resolve()

    if not args.source_root.is_dir():
        raise SystemExit(f"source-root does not exist or is not a directory: {args.source_root}")
    if args.ctc_software_dir is not None:
        args.ctc_software_dir = args.ctc_software_dir.resolve()
        if not args.ctc_software_dir.is_dir():
            raise SystemExit(f"ctc-software-dir does not exist or is not a directory: {args.ctc_software_dir}")

    if not args.dry_run:
        args.eval_root.mkdir(parents=True, exist_ok=True)

    sequences = [_normalize_sequence(sequence) for sequence in args.sequences]
    failures = []

    print(f"[PIPELINE] source_root={args.source_root}")
    print(f"[PIPELINE] eval_root={args.eval_root}")
    print(f"[PIPELINE] sequences={', '.join(sequences)}")

    for sequence in sequences:
        print(f"\n[PIPELINE] ===== sequence {sequence} =====", flush=True)
        try:
            _run_sequence(args, script_dir, sequence)
        except Exception as exc:
            failures.append((sequence, exc))
            print(f"[PIPELINE] sequence {sequence} failed: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    if failures:
        print("\n[PIPELINE] FAILED")
        for sequence, exc in failures:
            print(f"  {sequence}: {exc}")
        return 1

    print("\n[PIPELINE] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
