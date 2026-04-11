#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


DEFAULT_ROOT_DIR = Path(r"Z:\Interpolation_paper\Benchmark\BF-C2DL-HSC\Interpol\02\interpolated_exp4")
DEFAULT_START_FRAME = 2381
DEFAULT_END_FRAME = 7123
DEFAULT_OUTPUT_FOLDER = "trial_2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy a contiguous PNG frame range into a new trial folder."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help=f"Folder containing the PNG frames (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=DEFAULT_START_FRAME,
        help=f"First frame number to copy, inclusive (default: {DEFAULT_START_FRAME})",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=DEFAULT_END_FRAME,
        help=f"Last frame number to copy, inclusive (default: {DEFAULT_END_FRAME})",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Name of the destination folder created inside root-dir (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = args.root_dir

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root directory does not exist or is not a directory: {root_dir}")
    if args.start_frame > args.end_frame:
        raise ValueError("--start-frame must be less than or equal to --end-frame")

    output_dir = root_dir / args.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_files = []

    for frame_num in range(args.start_frame, args.end_frame + 1):
        filename = f"{frame_num:07d}.png"
        source_path = root_dir / filename
        dest_path = output_dir / filename

        if not source_path.is_file():
            missing_files.append(filename)
            continue

        shutil.copy2(source_path, dest_path)
        copied_count += 1

    print(f"Source folder: {root_dir}")
    print(f"Destination folder: {output_dir}")
    print(f"Copied {copied_count} PNG files.")

    if missing_files:
        print(f"Missing {len(missing_files)} files in requested range.")
        preview = ", ".join(missing_files[:10])
        print(f"First missing files: {preview}")
    else:
        print("No files were missing in the requested range.")


if __name__ == "__main__":
    main()
