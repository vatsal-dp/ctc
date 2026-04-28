#!/usr/bin/env python3

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from skimage import io
from skimage.transform import resize


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class RescaleResult:
    source: Path
    destination: Path
    original_shape: tuple[int, ...]
    resized_shape: tuple[int, ...]


def _natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _load_file_list(folder: Path, pattern: str):
    files = [
        path
        for path in sorted(folder.glob(pattern), key=lambda p: _natural_sort_key(p.name))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"No image files found in '{folder}' with pattern '{pattern}'.")
    return files


def _read_array(path: Path):
    if path.suffix.lower() in {".tif", ".tiff"}:
        return tifffile.imread(str(path))
    return io.imread(str(path))


def _write_array(path: Path, array: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(str(path), array)
    else:
        io.imsave(str(path), array, check_contrast=False)


def _spatial_output_shape(array: np.ndarray, scale: float):
    if array.ndim < 2:
        raise ValueError(f"Expected at least 2D image data, got shape {array.shape}.")

    height, width = array.shape[:2]
    scaled_height = max(1, int(round(height * scale)))
    scaled_width = max(1, int(round(width * scale)))

    if array.ndim == 2:
        return (scaled_height, scaled_width)
    return (scaled_height, scaled_width, *array.shape[2:])


def _restore_dtype(array: np.ndarray, dtype: np.dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        array = np.clip(np.rint(array), info.min, info.max)
    elif np.issubdtype(dtype, np.floating):
        array = array.astype(dtype, copy=False)
    elif dtype == np.bool_:
        array = array >= 0.5
    else:
        raise TypeError(f"Unsupported dtype for resized output: {dtype}")
    return array.astype(dtype, copy=False)


def resize_image_array(array: np.ndarray, scale: float, order: int = 1):
    output_shape = _spatial_output_shape(array, scale)
    resized = resize(
        array,
        output_shape=output_shape,
        order=order,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return _restore_dtype(resized, array.dtype)


def resize_mask_array(array: np.ndarray, scale: float):
    output_shape = _spatial_output_shape(array, scale)
    resized = resize(
        array,
        output_shape=output_shape,
        order=0,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )
    return _restore_dtype(resized, array.dtype)


def _rescale_files(files, output_dir: Path, scale: float, is_mask: bool, image_order: int):
    results = []
    for source in files:
        array = _read_array(source)
        resized = resize_mask_array(array, scale) if is_mask else resize_image_array(array, scale, image_order)
        destination = output_dir / source.name
        _write_array(destination, resized)
        results.append(
            RescaleResult(
                source=source,
                destination=destination,
                original_shape=tuple(array.shape),
                resized_shape=tuple(resized.shape),
            )
        )
    return results


def rescale_dataset(
    image_dir: Path,
    mask_dir: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    image_pattern: str = "*.tif",
    mask_pattern: str = "mask*.tif",
    scale: float = 0.5,
    image_order: int = 1,
    copy_track_file: bool = True,
):
    if not image_dir.is_dir():
        raise NotADirectoryError(f"image-dir does not exist or is not a directory: {image_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"mask-dir does not exist or is not a directory: {mask_dir}")
    if scale <= 0:
        raise ValueError("--scale must be greater than zero.")
    if image_order not in range(0, 6):
        raise ValueError("--image-order must be an integer from 0 to 5.")

    image_files = _load_file_list(image_dir, image_pattern)
    mask_files = _load_file_list(mask_dir, mask_pattern)

    image_results = _rescale_files(
        files=image_files,
        output_dir=output_image_dir,
        scale=scale,
        is_mask=False,
        image_order=image_order,
    )
    mask_results = _rescale_files(
        files=mask_files,
        output_dir=output_mask_dir,
        scale=scale,
        is_mask=True,
        image_order=image_order,
    )

    track_file = mask_dir / "res_track.txt"
    copied_track_file = None
    if copy_track_file and track_file.is_file():
        copied_track_file = output_mask_dir / track_file.name
        copied_track_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(track_file, copied_track_file)

    return image_results, mask_results, copied_track_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rescale image frames and tracking masks, preserving mask labels with nearest-neighbor resizing."
    )
    parser.add_argument("--image-dir", required=True, type=Path, help="Folder containing image frames.")
    parser.add_argument("--mask-dir", required=True, type=Path, help="Folder containing mask frames.")
    parser.add_argument("--output-image-dir", required=True, type=Path, help="Destination folder for resized images.")
    parser.add_argument("--output-mask-dir", required=True, type=Path, help="Destination folder for resized masks.")
    parser.add_argument("--image-pattern", default="*.tif", help="Glob for image frames (default: *.tif).")
    parser.add_argument("--mask-pattern", default="mask*.tif", help="Glob for mask frames (default: mask*.tif).")
    parser.add_argument("--scale", default=0.5, type=float, help="Spatial scale factor (default: 0.5).")
    parser.add_argument(
        "--image-order",
        default=1,
        type=int,
        help="Spline interpolation order for images, 0-5 (default: 1). Masks always use order 0.",
    )
    parser.add_argument(
        "--no-copy-track-file",
        action="store_true",
        help="Do not copy <mask-dir>/res_track.txt into the resized mask directory.",
    )
    return parser.parse_args()


def _shape_summary(results):
    first = results[0]
    last = results[-1]
    return (
        f"{len(results)} files; "
        f"first {first.original_shape}->{first.resized_shape}; "
        f"last {last.original_shape}->{last.resized_shape}"
    )


def main():
    args = parse_args()
    image_results, mask_results, copied_track_file = rescale_dataset(
        image_dir=args.image_dir.resolve(),
        mask_dir=args.mask_dir.resolve(),
        output_image_dir=args.output_image_dir.resolve(),
        output_mask_dir=args.output_mask_dir.resolve(),
        image_pattern=args.image_pattern,
        mask_pattern=args.mask_pattern,
        scale=args.scale,
        image_order=args.image_order,
        copy_track_file=not args.no_copy_track_file,
    )

    print(f"[IMAGES] {_shape_summary(image_results)}")
    print(f"[MASKS]  {_shape_summary(mask_results)}")
    if copied_track_file is not None:
        print(f"[TRACK]  copied {copied_track_file}")
    print("[DONE] Masks were resized with nearest-neighbor interpolation to preserve integer labels.")


if __name__ == "__main__":
    main()
