#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import Button


def _natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _load_file_list(folder: Path, pattern: str, exclude_substrings: list[str] | None = None):
    files = sorted(folder.glob(pattern), key=lambda p: _natural_sort_key(p.name))
    if exclude_substrings:
        files = [
            path
            for path in files
            if not any(substr in path.name for substr in exclude_substrings)
        ]
    if not files:
        raise FileNotFoundError(f"No files found in '{folder}' with pattern '{pattern}'.")
    return files


def _read_image(path: Path):
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(str(path))
    else:
        image = plt.imread(str(path))
    return _prepare_image_for_display(image)


def _read_mask(path: Path):
    if path.suffix.lower() in {".tif", ".tiff"}:
        mask = tifffile.imread(str(path))
    else:
        mask = plt.imread(str(path))
    return _prepare_mask_for_display(mask)


def _prepare_image_for_display(image: np.ndarray):
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return arr[..., :3]
        if arr.shape[0] in (3, 4):
            return np.moveaxis(arr[:3, ...], 0, -1)
        return arr[..., 0]
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _prepare_mask_for_display(mask: np.ndarray):
    arr = np.asarray(mask)
    if arr.ndim == 2:
        return arr.astype(np.int64)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return arr[..., 0].astype(np.int64)
        return arr[..., 0].astype(np.int64)
    raise ValueError(f"Unsupported mask shape: {arr.shape}")


def _label_overlay(mask: np.ndarray, alpha: float):
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    for label_value in labels.tolist():
        hue = (label_value * 0.618033988749895) % 1.0
        rgb = hsv_to_rgb([hue, 0.85, 1.0])
        pix = mask == label_value
        overlay[pix, :3] = rgb
        overlay[pix, 3] = alpha
    return overlay


class OverlayViewer:
    def __init__(self, image_files, mask_files, alpha: float, start_index: int):
        self.image_files = image_files
        self.mask_files = mask_files
        self.alpha = alpha
        self.frame_count = min(len(image_files), len(mask_files))
        self.index = max(0, min(start_index, self.frame_count - 1))

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.16)

        prev_ax = self.fig.add_axes([0.33, 0.04, 0.14, 0.07])
        next_ax = self.fig.add_axes([0.53, 0.04, 0.14, 0.07])
        self.prev_button = Button(prev_ax, "Prev")
        self.next_button = Button(next_ax, "Next")
        self.prev_button.on_clicked(self._on_prev)
        self.next_button.on_clicked(self._on_next)
        self.fig.canvas.mpl_connect("key_press_event", self._on_keypress)

        self._draw_frame()

    def _on_prev(self, _event):
        self.index = (self.index - 1) % self.frame_count
        self._draw_frame()

    def _on_next(self, _event):
        self.index = (self.index + 1) % self.frame_count
        self._draw_frame()

    def _on_keypress(self, event):
        if event.key in {"left", "a"}:
            self._on_prev(event)
        elif event.key in {"right", "d"}:
            self._on_next(event)

    def _draw_frame(self):
        image_path = self.image_files[self.index]
        mask_path = self.mask_files[self.index]

        image = _read_image(image_path)
        mask = _read_mask(mask_path)

        image_h, image_w = image.shape[:2]
        if mask.shape[0] != image_h or mask.shape[1] != image_w:
            raise ValueError(
                f"Shape mismatch at frame {self.index}: image={image.shape}, mask={mask.shape}. "
                "Masks and images must have identical width/height."
            )

        self.ax.clear()
        if image.ndim == 2:
            self.ax.imshow(image, cmap="gray")
        else:
            self.ax.imshow(image)

        self.ax.imshow(_label_overlay(mask, self.alpha))
        self.ax.set_axis_off()
        self.ax.set_title(
            f"Frame {self.index + 1}/{self.frame_count}\n"
            f"Image: {image_path.name} | Mask: {mask_path.name}"
        )
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive overlay viewer for original images + tracked masks."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Folder containing original image frames.",
    )
    parser.add_argument(
        "--mask-dir",
        required=True,
        type=Path,
        help="Folder containing tracked mask frames.",
    )
    parser.add_argument(
        "--image-pattern",
        default="*.tif",
        type=str,
        help="Glob for original images (default: *.tif).",
    )
    parser.add_argument(
        "--image-exclude-substring",
        action="append",
        default=[],
        help="Exclude original-image files containing this substring. Can be passed multiple times.",
    )
    parser.add_argument(
        "--mask-pattern",
        default="mask*.tif",
        type=str,
        help="Glob for tracked masks (default: mask*.tif).",
    )
    parser.add_argument(
        "--alpha",
        default=0.45,
        type=float,
        help="Overlay alpha between 0 and 1 (default: 0.45).",
    )
    parser.add_argument(
        "--start-index",
        default=0,
        type=int,
        help="0-based initial frame index (default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = args.image_dir.resolve()
    mask_dir = args.mask_dir.resolve()

    if not image_dir.is_dir():
        raise NotADirectoryError(f"image-dir does not exist or is not a directory: {image_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"mask-dir does not exist or is not a directory: {mask_dir}")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be between 0 and 1.")

    image_files = _load_file_list(
        image_dir,
        args.image_pattern,
        exclude_substrings=args.image_exclude_substring,
    )
    mask_files = _load_file_list(mask_dir, args.mask_pattern)

    if len(image_files) != len(mask_files):
        print(
            "[WARN] Image/mask count mismatch. "
            f"images={len(image_files)}, masks={len(mask_files)}. "
            f"Using first {min(len(image_files), len(mask_files))} pairs by sorted order."
        )

    viewer = OverlayViewer(
        image_files=image_files,
        mask_files=mask_files,
        alpha=args.alpha,
        start_index=args.start_index,
    )
    viewer.show()


if __name__ == "__main__":
    main()
