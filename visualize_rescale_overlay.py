#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure


GOLDEN_RATIO_CONJUGATE = 0.618033988749895
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _natural_sort_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _load_file_list(folder: Path, pattern: str):
    files = [
        path
        for path in sorted(folder.glob(pattern), key=lambda p: _natural_sort_key(p.name))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
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


def _track_color(label_value: int):
    hue = (int(label_value) * GOLDEN_RATIO_CONJUGATE) % 1.0
    return np.asarray(hsv_to_rgb([hue, 0.85, 1.0]), dtype=np.float32)


def _label_overlay(mask: np.ndarray, alpha: float):
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    for label_value in labels.tolist():
        pix = mask == label_value
        overlay[pix, :3] = _track_color(int(label_value))
        overlay[pix, 3] = alpha
    return overlay


def _validate_pair_shapes(image: np.ndarray, mask: np.ndarray, label: str):
    if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
        raise ValueError(f"{label} image/mask shape mismatch: image={image.shape}, mask={mask.shape}")


def _labels(mask: np.ndarray):
    labels = np.unique(mask)
    return set(labels[labels != 0].astype(int).tolist())


def _frame_metrics(original_mask: np.ndarray, scaled_mask: np.ndarray, scale: float):
    original_labels = _labels(original_mask)
    scaled_labels = _labels(scaled_mask)
    original_fg = int(np.count_nonzero(original_mask))
    scaled_fg = int(np.count_nonzero(scaled_mask))
    expected_scaled_fg = original_fg * scale * scale
    normalized_fg_ratio = scaled_fg / expected_scaled_fg if expected_scaled_fg > 0 else 1.0

    return {
        "original_labels": len(original_labels),
        "scaled_labels": len(scaled_labels),
        "lost_labels": sorted(original_labels - scaled_labels),
        "new_labels": sorted(scaled_labels - original_labels),
        "original_foreground_pixels": original_fg,
        "scaled_foreground_pixels": scaled_fg,
        "normalized_foreground_ratio": normalized_fg_ratio,
    }


def _draw_overlay(ax, image: np.ndarray, mask: np.ndarray, title: str, alpha: float):
    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)
    ax.imshow(_label_overlay(mask, alpha))
    ax.set_title(title)
    ax.set_axis_off()


def _select_indices(frame_count: int, max_frames: int | None, every: int):
    indices = list(range(0, frame_count, every))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def export_rescale_overlay_comparisons(
    original_image_files,
    original_mask_files,
    scaled_image_files,
    scaled_mask_files,
    output_dir: Path,
    scale: float = 0.5,
    alpha: float = 0.45,
    dpi: int = 150,
    max_frames: int | None = None,
    every: int = 1,
):
    frame_count = min(
        len(original_image_files),
        len(original_mask_files),
        len(scaled_image_files),
        len(scaled_mask_files),
    )
    if frame_count == 0:
        raise ValueError("No complete original/rescaled image-mask pairs are available.")
    if scale <= 0:
        raise ValueError("--scale must be greater than zero.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("--alpha must be between 0 and 1.")
    if every <= 0:
        raise ValueError("--every must be positive.")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(12, 6))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(wspace=0.04)

    output_paths = []
    metrics_rows = []

    for frame_index in _select_indices(frame_count, max_frames=max_frames, every=every):
        original_image = _read_image(original_image_files[frame_index])
        original_mask = _read_mask(original_mask_files[frame_index])
        scaled_image = _read_image(scaled_image_files[frame_index])
        scaled_mask = _read_mask(scaled_mask_files[frame_index])

        _validate_pair_shapes(original_image, original_mask, "Original")
        _validate_pair_shapes(scaled_image, scaled_mask, "Rescaled")

        metrics = _frame_metrics(original_mask, scaled_mask, scale=scale)
        metrics_rows.append(
            {
                "frame_index": frame_index,
                "original_image": original_image_files[frame_index].name,
                "original_mask": original_mask_files[frame_index].name,
                "scaled_image": scaled_image_files[frame_index].name,
                "scaled_mask": scaled_mask_files[frame_index].name,
                "original_shape": "x".join(str(v) for v in original_image.shape[:2]),
                "scaled_shape": "x".join(str(v) for v in scaled_image.shape[:2]),
                "original_labels": metrics["original_labels"],
                "scaled_labels": metrics["scaled_labels"],
                "lost_labels": " ".join(str(v) for v in metrics["lost_labels"]),
                "new_labels": " ".join(str(v) for v in metrics["new_labels"]),
                "original_foreground_pixels": metrics["original_foreground_pixels"],
                "scaled_foreground_pixels": metrics["scaled_foreground_pixels"],
                "normalized_foreground_ratio": f"{metrics['normalized_foreground_ratio']:.6f}",
            }
        )

        for ax in axes:
            ax.clear()

        _draw_overlay(
            axes[0],
            original_image,
            original_mask,
            f"Original frame {frame_index}\n{original_image.shape[0]}x{original_image.shape[1]}",
            alpha,
        )
        _draw_overlay(
            axes[1],
            scaled_image,
            scaled_mask,
            (
                f"Rescaled frame {frame_index}\n"
                f"{scaled_image.shape[0]}x{scaled_image.shape[1]} | "
                f"labels {metrics['scaled_labels']}/{metrics['original_labels']}"
            ),
            alpha,
        )

        out_path = output_dir / f"frame_{frame_index:04d}_rescale_overlay.png"
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        output_paths.append(out_path)

    csv_path = output_dir / "rescale_overlay_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    return output_paths, csv_path, metrics_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export side-by-side original/rescaled image-mask overlays for visual QA."
    )
    parser.add_argument("--original-image-dir", required=True, type=Path, help="Folder containing original images.")
    parser.add_argument("--original-mask-dir", required=True, type=Path, help="Folder containing original masks.")
    parser.add_argument("--scaled-image-dir", required=True, type=Path, help="Folder containing resized images.")
    parser.add_argument("--scaled-mask-dir", required=True, type=Path, help="Folder containing resized masks.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder for overlay PNGs and CSV summary.")
    parser.add_argument("--image-pattern", default="*.tif", help="Glob for image frames (default: *.tif).")
    parser.add_argument("--mask-pattern", default="mask*.tif", help="Glob for mask frames (default: mask*.tif).")
    parser.add_argument("--scale", default=0.5, type=float, help="Expected spatial scale factor (default: 0.5).")
    parser.add_argument("--alpha", default=0.45, type=float, help="Mask overlay alpha, 0-1 (default: 0.45).")
    parser.add_argument("--dpi", default=150, type=int, help="PNG export DPI (default: 150).")
    parser.add_argument("--max-frames", default=25, type=int, help="Maximum frames to export (default: 25).")
    parser.add_argument("--every", default=1, type=int, help="Export every Nth frame (default: 1).")
    return parser.parse_args()


def main():
    args = parse_args()
    original_image_files = _load_file_list(args.original_image_dir.resolve(), args.image_pattern)
    original_mask_files = _load_file_list(args.original_mask_dir.resolve(), args.mask_pattern)
    scaled_image_files = _load_file_list(args.scaled_image_dir.resolve(), args.image_pattern)
    scaled_mask_files = _load_file_list(args.scaled_mask_dir.resolve(), args.mask_pattern)

    output_paths, csv_path, metrics_rows = export_rescale_overlay_comparisons(
        original_image_files=original_image_files,
        original_mask_files=original_mask_files,
        scaled_image_files=scaled_image_files,
        scaled_mask_files=scaled_mask_files,
        output_dir=args.output_dir.resolve(),
        scale=args.scale,
        alpha=args.alpha,
        dpi=args.dpi,
        max_frames=args.max_frames,
        every=args.every,
    )

    lost_label_frames = [row for row in metrics_rows if row["lost_labels"]]
    print(f"[EXPORT] wrote {len(output_paths)} overlay PNGs to {args.output_dir.resolve()}")
    print(f"[CSV]    wrote {csv_path}")
    if lost_label_frames:
        print(f"[WARN]   {len(lost_label_frames)} exported frames lost one or more labels after rescaling.")
    else:
        print("[OK]     No label IDs were lost in exported frames.")


if __name__ == "__main__":
    main()
