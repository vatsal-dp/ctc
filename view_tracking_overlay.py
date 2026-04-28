#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import hsv_to_rgb
from matplotlib.figure import Figure
from matplotlib.widgets import Slider


GOLDEN_RATIO_CONJUGATE = 0.618033988749895


@dataclass(frozen=True)
class TrackRow:
    label: int
    begin: int
    end: int
    parent: int


@dataclass
class LineageLayout:
    y_positions: dict[int, float]
    children: dict[int, list[int]]
    roots: list[int]
    max_frame: int
    max_leaf_row: int


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


def _track_color(label_value: int):
    hue = (int(label_value) * GOLDEN_RATIO_CONJUGATE) % 1.0
    return np.asarray(hsv_to_rgb([hue, 0.85, 1.0]), dtype=np.float32)


def _build_track_color_map(track_rows: dict[int, TrackRow]):
    return {track_id: _track_color(track_id) for track_id in sorted(track_rows)}


def _label_overlay(mask: np.ndarray, alpha: float, color_map: dict[int, np.ndarray] | None = None):
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    for label_value in labels.tolist():
        rgb = _track_color(label_value)
        if color_map is not None:
            rgb = np.asarray(color_map.get(label_value, rgb), dtype=np.float32)
        pix = mask == label_value
        overlay[pix, :3] = rgb
        overlay[pix, 3] = alpha
    return overlay


def _load_frame_pair(image_path: Path, mask_path: Path, frame_index: int):
    image = _read_image(image_path)
    mask = _read_mask(mask_path)

    image_h, image_w = image.shape[:2]
    if mask.shape[0] != image_h or mask.shape[1] != image_w:
        raise ValueError(
            f"Shape mismatch at frame {frame_index}: image={image.shape}, mask={mask.shape}. "
            "Masks and images must have identical width/height."
        )
    return image, mask


def _parse_track_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Track file does not exist: {path}")

    rows: dict[int, TrackRow] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Track file line {line_number} must contain at least 4 integers: {line}")
            label, begin, end, parent = [int(part) for part in parts[:4]]
            if label in rows:
                raise ValueError(f"Duplicate track ID {label} in {path}")
            if begin < 0 or end < begin:
                raise ValueError(f"Invalid frame range for track {label}: begin={begin}, end={end}")
            rows[label] = TrackRow(label=label, begin=begin, end=end, parent=parent)

    for track_id, row in rows.items():
        if row.parent == track_id:
            raise ValueError(f"Track {track_id} cannot reference itself as parent.")
        if row.parent != 0 and row.parent not in rows:
            raise ValueError(f"Track {track_id} references missing parent track {row.parent}.")

    return rows


def _build_lineage_layout(track_rows: dict[int, TrackRow]):
    if not track_rows:
        return None

    children: dict[int, list[int]] = defaultdict(list)
    for row in track_rows.values():
        children[row.parent].append(row.label)

    def child_sort_key(track_id: int):
        row = track_rows[track_id]
        return (row.begin, track_id)

    for parent_id in list(children):
        children[parent_id] = sorted(children[parent_id], key=child_sort_key)

    roots = children.get(0, [])
    y_positions: dict[int, float] = {}
    next_leaf_row = 0

    def assign(track_id: int):
        nonlocal next_leaf_row
        child_ids = children.get(track_id, [])
        if not child_ids:
            y_value = float(next_leaf_row)
            next_leaf_row += 1
        else:
            child_positions = [assign(child_id) for child_id in child_ids]
            y_value = float(sum(child_positions) / len(child_positions))
        y_positions[track_id] = y_value
        return y_value

    for root_id in roots:
        if root_id not in y_positions:
            assign(root_id)

    for track_id in sorted(track_rows):
        if track_id not in y_positions:
            assign(track_id)

    return LineageLayout(
        y_positions=y_positions,
        children={track_id: children.get(track_id, []) for track_id in track_rows},
        roots=roots,
        max_frame=max(row.end for row in track_rows.values()),
        max_leaf_row=max(next_leaf_row - 1, 0),
    )


def _lineage_focus_bounds(current_frame: int, max_frame: int, lineage_window: int | None):
    if lineage_window is None:
        return 0, max_frame
    return max(0, current_frame - lineage_window), min(max_frame, current_frame + lineage_window)


def _track_overlaps_range(row: TrackRow, start_frame: int, end_frame: int):
    return row.begin <= end_frame and row.end >= start_frame


def _filter_lineage_track_rows(track_rows: dict[int, TrackRow], start_frame: int, end_frame: int):
    return {
        track_id: row
        for track_id, row in track_rows.items()
        if _track_overlaps_range(row, start_frame, end_frame)
    }


def _lineage_plot_segments(
    track_rows: dict[int, TrackRow],
    lineage_layout: LineageLayout | None,
    current_frame: int,
    x_start: int | None = None,
    x_end: int | None = None,
    reveal_until_frame: int | None = None,
):
    if not track_rows or lineage_layout is None:
        return {"tracks": [], "connectors": [], "active": []}

    reveal_until = current_frame if reveal_until_frame is None else reveal_until_frame
    track_segments = []
    connector_segments = []
    active_points = []

    for track_id, row in sorted(track_rows.items(), key=lambda item: (item[1].begin, item[0])):
        if reveal_until < row.begin:
            continue

        y_value = lineage_layout.y_positions[track_id]
        segment_start = row.begin if x_start is None else max(row.begin, x_start)
        segment_end = min(row.end, reveal_until)
        if x_end is not None:
            segment_end = min(segment_end, x_end)

        if segment_end < segment_start:
            continue

        track_segments.append(
            {
                "track_id": track_id,
                "x0": segment_start,
                "x1": segment_end,
                "y": y_value,
            }
        )

        current_in_view = (
            row.begin <= current_frame <= row.end
            and (x_start is None or current_frame >= x_start)
            and (x_end is None or current_frame <= x_end)
        )
        if current_in_view:
            active_points.append({"track_id": track_id, "x": current_frame, "y": y_value})

        if row.parent != 0 and row.parent in track_rows and reveal_until >= row.begin:
            parent_row = track_rows[row.parent]
            connector_x0 = min(parent_row.end, row.begin)
            connector_x1 = row.begin
            if x_start is not None and max(connector_x0, connector_x1) < x_start:
                continue
            if x_end is not None and min(connector_x0, connector_x1) > x_end:
                continue
            connector_segments.append(
                {
                    "parent_id": row.parent,
                    "child_id": track_id,
                    "x0": connector_x0,
                    "y0": lineage_layout.y_positions[row.parent],
                    "x1": connector_x1,
                    "y1": y_value,
                }
            )

    return {"tracks": track_segments, "connectors": connector_segments, "active": active_points}


def _resolve_track_file(track_file_arg: Path | None, mask_dir: Path):
    if track_file_arg is not None:
        track_file = track_file_arg.expanduser().resolve()
        if not track_file.is_file():
            raise FileNotFoundError(f"track-file does not exist: {track_file}")
        return track_file

    default_track_file = (mask_dir / "res_track.txt").resolve()
    if default_track_file.is_file():
        return default_track_file
    return None


def _validate_track_rows_against_frame_count(track_rows: dict[int, TrackRow], frame_count: int):
    if not track_rows:
        return

    if frame_count <= 0:
        raise ValueError("No frames available for lineage rendering.")

    max_frame = frame_count - 1
    for track_id, row in sorted(track_rows.items()):
        if row.begin > max_frame or row.end > max_frame:
            raise ValueError(
                f"Track {track_id} uses frame range {row.begin}-{row.end}, but available frames end at {max_frame}."
            )


def _validate_masks_against_tracks(mask_files, track_rows: dict[int, TrackRow]):
    observed_track_ids: set[int] = set()

    for frame_index, mask_path in enumerate(mask_files):
        mask = _read_mask(mask_path)
        labels = np.unique(mask)
        labels = labels[labels != 0].astype(int).tolist()

        for label_value in labels:
            observed_track_ids.add(label_value)
            row = track_rows.get(label_value)
            if row is None:
                raise ValueError(
                    f"Frame {frame_index} mask {mask_path.name} contains label {label_value}, "
                    "but that track is missing from the lineage file."
                )
            if not (row.begin <= frame_index <= row.end):
                raise ValueError(
                    f"Frame {frame_index} mask {mask_path.name} contains label {label_value}, "
                    f"but res_track.txt says it is only active in frames {row.begin}-{row.end}."
                )

    missing_track_ids = sorted(set(track_rows) - observed_track_ids)
    if missing_track_ids:
        preview = ", ".join(str(track_id) for track_id in missing_track_ids[:10])
        suffix = "..." if len(missing_track_ids) > 10 else ""
        raise ValueError(f"Track file contains IDs that never appear in the masks: {preview}{suffix}")


def _create_figure(interactive: bool, has_lineage: bool):
    ncols = 2 if has_lineage else 1
    width_ratios = [1.2, 1.0] if has_lineage else None
    figsize = (16, 8) if has_lineage else (10, 8)
    gridspec_kw = {"width_ratios": width_ratios} if width_ratios is not None else None

    if interactive:
        fig, axes = plt.subplots(1, ncols, figsize=figsize, gridspec_kw=gridspec_kw)
    else:
        fig = Figure(figsize=figsize)
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, ncols, gridspec_kw=gridspec_kw)

    if has_lineage:
        overlay_ax, lineage_ax = axes
    else:
        overlay_ax = axes
        lineage_ax = None
    return fig, overlay_ax, lineage_ax


class OverlayLineageRenderer:
    def __init__(
        self,
        image_files,
        mask_files,
        alpha: float,
        track_rows: dict[int, TrackRow] | None = None,
        lineage_window: int | None = None,
        lineage_active_only: bool = False,
    ):
        self.image_files = image_files
        self.mask_files = mask_files
        self.alpha = alpha
        self.track_rows = track_rows or {}
        self.lineage_window = lineage_window
        self.lineage_active_only = lineage_active_only
        self.frame_count = min(len(image_files), len(mask_files))
        self.has_lineage = bool(self.track_rows)
        self.color_map = _build_track_color_map(self.track_rows)
        self.lineage_layout = _build_lineage_layout(self.track_rows) if self.has_lineage else None

        lineage_max_frame = self.lineage_layout.max_frame if self.lineage_layout is not None else -1
        self.max_frame = max(self.frame_count - 1, lineage_max_frame)

    def draw_frame(self, frame_index: int, overlay_ax, lineage_ax=None):
        image_path = self.image_files[frame_index]
        mask_path = self.mask_files[frame_index]
        image, mask = _load_frame_pair(image_path, mask_path, frame_index)

        overlay_ax.clear()
        if image.ndim == 2:
            overlay_ax.imshow(image, cmap="gray")
        else:
            overlay_ax.imshow(image)
        overlay_ax.imshow(_label_overlay(mask, self.alpha, self.color_map))
        overlay_ax.set_axis_off()
        overlay_ax.set_title(
            f"Frame {frame_index + 1}/{self.frame_count}\n"
            f"Image: {image_path.name} | Mask: {mask_path.name}"
        )

        if lineage_ax is not None:
            self._draw_lineage(frame_index, lineage_ax)

    def _draw_lineage(self, frame_index: int, lineage_ax):
        lineage_ax.clear()
        title_bits = ["Lineage"]
        if self.lineage_window is not None:
            title_bits.append(f"+/- {self.lineage_window} frames")
        if self.lineage_active_only:
            title_bits.append("active in view")
        lineage_ax.set_title(" | ".join(title_bits))

        if not self.has_lineage or self.lineage_layout is None:
            lineage_ax.text(
                0.5,
                0.5,
                "No lineage file available",
                ha="center",
                va="center",
                transform=lineage_ax.transAxes,
            )
            lineage_ax.set_axis_off()
            return

        focus_start, focus_end = _lineage_focus_bounds(
            current_frame=frame_index,
            max_frame=self.max_frame,
            lineage_window=self.lineage_window,
        )
        track_rows = self.track_rows
        lineage_layout = self.lineage_layout

        if self.lineage_active_only:
            track_rows = _filter_lineage_track_rows(track_rows, focus_start, focus_end)
            lineage_layout = _build_lineage_layout(track_rows)
            if not track_rows or lineage_layout is None:
                lineage_ax.text(
                    0.5,
                    0.5,
                    "No tracks in selected lineage view",
                    ha="center",
                    va="center",
                    transform=lineage_ax.transAxes,
                )
                lineage_ax.set_axis_off()
                return

        x_start = focus_start if self.lineage_window is not None else None
        x_end = focus_end if self.lineage_window is not None else None
        reveal_until_frame = focus_end if self.lineage_window is not None else frame_index

        plot_data = _lineage_plot_segments(
            track_rows,
            lineage_layout,
            current_frame=frame_index,
            x_start=x_start,
            x_end=x_end,
            reveal_until_frame=reveal_until_frame,
        )

        for connector in plot_data["connectors"]:
            child_color = self.color_map.get(connector["child_id"], _track_color(connector["child_id"]))
            lineage_ax.plot(
                [connector["x0"], connector["x1"]],
                [connector["y0"], connector["y1"]],
                color=child_color,
                linewidth=1.8,
                alpha=0.9,
                zorder=1,
            )

        for segment in plot_data["tracks"]:
            track_color = self.color_map.get(segment["track_id"], _track_color(segment["track_id"]))
            lineage_ax.plot(
                [segment["x0"], segment["x1"]],
                [segment["y"], segment["y"]],
                color=track_color,
                linewidth=3.0,
                solid_capstyle="round",
                zorder=2,
            )

        if plot_data["active"]:
            lineage_ax.scatter(
                [point["x"] for point in plot_data["active"]],
                [point["y"] for point in plot_data["active"]],
                s=26,
                c=[self.color_map.get(point["track_id"], _track_color(point["track_id"])) for point in plot_data["active"]],
                edgecolors="black",
                linewidths=0.35,
                zorder=3,
            )

        lineage_ax.axvline(frame_index, color="0.2", linewidth=1.0, linestyle="--", alpha=0.45, zorder=0)
        if self.lineage_window is None:
            lineage_ax.set_xlim(-0.5, self.max_frame + 0.5)
        else:
            lineage_ax.set_xlim(focus_start - 0.5, focus_end + 0.5)
        lineage_ax.set_ylim(lineage_layout.max_leaf_row + 0.6, -0.6)
        lineage_ax.set_xlabel("Frame")
        lineage_ax.set_yticks([])
        lineage_ax.grid(axis="x", alpha=0.2, linewidth=0.5)
        lineage_ax.spines["right"].set_visible(False)
        lineage_ax.spines["top"].set_visible(False)


class OverlayViewer:
    def __init__(self, renderer: OverlayLineageRenderer, start_index: int):
        self.renderer = renderer
        self.index = max(0, min(start_index, self.renderer.frame_count - 1))
        self.frame_slider = None

        self.fig, self.overlay_ax, self.lineage_ax = _create_figure(
            interactive=True,
            has_lineage=self.renderer.has_lineage,
        )
        self.fig.subplots_adjust(bottom=0.18, wspace=0.08)

        if self.renderer.frame_count > 1:
            slider_ax = self.fig.add_axes([0.16, 0.06, 0.68, 0.05])
            self.frame_slider = Slider(
                ax=slider_ax,
                label="Frame",
                valmin=1,
                valmax=self.renderer.frame_count,
                valinit=self.index + 1,
                valstep=1,
            )
            self.frame_slider.on_changed(self._on_slider_changed)

        self.fig.canvas.mpl_connect("key_press_event", self._on_keypress)
        self._draw_frame()

    def _set_index(self, new_index: int, sync_slider: bool):
        clamped_index = max(0, min(new_index, self.renderer.frame_count - 1))
        if clamped_index == self.index:
            return

        self.index = clamped_index
        if sync_slider and self.frame_slider is not None:
            self.frame_slider.set_val(self.index + 1)
            return

        self._draw_frame()

    def _on_slider_changed(self, value):
        self._set_index(int(value) - 1, sync_slider=False)

    def _on_keypress(self, event):
        if event.key in {"left", "a"}:
            self._set_index(self.index - 1, sync_slider=True)
        elif event.key in {"right", "d"}:
            self._set_index(self.index + 1, sync_slider=True)

    def _draw_frame(self):
        self.renderer.draw_frame(self.index, self.overlay_ax, self.lineage_ax)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def export_overlay_lineage_frames(
    image_files,
    mask_files,
    alpha: float,
    export_dir: Path,
    track_rows: dict[int, TrackRow],
    dpi: int = 150,
    lineage_window: int | None = None,
    lineage_active_only: bool = False,
):
    if not track_rows:
        raise ValueError("Export mode requires a lineage file with track rows.")
    if len(image_files) != len(mask_files):
        raise ValueError(
            "Export mode requires the same number of image and mask files so the side-by-side frames line up."
        )

    _validate_track_rows_against_frame_count(track_rows, len(mask_files))
    _validate_masks_against_tracks(mask_files, track_rows)

    export_dir = export_dir.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    renderer = OverlayLineageRenderer(
        image_files=image_files,
        mask_files=mask_files,
        alpha=alpha,
        track_rows=track_rows,
        lineage_window=lineage_window,
        lineage_active_only=lineage_active_only,
    )
    fig, overlay_ax, lineage_ax = _create_figure(interactive=False, has_lineage=True)
    fig.subplots_adjust(bottom=0.1, wspace=0.08)

    output_paths = []
    for frame_index in range(renderer.frame_count):
        renderer.draw_frame(frame_index, overlay_ax, lineage_ax)
        out_path = export_dir / f"{image_files[frame_index].stem}_overlay_lineage.png"
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        output_paths.append(out_path)

    return output_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive viewer and PNG exporter for tracked-mask overlays plus lineage."
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
        "--track-file",
        default=None,
        type=Path,
        help="Optional path to res_track.txt (default: <mask-dir>/res_track.txt if present).",
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
        "--export-dir",
        default=None,
        type=Path,
        help="If set, write one combined PNG per frame to this directory instead of opening the viewer.",
    )
    parser.add_argument(
        "--dpi",
        default=150,
        type=int,
        help="PNG export DPI used with --export-dir (default: 150).",
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
        help="0-based initial frame index for interactive viewing (default: 0).",
    )
    parser.add_argument(
        "--lineage-window",
        default=None,
        type=int,
        help="Show lineage within this many frames before/after the current frame.",
    )
    parser.add_argument(
        "--lineage-active-only",
        action="store_true",
        help="Only draw tracks whose lifetime overlaps the current lineage view.",
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
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if args.lineage_window is not None and args.lineage_window < 0:
        raise ValueError("--lineage-window must be zero or positive.")

    image_files = _load_file_list(
        image_dir,
        args.image_pattern,
        exclude_substrings=args.image_exclude_substring,
    )
    mask_files = _load_file_list(mask_dir, args.mask_pattern)

    track_file = _resolve_track_file(args.track_file, mask_dir)
    track_rows = _parse_track_file(track_file) if track_file is not None else {}

    if args.export_dir is not None:
        if len(image_files) != len(mask_files):
            raise ValueError(
                "Export mode requires exact image/mask count parity. "
                f"images={len(image_files)}, masks={len(mask_files)}."
            )
        if track_file is None:
            raise FileNotFoundError(
                "Export mode requires a lineage file. Pass --track-file or place res_track.txt in the mask directory."
            )

        output_paths = export_overlay_lineage_frames(
            image_files=image_files,
            mask_files=mask_files,
            alpha=args.alpha,
            export_dir=args.export_dir,
            track_rows=track_rows,
            dpi=args.dpi,
            lineage_window=args.lineage_window,
            lineage_active_only=args.lineage_active_only,
        )
        print(f"[EXPORT] wrote {len(output_paths)} frames to {args.export_dir.resolve()}")
        if output_paths:
            print(f"[EXPORT] first={output_paths[0].name}")
            print(f"[EXPORT] last={output_paths[-1].name}")
        return

    if len(image_files) != len(mask_files):
        print(
            "[WARN] Image/mask count mismatch. "
            f"images={len(image_files)}, masks={len(mask_files)}. "
            f"Using first {min(len(image_files), len(mask_files))} pairs by sorted order."
        )

    renderer = OverlayLineageRenderer(
        image_files=image_files,
        mask_files=mask_files,
        alpha=args.alpha,
        track_rows=track_rows,
        lineage_window=args.lineage_window,
        lineage_active_only=args.lineage_active_only,
    )
    viewer = OverlayViewer(renderer=renderer, start_index=args.start_index)
    viewer.show()


if __name__ == "__main__":
    main()
