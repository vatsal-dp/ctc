"""
Optimized FILM interpolation script with CUDA support and Async I/O.
"""
import os

# ── Set before any TensorFlow import ──────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import numpy as np
import cv2
import tifffile
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

# ── Configure TensorFlow GPU ───
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow detected {len(gpus)} GPU(s): {gpus}")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  Memory growth enabled for {gpu.name}")
        except RuntimeError as e:
            print(f"  Warning: {e}")
else:
    print("  [NO GPU] FILM will run on CPU.")

import interpolator as interpolator_lib
from tqdm import tqdm


SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
TIFF_EXTENSIONS = ('.tif', '.tiff')
MASK_SUFFIXES = ('_cp_masks.tif', '_cp_masks.tiff')


def is_supported_frame_name(filename):
    lower = filename.lower()
    return (
        lower.endswith(SUPPORTED_IMAGE_EXTENSIONS)
        and not lower.endswith(MASK_SUFFIXES)
    )


def list_frame_names(input_dir):
    return sorted(
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and is_supported_frame_name(f)
    )


def read_raw_image(path):
    """Read PNG/JPEG/TIFF without changing bit depth."""
    ext = Path(path).suffix.lower()
    if ext in TIFF_EXTENSIONS:
        return np.asarray(tifffile.imread(path))

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


class FrameIO:
    """Converts disk frames to FILM tensors and writes tensors back to disk."""

    def __init__(
        self,
        output_ext: Optional[str] = None,
        output_prefix: Optional[str] = None,
        output_digits: Optional[int] = None,
    ):
        self.output_ext = self._normalize_ext(output_ext) if output_ext else None
        self.output_prefix = output_prefix
        self.output_digits = output_digits
        self.dtype = None
        self.dtype_min = 0.0
        self.dtype_max = 1.0
        self.layout = None

    @staticmethod
    def _normalize_ext(ext):
        ext = ext.lower()
        return ext if ext.startswith('.') else f'.{ext}'

    def configure_from(self, input_dir, image_names):
        if not image_names:
            raise RuntimeError(f"No supported image frames found in {input_dir}")

        first_name = image_names[0]
        first_path = os.path.join(input_dir, first_name)
        first = self._standardize_raw_shape(read_raw_image(first_path), first_path)

        if self.dtype is None:
            self.dtype = first.dtype
            if np.issubdtype(self.dtype, np.integer):
                info = np.iinfo(self.dtype)
                self.dtype_min = float(info.min)
                self.dtype_max = float(info.max)
            elif np.issubdtype(self.dtype, np.floating):
                self.dtype_min = 0.0
                self.dtype_max = 1.0
            else:
                raise TypeError(f"Unsupported image dtype {self.dtype} in {first_path}")

            if first.ndim == 2 or (first.ndim == 3 and first.shape[-1] == 1):
                self.layout = 'gray'
            elif first.ndim == 3 and first.shape[-1] >= 3:
                self.layout = 'rgb'
            else:
                raise ValueError(
                    f"Unsupported image shape {first.shape} in {first_path}. "
                    "Expected 2D grayscale or HxWxC color image."
                )

        if self.output_ext is None:
            self.output_ext = self._normalize_ext(Path(first_name).suffix)
        if self.output_prefix is None or self.output_digits is None:
            match = re.match(r'^(.*?)(\d+)$', Path(first_name).stem)
            if self.output_prefix is None:
                self.output_prefix = match.group(1) if match else ''
            if self.output_digits is None:
                self.output_digits = len(match.group(2)) if match else 5

        print(
            f"Frame I/O: dtype={self.dtype}, layout={self.layout}, "
            f"output=*{self.output_ext}, names={self.output_prefix}"
            f"{'0' * self.output_digits}"
        )

    def output_name(self, idx):
        return f"{self.output_prefix}{idx:0{self.output_digits}d}{self.output_ext}"

    def read_frame(self, path):
        raw = self._standardize_raw_shape(read_raw_image(path), path)
        return self._to_model_input(raw, path)

    def write_frame(self, path, image):
        output = self._from_model_output(image)
        ext = Path(path).suffix.lower()
        if ext in TIFF_EXTENSIONS:
            tifffile.imwrite(path, output)
            return

        if output.ndim == 3 and output.shape[-1] == 3:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        elif output.ndim == 3 and output.shape[-1] == 4:
            output = cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA)
        if not cv2.imwrite(path, output):
            raise RuntimeError(f"Could not write image: {path}")

    def _standardize_raw_shape(self, image, path):
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] == 1 and image.shape[-1] not in (1, 3, 4):
            image = image[0]
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Unsupported image shape {image.shape} in {path}. "
                "Expected one 2D frame, not a stack."
            )
        return image

    def _to_model_input(self, image, path):
        if self.dtype is None:
            raise RuntimeError("FrameIO must be configured before reading frames.")
        if image.dtype != self.dtype:
            raise TypeError(
                f"Frame dtype changed from {self.dtype} to {image.dtype}: {path}"
            )

        image = image.astype(np.float32, copy=False)
        scale = self.dtype_max - self.dtype_min
        if scale <= 0:
            raise ValueError(f"Invalid dtype range for {self.dtype}")
        if np.issubdtype(self.dtype, np.integer):
            image = (image - self.dtype_min) / scale

        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0.0, 1.0)

        if image.ndim == 2:
            image = image[..., np.newaxis]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] > 3:
            image = image[..., :3]
        return image.astype(np.float32, copy=False)

    def _from_model_output(self, image):
        image = np.asarray(image, dtype=np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0.0, 1.0)

        if self.layout == 'gray':
            image = image.mean(axis=-1)
        elif self.layout == 'rgb':
            image = image[..., :3]
        else:
            raise RuntimeError("FrameIO must be configured before writing frames.")

        if np.issubdtype(self.dtype, np.integer):
            scale = self.dtype_max - self.dtype_min
            image = image * scale + self.dtype_min
            image = np.rint(image)
            image = np.clip(image, self.dtype_min, self.dtype_max)
        return image.astype(self.dtype, copy=False)


class ImagePairDataset(Dataset):
    """
    Optimized dataset that keeps data in HWC format (Numpy) to avoid
    costly transposes to/from PyTorch CHW format.
    """
    def __init__(self, input_dir, image_names, frame_io):
        self.input_dir = input_dir
        self.image_names = image_names
        self.frame_io = frame_io

    def __len__(self):
        return len(self.image_names) - 1

    def __getitem__(self, idx):
        frame1_path = os.path.join(self.input_dir, self.image_names[idx])
        frame2_path = os.path.join(self.input_dir, self.image_names[idx + 1])
        frame1 = self.frame_io.read_frame(frame1_path)
        frame2 = self.frame_io.read_frame(frame2_path)
        return frame1, frame2, idx


class MemoryDataset(Dataset):
    """Dataset for arrays already in memory."""
    def __init__(self, arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays) - 1

    def __getitem__(self, idx):
        return self.arrays[idx], self.arrays[idx + 1], idx


def run_film_batch(interpolator, frames1_np, frames2_np):
    """
    Run FILM interpolator on a batch of numpy image pairs.
    frames1_np, frames2_np: (B, H, W, C) float32 numpy arrays.
    """
    batch_size = frames1_np.shape[0]
    batch_dt = np.full(shape=(batch_size,), fill_value=0.5, dtype=np.float32)
    return interpolator(frames1_np, frames2_np, batch_dt)


# ── Video I/O ─────────────────────────────────────────────────────────

def video_to_frames(video_path, output_dir):
    """
    Extract all frames from a video file to numbered PNG files.
    Returns the source FPS.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Extracting {total} frames at {fps:.3f}fps ({width}x{height})...")

    idx = 0
    with tqdm(total=total, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f"{output_dir}/{idx:05}.png", frame)
            idx += 1
            pbar.update(1)

    cap.release()
    print(f"Extracted {idx} frames to {output_dir}")
    return fps


def frames_to_video(frames_dir, output_path, fps):
    """
    Reassemble numbered image frames into an mp4 video.
    """
    frames = list_frame_names(frames_dir)
    if not frames:
        raise RuntimeError(f"No image frames found in {frames_dir}")

    first = cv2.imread(os.path.join(frames_dir, frames[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Could not read first video frame: {frames[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    print(f"Assembling {len(frames)} frames at {fps:.3f}fps -> {output_path}")
    for fname in tqdm(frames, desc="Writing video"):
        frame = cv2.imread(os.path.join(frames_dir, fname), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"Could not read video frame: {fname}")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        out.write(frame)

    out.release()
    print(f"Saved: {output_path}")


# ── Core interpolation pipeline ───────────────────────────────────────

def interpolate_from_disk(
    input_dir, output_dir, interpolator, batch_size, num_workers, write_threads,
    frame_io
):
    """
    Pipeline: Disk Read (Parallel) -> GPU Interpolate -> Disk Write (Async Background)
    """
    image_names = list_frame_names(input_dir)
    if len(image_names) < 2:
        return
    frame_io.configure_from(input_dir, image_names)

    os.makedirs(output_dir, exist_ok=True)

    dataset = ImagePairDataset(input_dir, image_names, frame_io)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Async writer pool
    write_executor = ThreadPoolExecutor(max_workers=write_threads)
    futures = []

    def save_pair(idx_val, img1, img_mid, out_d):
        """Helper to write files in background thread."""
        img_count = idx_val * 2
        frame_io.write_frame(os.path.join(out_d, frame_io.output_name(img_count)), img1)
        frame_io.write_frame(
            os.path.join(out_d, frame_io.output_name(img_count + 1)), img_mid
        )

    print(f"Processing {len(dataset)} pairs (Disk Mode)...")

    for batch_idx, (f1, f2, indices) in enumerate(tqdm(dataloader, desc="Interpolating")):
        f1_np = f1.numpy().astype(np.float32)
        f2_np = f2.numpy().astype(np.float32)

        # GPU Inference
        mid_frames = run_film_batch(interpolator, f1_np, f2_np)

        # Offload writing to background threads so GPU doesn't wait
        for i, idx in enumerate(indices):
            idx_val = idx.item()
            f = write_executor.submit(
                save_pair,
                idx_val,
                f1_np[i],
                mid_frames[i],
                output_dir
            )
            futures.append(f)

        # Clean up finished futures to prevent RAM leak
        done_futures = [f for f in futures if f.done()]
        for f in done_futures:
            f.result()
        futures = [f for f in futures if not f.done()]

    # Wait for all writes to finish
    print("\nWaiting for pending writes to finish...")
    write_executor.shutdown(wait=True)
    for f in futures:
        f.result()

    # Write last frame
    last_img = frame_io.read_frame(os.path.join(input_dir, image_names[-1]))
    img_count = (len(image_names) - 1) * 2
    frame_io.write_frame(os.path.join(output_dir, frame_io.output_name(img_count)), last_img)


def load_images_to_memory_parallel(input_dir, frame_io, num_threads=16):
    """
    Optimized: Reads images in parallel using ThreadPool.
    Returns list of numpy arrays (H, W, C).
    """
    print("Loading images into memory (Parallel)...")
    image_names = list_frame_names(input_dir)
    if not image_names:
        raise RuntimeError(f"No supported image frames found in {input_dir}")
    frame_io.configure_from(input_dir, image_names)

    images = [None] * len(image_names)

    def load_one(args):
        i, fname = args
        path = os.path.join(input_dir, fname)
        return i, frame_io.read_frame(path)

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        tasks = [(i, name) for i, name in enumerate(image_names)]
        for i, img in ex.map(load_one, tasks):
            images[i] = img
            if (i + 1) % 50 == 0:
                print(f"Loaded {i+1}/{len(image_names)}", end='\r')

    print(f"\nLoaded {len(images)} images.")
    return images


def interpolate_from_memory(image_arrays, interpolator, batch_size):
    """
    Memory mode: Image arrays are (H, W, C) float32 numpy arrays.
    """
    n_pairs = len(image_arrays) - 1
    result_arrays = []

    for start in tqdm(range(0, n_pairs, batch_size), desc="Interpolating"):
        end = min(start + batch_size, n_pairs)

        # Stack into (B, H, W, C) float32 — pure numpy, no PyTorch collation
        f1_np = np.stack(
            [image_arrays[i].astype(np.float32) for i in range(start, end)], axis=0
        )
        f2_np = np.stack(
            [image_arrays[i + 1].astype(np.float32) for i in range(start, end)], axis=0
        )

        mid_frames = run_film_batch(interpolator, f1_np, f2_np)

        for i in range(end - start):
            result_arrays.append(f1_np[i])
            result_arrays.append(mid_frames[i])

    print()
    result_arrays.append(image_arrays[-1])
    return result_arrays


def save_arrays_parallel(arrays, output_dir, frame_io, num_threads):
    """Save HWC numpy arrays to disk."""
    print(f"Saving {len(arrays)} images with {num_threads} threads...")
    os.makedirs(output_dir, exist_ok=True)

    def write_single(idx, arr):
        frame_io.write_frame(os.path.join(output_dir, frame_io.output_name(idx)), arr)

    futures = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i, arr in enumerate(tqdm(arrays, desc="Saving")):
            futures.append(executor.submit(write_single, i, arr))
        for f in futures:
            f.result()

    print(f"\nSaved to {output_dir}")


def select_tensorflow_gpu(gpu_idx):
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return
    if gpu_idx >= len(gpus):
        gpu_idx = 0
    try:
        tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
    except Exception as e:
        print(f"TF GPU Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory of PNG/JPEG/TIFF frames (mutually exclusive with --video)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for interpolated image frames')
    parser.add_argument('--model_path', type=str, default='C:/Users/oargell.lab/Desktop/testets/pretrained_models/film_net/Style/saved_model')
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--write_threads', type=int, default=10)
    parser.add_argument('--memory_mode', action='store_true')
    parser.add_argument('--output_ext', type=str, default=None,
                        help='Output extension, e.g. tif or png. Default: same as first input frame.')
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Output filename prefix. Default: inferred from first input frame, e.g. t0000 -> t.')
    parser.add_argument('--output_digits', type=int, default=None,
                        help='Zero-padding width for output frame numbers. Default: inferred from first input frame.')

    # ── Video args ────────────────────────────────────────────────────
    parser.add_argument('--video', type=str, default=None,
                        help='Input video file (.mp4, .avi, etc.) Instead of --input_dir')
    parser.add_argument('--video_out', type=str, default=None,
                        help='Output video file path (e.g. output.mp4). '
                             'If omitted, only interpolated frames are saved to --output_dir')
    parser.add_argument('--keep_frames', action='store_true',
                        help='Keep intermediate PNG frames after video assembly')
    parser.add_argument('--slow_motion', action='store_true',
                    help='Output at original FPS making video longer (true slow motion). '
                         'Default is to multiply FPS keeping same duration.')

    args = parser.parse_args()

    # ── Validate args ─────────────────────────────────────────────────
    if args.video and args.input_dir:
        parser.error("Use either --video or --input_dir, not both.")
    if not args.video and not args.input_dir:
        parser.error("Either --video or --input_dir is required.")

    select_tensorflow_gpu(args.gpu_id)
    frame_io = FrameIO(
        output_ext=args.output_ext,
        output_prefix=args.output_prefix,
        output_digits=args.output_digits,
    )

    print(f"Loading Model: {args.model_path}")
    interp = interpolator_lib.Interpolator(args.model_path, align=64, block_shape=[1, 1])

    # ── Video input: extract frames first ────────────────────────────
    source_fps = None
    video_frames_dir = os.path.join(SCRIPT_DIR, "temp_video_frames")

    if args.video:
        if os.path.exists(video_frames_dir):
            shutil.rmtree(video_frames_dir)
        source_fps = video_to_frames(args.video, video_frames_dir)
        args.input_dir = video_frames_dir

    # ── Run interpolation (image or video frames) ─────────────────────
    if args.memory_mode:
        # ── MEMORY MODE ───────────────────────────────────────────────
        images = load_images_to_memory_parallel(args.input_dir, frame_io, args.write_threads)

        for c in range(args.cycles):
            print(f"Cycle {c+1}/{args.cycles}")
            images = interpolate_from_memory(images, interp, args.batch_size)

        save_arrays_parallel(images, args.output_dir, frame_io, args.write_threads)

    else:
        # ── DISK MODE ─────────────────────────────────────────────────
        current_in = args.input_dir
        final_out = args.output_dir

        for c in range(args.cycles):
            is_last = (c == args.cycles - 1)

            if is_last:
                out = final_out
            else:
                out = os.path.join(SCRIPT_DIR, f"temp_interpolation_{c+1}")

            # Clean temp if exists
            if not is_last and os.path.exists(out):
                shutil.rmtree(out)
            os.makedirs(out, exist_ok=True)

            print(f"\n=== Cycle {c+1}/{args.cycles} ===")
            interpolate_from_disk(
                current_in, out, interp,
                args.batch_size, args.num_workers, args.write_threads,
                frame_io
            )

            # Delete previous temp (not original input or extracted video frames)
            if c > 0 and current_in != args.input_dir:
                shutil.rmtree(current_in)

            current_in = out

    # ── Video output: reassemble frames ───────────────────────────────
    if args.video and args.video_out:
        # Each cycle doubles the frame count, so multiply fps accordingly
        if args.slow_motion:
            new_fps = source_fps  # same fps, more frames = longer video
            print(f"\nSlow motion mode: FPS unchanged at {source_fps:.3f}, duration x{2**args.cycles}")
        else:
            new_fps = source_fps * (2 ** args.cycles)  # same duration, higher fps
            print(f"\nOriginal FPS: {source_fps:.3f} -> Output FPS: {new_fps:.3f}")

        frames_to_video(args.output_dir, args.video_out, new_fps)

        if not args.keep_frames:
            print("Cleaning up extracted frames...")
            shutil.rmtree(video_frames_dir)

    elif args.video and not args.video_out:
        print("\n[INFO] No --video_out specified. Interpolated frames saved to --output_dir only.")
        if not args.keep_frames:
            shutil.rmtree(video_frames_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

