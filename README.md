# Cell Tracking Challenge Evaluation

This repository contains helper scripts for producing, validating, and
evaluating Cell Tracking Challenge (CTC) tracking results. The main
evaluation path is:

1. Put your predicted tracks in CTC result format.
2. Validate the result folder with `validate_ctc_result_format.py`.
3. Run the official CTC metrics through `evaluate_ctc_results.py`.

The official evaluation binaries are already included under
`EvaluationSoftware/` for macOS, Linux, and Windows.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "numpy<2" tifffile imagecodecs
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "numpy<2" tifffile imagecodecs
```

That is enough for validation and evaluation. To run the full interpolation,
segmentation, and tracking workflow, use Python 3.10 and install the full
project dependencies instead:

```bash
python -m pip install -r requirements.txt
python -m pip install imagecodecs
```

For standalone RAM tip tracking without running FILM or Cellpose from this
repo, the tracking script needs:

```bash
python -m pip install "numpy<2" scipy scikit-image tifffile imagecodecs
```

`imagecodecs` is needed when TIFF files are LZW-compressed. Without it,
`tifffile` may fail with a message like `COMPRESSION.LZW requires the
imagecodecs package`.

If an official binary is not executable on macOS or Linux, run:

```bash
chmod +x EvaluationSoftware/Mac/*Measure EvaluationSoftware/Linux/*Measure
```

## Command Shell Notes

Most multi-line commands below use Bash/zsh syntax, where a trailing `\`
continues the command on the next line. That works in macOS/Linux shells and
Git Bash, but not in Windows PowerShell.

In PowerShell, either put the command on one line or replace each trailing `\`
with a trailing backtick. The backtick must be the last character on the line,
with no spaces after it:

```powershell
python evaluate_ctc_results.py `
  --dataset-root C:\path\to\results\BF-C2DL-HSC `
  --source-root C:\path\to\source\BF-C2DL-HSC `
  --sequence 01 `
  --digits auto `
  --ctc-software-dir .\EvaluationSoftware `
  --metrics TRA SEG DET
```

In `cmd.exe`, the equivalent line-continuation character is `^`.

## Repository Layout

The root-level commands are kept for compatibility. For example, these still
work from the repository root:

```bash
python evaluate_ctc_results.py --help
python validate_ctc_result_format.py --help
python ram_run_tiptracking_standalone_optimized.py --help
bash run_ctc_full_workflow.sh --help
```

The implementation lives under `src/ctc_tracking/`:

```text
src/ctc_tracking/
  analysis/        Failure reports and tracking diagnostics.
  evaluation/      CTC format validation and official metric wrappers.
  interpolation/   FILM interpolation helpers.
  preprocessing/   Dataset subsetting, resizing, and segmentation checks.
  tracking/        Tip tracking, blockwise tracking, CTC export, downsampling.
  visualization/   Overlay and lineage visual QA tools.
  workflows/       Python orchestration pipeline.
tests/             Unit and workflow tests.
scripts/           Reserved for future small one-off operational scripts.
```

## Expected CTC Layout

The evaluation scripts expect a CTC-style result root that contains one result
folder per sequence:

```text
/path/to/results/BF-C2DL-HSC/
  01_RES/
    mask000.tif
    mask001.tif
    mask002.tif
    res_track.txt
  02_RES/
    mask000.tif
    mask001.tif
    res_track.txt
```

Ground truth can either be beside the result folders:

```text
/path/to/results/BF-C2DL-HSC/
  01_GT/
    TRA/
      man_track000.tif
      man_track001.tif
      man_track.txt
    SEG/
      man_seg000.tif
      man_seg001.tif
```

or in the original dataset root:

```text
/path/to/source/BF-C2DL-HSC/
  01/
    t000.tif
    t001.tif
  01_GT/
    TRA/
    SEG/
```

If ground truth is not beside the results, pass the original dataset root with
`--source-root`.

Result masks must be named with a fixed-width frame number, such as
`mask000.tif` or `mask00000.tif`. They must be contiguous, integer-labeled,
and preferably `uint16`. Label `0` is background. Positive labels must match
the labels listed in `res_track.txt`.

`res_track.txt` has one row per track:

```text
L B E P
```

where `L` is the track label, `B` is the first frame, `E` is the last frame,
and `P` is the parent track label. Use parent `0` when the track has no parent.

## Track Masks With RAM Tip Tracking

Use `ram_run_tiptracking_standalone_optimized.py` when you already have one
instance-segmentation mask TIFF per frame and need to convert those per-frame
masks into CTC tracking output. The input masks should contain `0` for
background and positive integer object labels. They do not need stable IDs
across frames; the tracker writes stable track labels, `mask*.tif`, and
`res_track.txt`.

For masks that are already on the original CTC timeline, write the tracker
output directly as a CTC result folder:

```bash
python ram_run_tiptracking_standalone_optimized.py \
  --mask-dir /path/to/segmentation_masks/01 \
  --mask-pattern "*.tif" \
  --output-dir /path/to/results/BF-C2DL-HSC \
  --position 01 \
  --time-series-threshold 1 \
  --output-digits auto \
  --stack-storage ram
```

This writes:

```text
/path/to/results/BF-C2DL-HSC/
  01_tracking_input_manifest.txt
  01_RES/
    mask000.tif
    mask001.tif
    res_track.txt
```

If the masks came from Cellpose, the useful mask pattern is often
`"*_cp_masks.tif"`:

```bash
python ram_run_tiptracking_standalone_optimized.py \
  --mask-dir /path/to/cellpose_output/01/masks \
  --mask-pattern "*_cp_masks.tif" \
  --output-dir /path/to/results/BF-C2DL-HSC \
  --position 01
```

For masks on an interpolated timeline, use `--export-mode final-only` so the
script samples the tracked stack back to the original CTC frames and writes the
final result folder directly:

```bash
python ram_run_tiptracking_standalone_optimized.py \
  --mask-dir /path/to/work/01/images_interpolated_2x_tif/masks \
  --mask-pattern "*_cp_masks.tif" \
  --output-dir /path/to/work/01/tracking \
  --position 01_interp \
  --export-mode final-only \
  --final-output-dir /path/to/results/BF-C2DL-HSC/01_RES \
  --source-root /path/to/source/BF-C2DL-HSC \
  --sequence 01 \
  --temporal-downsample-factor 2 \
  --temporal-downsample-offset 0 \
  --final-output-digits auto \
  --time-series-threshold 1 \
  --identity-rescue-gap 1 \
  --rescue-confidence-threshold 0.50 \
  --max-centroid-dist-px 50 \
  --gap-fill-frames 1
```

In `final-only` mode, `--output-dir` is scratch/manifest space, while
`--final-output-dir` is the CTC folder that will be validated and evaluated.
Use `--source-frame-count N` instead of `--source-root` only when the original
source frames are unavailable. Evaluation still needs ground truth either
beside the result folder or available through `--source-root`.

After tracking, validate and evaluate the result:

```bash
python validate_ctc_result_format.py \
  --dataset-root /path/to/results/BF-C2DL-HSC \
  --source-root /path/to/source/BF-C2DL-HSC \
  --sequence 01 \
  --digits auto

python evaluate_ctc_results.py \
  --dataset-root /path/to/results/BF-C2DL-HSC \
  --source-root /path/to/source/BF-C2DL-HSC \
  --sequence 01 \
  --digits auto \
  --ctc-software-dir ./EvaluationSoftware \
  --metrics TRA SEG DET
```

For large movies, the default `--stack-storage ram` is fastest when enough
memory is available. If RAM is not enough, switch to a disk-backed stack:

```bash
--stack-storage mmap --mmap-dir /path/to/fast/local/scratch
```

## Validate A Result Folder

Run validation before official scoring:

```bash
python validate_ctc_result_format.py \
  --dataset-root /path/to/results/BF-C2DL-HSC \
  --source-root /path/to/source/BF-C2DL-HSC \
  --sequence 01 \
  --digits auto
```

You can omit `--source-root` when source images and ground truth are already
beside the result folders. Use `--digits auto` unless auto-detection fails; in
that case pass the explicit width, for example `--digits 3` for `mask000.tif`.

A successful validation prints:

```text
[CTC FORMAT] OK sequence=01 digits=3 frames=<n> tracks=<n> result_dir=<path>
```

## Run Official Metrics

Run TRA, SEG, and DET for one sequence:

```bash
python evaluate_ctc_results.py \
  --dataset-root /path/to/results/BF-C2DL-HSC \
  --source-root /path/to/source/BF-C2DL-HSC \
  --sequence 01 \
  --digits auto \
  --ctc-software-dir ./EvaluationSoftware \
  --metrics TRA SEG DET
```

If only some ground truth is available, request only the matching metrics. For
example, use `--metrics TRA` for a sequence with tracking ground truth only.
SEG requires `*_GT/SEG`; TRA and DET require tracking ground truth.

To evaluate multiple sequences:

```bash
for seq in 01 02; do
  python evaluate_ctc_results.py \
    --dataset-root /path/to/results/BF-C2DL-HSC \
    --source-root /path/to/source/BF-C2DL-HSC \
    --sequence "$seq" \
    --digits auto \
    --ctc-software-dir ./EvaluationSoftware \
    --metrics TRA SEG DET
done
```

## Outputs

For each sequence, the wrapper writes files inside `<sequence>_RES/`:

- `ctc_metrics_summary.csv`: one row per metric with parsed scores.
- `TRA_runner_output.txt`, `SEG_runner_output.txt`, `DET_runner_output.txt`:
  command, return code, stdout, and stderr from the official binaries.
- `TRA_log.txt`, `SEG_log.txt`, `DET_log.txt`: official metric logs when
  produced by the binaries.
- `ctc_metric_logs/`: archived copies of official logs.

It also writes dataset-level summaries next to the result folders:

- `ctc_TRA_penalty_counts.csv`
- `ctc_DET_penalty_counts.csv`
- `ctc_SEG_summary.csv`
- `ctc_SEG_low_jaccard_objects.csv`

Because these CSV files are written next to the result folders, use a writable
result root. The bundled `EvaluationSoftware/testing_dataset` is useful as
reference data, but its top-level folder may be read-only.

To regenerate CSV summaries from existing official logs without rerunning the
binaries:

```bash
python evaluate_ctc_results.py \
  --dataset-root /path/to/results/BF-C2DL-HSC \
  --parse-logs-only
```

## Smoke Test With Bundled Data

The bundled sample sequence `03` has tracking ground truth. Copy it to a
writable location before testing:

```bash
SMOKE_ROOT=$(mktemp -d /tmp/ctc_eval_smoke.XXXXXX)
cp -R EvaluationSoftware/testing_dataset "$SMOKE_ROOT/testing_dataset"
chmod -R u+w "$SMOKE_ROOT/testing_dataset"

python evaluate_ctc_results.py \
  --dataset-root "$SMOKE_ROOT/testing_dataset" \
  --sequence 03 \
  --digits 3 \
  --ctc-software-dir ./EvaluationSoftware \
  --metrics TRA
```

The expected TRA score for this sample is approximately `0.62298`.

## Full Workflow With Evaluation

If you are running the complete pipeline from raw sequence images through
tracking and final scoring, use `run_ctc_full_workflow.sh` and add
`--run-evaluation`:

```bash
bash run_ctc_full_workflow.sh \
  --dataset-root /path/to/source/BF-C2DL-HSC \
  --work-root /path/to/work/BF-C2DL-HSC \
  --output-root /path/to/results/BF-C2DL-HSC \
  --cellpose-model /path/to/CTC_fullscale \
  --sequences "01 02" \
  --run-evaluation \
  --ctc-software-dir ./EvaluationSoftware
```

The full workflow validates each final `<sequence>_RES` folder before running
metrics unless `--skip-validation` is supplied.

## Common Problems

`ModuleNotFoundError: No module named 'numpy'`: activate the virtual
environment and install the setup packages above.

`COMPRESSION.LZW requires the 'imagecodecs' package`: install `imagecodecs` in
the active environment.

`Could not find executable for metric TRA`: check that `--ctc-software-dir`
points to the folder containing `Mac/`, `Linux/`, and `Win/`. The wrapper picks
the platform-specific binary automatically.

`PermissionError` while writing `ctc_*.csv`: run evaluation against a writable
copy of the result root.

`[CTC FORMAT] FAIL`: fix the format issue before scoring. The validator catches
problems such as non-contiguous masks, wrong digit width, labels missing from
`res_track.txt`, frame gaps inside a track, shape mismatches, and invalid parent
IDs.
