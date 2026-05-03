# CTC Workflow Environment Bootstrap Design

## Goal

Make `run_ctc_full_workflow.sh` start from a clean machine more reliably by creating or reusing a dedicated Python environment before any expensive pipeline stage runs. The bootstrap must work on machines with Anaconda and on machines with only a standard Python installation.

## Default Behavior

The workflow uses `--env-manager auto` by default. In auto mode it tries, in order:

1. `mamba` or `conda`, if available on `PATH`.
2. Python `venv`, if a usable `python3` or `python` command is available.
3. A clear early failure if neither environment manager can be used.

After creating or finding the environment, the script sets `PYTHON_BIN` to that environment's Python executable and runs all Python stages through it. The existing `--python` option remains an explicit override.

## User Options

Add these options:

- `--env-manager auto|conda|venv|none`: choose bootstrap strategy. Default: `auto`.
- `--env-name NAME`: named conda environment when using conda or mamba. Default: `ctc-workflow`.
- `--env-dir PATH`: local venv directory when using `venv`. Default: `$WORK_ROOT/.ctc-env`.
- `--no-env-bootstrap`: alias for `--env-manager none`.

If `--python` is provided, the script uses that Python directly and skips environment creation. It still runs dependency preflight checks before expensive stages.

## Dependencies

The workflow installs the runtime packages needed by interpolation, segmentation, tracking, downsampling, validation, and optional analysis helpers:

- Existing `requirements.txt` packages: `numpy`, `scipy`, `scikit-image`, `tifffile`, `matplotlib`
- Missing runtime packages used by the workflow: `cellpose`, `opencv-python`, `tensorflow`, `torch`, `tqdm`

The dependency list should live in `requirements.txt` so the script and local tests agree on one source of truth.

## Flow

1. Parse CLI arguments.
2. Validate required paths that do not depend on Python packages.
3. Bootstrap the environment unless disabled or `--python` was explicitly supplied.
4. Upgrade `pip` inside the selected environment.
5. Install all packages from `requirements.txt`.
6. Run import preflight checks for the packages required by the selected stages.
7. Start interpolation, Cellpose segmentation, tracking, downsampling, validation, and evaluation as before.

## Error Handling

Environment creation and dependency installation fail before any sequence processing starts. Error messages should name the selected manager, the environment name or path, and the failed command. If dependency installation fails on a GPU machine, the message should note that PyTorch/TensorFlow CUDA builds may need site-specific installation commands.

## Testing

Add dry-run or fake-command tests that prove:

- Auto mode chooses conda/mamba when available.
- Auto mode falls back to `venv` when conda/mamba are absent.
- `--python` skips bootstrap and uses the supplied Python.
- Missing dependencies are reported before interpolation.
- `--no-env-bootstrap` preserves the old manual-environment behavior.
