#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep system/user-site Python packages from leaking into the workflow env.
export PYTHONNOUSERSITE=1

DATASET_ROOT=""
WORK_ROOT=""
OUTPUT_ROOT=""
FILM_MODEL=""
CELLPOSE_MODEL=""
PYTHON_BIN="${PYTHON:-python}"
PYTHON_EXPLICIT=0
ENV_MANAGER="${CTC_ENV_MANAGER:-auto}"
ENV_NAME="${CTC_ENV_NAME:-ctc-workflow}"
ENV_DIR="${CTC_ENV_DIR:-}"
ENV_PYTHON_VERSION="${CTC_ENV_PYTHON_VERSION:-3.10}"
SEQUENCES=()
CTC_ENTRYPOINT_MODE=0
CTC_DATASET_NAME=""

INTERPOLATION_FACTOR=2
FILM_CYCLES=""
FILM_BATCH_SIZE=1
FILM_NUM_WORKERS=0
FILM_WRITE_THREADS=10
FILM_OUTPUT_DIGITS=5

CELLPOSE_CHAN=0
CELLPOSE_DIAMETER=0
CELLPOSE_USE_GPU=1
CELLPOSE_VERBOSE=1
CELLPOSE_MASK_DIR=""
CELLPOSE_MASK_SUBDIR="masks"
SEG_MASK_PATTERN="*_cp_masks.tif"
CELLPOSE_EXTRA_ARGS=()

TRACKING_SCRIPT="$SCRIPT_DIR/ram_run_tiptracking_standalone_optimized.py"
TIME_SERIES_THRESHOLD=1
TRACK_OUTPUT_DIGITS="auto"
FINAL_OUTPUT_DIGITS="auto"
TEMPORAL_DOWNSAMPLE_OFFSET=0
STAGE_GT="auto"
CTC_SOFTWARE_DIR=""
RUN_EVALUATION=0
SKIP_INTERPOLATION=0
SKIP_SEGMENTATION=0
SKIP_TRACKING=0
SKIP_DOWNSAMPLE=0
SKIP_VALIDATION=0
PAD_MISSING_WITH_EMPTY=0
DRY_RUN=0
OVERWRITE=0

if [[ "${CTC_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1
fi

usage() {
  cat <<'EOF'
Run the full CTC interpolation -> segmentation -> tracking -> downsample workflow.

CTC entrypoint mode:
  If this script is copied or symlinked in an SW folder with a name like
  DatasetName-01.sh and run with no arguments, it follows the CTC software
  convention automatically:

    input images: ../DatasetName/01
    output:       ../DatasetName/01_RES
    scratch:      ./work/DatasetName-01
    FILM model:   ./models/film/saved_model
    Cellpose:     ./models/CTC_fullscale

Required paths:
  --dataset-root PATH       CTC dataset root containing sequence folders like 01, 02.
  --work-root PATH          Scratch/work folder for interpolated frames, masks, logs, and intermediate tracking.
  --output-root PATH        Final CTC-format output root containing 01_RES, 02_RES, etc.
  --cellpose-model PATH     Cellpose pretrained model path.

Core options:
  --sequences "01 02"       Space-separated sequence IDs. Default: auto-detect numeric folders.
  --interpolation-factor N  Temporal factor. Must be a power of two; default: 2.
  --film-cycles N           Override FILM cycles. Default: log2(interpolation-factor).
  --film-model PATH         Optional FILM SavedModel path. If omitted, interpolate_between_series_rapid.py uses its default.
  --python CMD              Python command for all Python stages; must import required modules unless bootstrap is enabled. Default: python.
  --env-manager MODE       Environment bootstrap: auto, conda, venv, or none. Default: auto.
  --env-name NAME          Conda/mamba environment name. Default: ctc-workflow.
  --env-dir PATH           Python venv directory. Default: WORK_ROOT/.ctc-env.
  --env-python-version N    Python version for new conda/mamba envs. Default: 3.10.
  --no-env-bootstrap       Alias for --env-manager none.
  --dry-run                 Print/log commands without running FILM, Cellpose, tracking, or validation.
  --overwrite               Remove this script's per-sequence work/output folders before running.

Cellpose options:
  --cellpose-chan N         Cellpose --chan value. Default: 0.
  --cellpose-diameter N     Cellpose --diameter value. Default: 0.
  --cellpose-no-gpu         Do not pass --use_gpu.
  --cellpose-quiet          Do not pass --verbose.
  --cellpose-mask-dir PATH  Exact mask directory to track. Relative paths are resolved inside each interpolated image dir.
  --cellpose-mask-subdir N  Expected mask subfolder under interpolated images. Default: masks.
  --seg-mask-pattern GLOB   Mask glob passed to tracking. Default: *_cp_masks.tif.
  --cellpose-extra-arg ARG  Extra argument passed to Cellpose; repeat for multiple args.

Tracking/downsampling options:
  --tracking-script PATH    Tracking runner. Default: ram_run_tiptracking_standalone_optimized.py.
  --time-series-threshold N Tracking time-series threshold. Default: 1.
  --track-output-digits N   Intermediate tracking mask digit width. Default: auto.
  --output-digits N         Final mask digit width. Default: auto.
  --downsample-offset N     First interpolated frame to keep. Default: 0.
  --pad-missing-with-empty  Let downsampling write blank masks for missing selected frames.

Validation/evaluation options:
  --stage-gt MODE           auto, copy, symlink, or none. Default: auto.
  --skip-validation         Do not run validate_ctc_result_format.py.
  --run-evaluation          Run evaluate_ctc_results.py after validation.
  --ctc-software-dir PATH   EvaluationSoftware root for official metrics.

Resume/debug options:
  --skip-interpolation
  --skip-segmentation
  --skip-tracking
  --skip-downsample
  --help

Example:
  bash run_ctc_full_workflow.sh \
    --dataset-root /data/BF-C2DL-HSC \
    --work-root /scratch/ctc_work/BF-C2DL-HSC \
    --output-root /scratch/ctc_submission/BF-C2DL-HSC \
    --cellpose-model /models/CTC_fullscale
EOF
}

log() {
  printf '[CTC WORKFLOW] %s\n' "$*"
}

die() {
  printf '[CTC WORKFLOW] ERROR: %s\n' "$*" >&2
  exit 1
}

quote_cmd() {
  printf '%q ' "$@"
}

run_cmd() {
  local log_file="$1"
  shift
  local -a cmd=("$@")
  mkdir -p "$(dirname "$log_file")"

  {
    printf 'timestamp=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf 'command='
    quote_cmd "${cmd[@]}"
    printf '\n\n'
  } > "$log_file"

  printf '\n[CTC WORKFLOW] '
  quote_cmd "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[DRY RUN] Command was not executed.\n' >> "$log_file"
    return 0
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee -a "$log_file"
  local status="${PIPESTATUS[0]}"
  set -e
  printf '\nreturncode=%s\n' "$status" >> "$log_file"
  if [[ "$status" -ne 0 ]]; then
    die "command failed with return code $status. See $log_file"
  fi
}

run_setup_cmd() {
  local -a cmd=("$@")

  printf '\n[CTC WORKFLOW] '
  quote_cmd "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[DRY RUN] Setup command was not executed.\n'
    return 0
  fi

  set +e
  "${cmd[@]}"
  local status="$?"
  set -e
  if [[ "$status" -ne 0 ]]; then
    die "setup command failed with return code $status"
  fi
}

path_arg() {
  local value="$1"
  if [[ "$value" == "." ]]; then
    pwd
  else
    printf '%s\n' "$value"
  fi
}

abs_existing_dir() {
  local value="$1"
  (cd "$value" && pwd)
}

abs_path_under_script_dir() {
  local value="$1"
  if [[ "$value" = /* || "$value" =~ ^[A-Za-z]:[\\/].* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$SCRIPT_DIR/$value"
  fi
}

configure_ctc_entrypoint_defaults() {
  local script_name dataset sequence
  script_name="$(basename "${BASH_SOURCE[0]}")"

  if [[ "$script_name" =~ ^(.+)-([0-9][0-9])\.sh$ ]]; then
    dataset="${BASH_REMATCH[1]}"
    sequence="${BASH_REMATCH[2]}"
  else
    return 0
  fi

  CTC_ENTRYPOINT_MODE=1
  CTC_DATASET_NAME="$dataset"
  DATASET_ROOT="$(abs_existing_dir "$SCRIPT_DIR/../$dataset")"
  WORK_ROOT="$SCRIPT_DIR/work/${dataset}-${sequence}"
  OUTPUT_ROOT="$DATASET_ROOT"
  FILM_MODEL="$(abs_path_under_script_dir "${CTC_FILM_MODEL:-models/film/saved_model}")"
  CELLPOSE_MODEL="$(abs_path_under_script_dir "${CTC_CELLPOSE_MODEL:-models/CTC_fullscale}")"
  SEQUENCES=("$sequence")
  STAGE_GT="none"
  SKIP_VALIDATION=1
}

normalize_sequence() {
  local sequence="$1"
  if [[ "$sequence" =~ ^[0-9]+$ ]]; then
    printf '%02d\n' "$((10#$sequence))"
  else
    printf '%s\n' "$sequence"
  fi
}

discover_sequences() {
  local -a found=()
  local path base
  shopt -s nullglob
  for path in "$DATASET_ROOT"/*; do
    [[ -d "$path" ]] || continue
    base="$(basename "$path")"
    if [[ "$base" =~ ^[0-9]+$ ]]; then
      found+=("$(normalize_sequence "$base")")
    fi
  done
  shopt -u nullglob

  if [[ "${#found[@]}" -eq 0 ]]; then
    die "no numeric sequence folders found under $DATASET_ROOT"
  fi
  printf '%s\n' "${found[@]}" | sort -u
}

cycles_for_factor() {
  local factor="$1"
  local cycles=0
  local value=1
  if ! [[ "$factor" =~ ^[0-9]+$ ]] || [[ "$factor" -lt 1 ]]; then
    die "--interpolation-factor must be a positive power of two"
  fi
  while [[ "$value" -lt "$factor" ]]; do
    value=$((value * 2))
    cycles=$((cycles + 1))
  done
  if [[ "$value" -ne "$factor" ]]; then
    die "--interpolation-factor must be a power of two"
  fi
  printf '%s\n' "$cycles"
}

find_conda_manager() {
  if command -v mamba >/dev/null 2>&1; then
    command -v mamba
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  return 1
}

conda_env_exists() {
  local manager="$1"
  "$manager" run -n "$ENV_NAME" python -c "import sys" >/dev/null 2>&1
}

resolve_conda_python() {
  local manager="$1"
  local resolved
  resolved="$("$manager" run -n "$ENV_NAME" python -c "import sys; print(sys.executable)")" \
    || die "could not resolve Python executable for conda environment '$ENV_NAME'"
  [[ -n "$resolved" ]] || die "could not resolve Python executable for conda environment '$ENV_NAME'"
  PYTHON_BIN="$resolved"
}

find_base_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  return 1
}

resolve_venv_python() {
  local env_dir="$1"
  local candidate
  local -a candidates=(
    "$env_dir/bin/python"
    "$env_dir/Scripts/python.exe"
    "$env_dir/Scripts/python"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      return 0
    fi
  done

  die "could not find Python executable inside venv: $env_dir"
}

install_python_requirements() {
  local requirements_file="$SCRIPT_DIR/requirements.txt"
  [[ -f "$requirements_file" ]] || die "requirements.txt not found next to workflow script: $requirements_file"

  run_setup_cmd "$PYTHON_BIN" -m pip install --upgrade pip
  run_setup_cmd "$PYTHON_BIN" -m pip install -r "$requirements_file"
}

bootstrap_conda_env() {
  local manager="$1"
  if conda_env_exists "$manager"; then
    log "using existing conda/mamba environment: $ENV_NAME"
  else
    log "creating conda/mamba environment: $ENV_NAME"
    run_setup_cmd "$manager" create -y -n "$ENV_NAME" "python=$ENV_PYTHON_VERSION" pip
  fi

  resolve_conda_python "$manager"
  log "environment python: $PYTHON_BIN"
  install_python_requirements
}

bootstrap_venv_env() {
  local base_python="$1"
  if [[ -z "$ENV_DIR" ]]; then
    ENV_DIR="$WORK_ROOT/.ctc-env"
  fi

  if [[ -x "$ENV_DIR/bin/python" || -x "$ENV_DIR/Scripts/python.exe" || -x "$ENV_DIR/Scripts/python" ]]; then
    log "using existing venv: $ENV_DIR"
  else
    log "creating venv: $ENV_DIR"
    mkdir -p "$(dirname "$ENV_DIR")"
    run_setup_cmd "$base_python" -m venv "$ENV_DIR"
  fi

  resolve_venv_python "$ENV_DIR"
  log "environment python: $PYTHON_BIN"
  install_python_requirements
}

bootstrap_python_environment() {
  local manager base_python

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: skipping environment bootstrap"
    return 0
  fi
  if [[ "$PYTHON_EXPLICIT" -eq 1 ]]; then
    log "using explicit --python; skipping environment bootstrap"
    return 0
  fi
  if [[ "$ENV_MANAGER" == "none" ]]; then
    log "environment bootstrap disabled"
    return 0
  fi

  case "$ENV_MANAGER" in
    auto)
      if manager="$(find_conda_manager)"; then
        bootstrap_conda_env "$manager"
        return 0
      fi
      base_python="$(find_base_python)" \
        || die "could not bootstrap environment: neither mamba/conda nor python3/python is available on PATH"
      bootstrap_venv_env "$base_python"
      ;;
    conda)
      manager="$(find_conda_manager)" \
        || die "--env-manager conda requested, but neither mamba nor conda is available on PATH"
      bootstrap_conda_env "$manager"
      ;;
    venv)
      base_python="$(find_base_python)" \
        || die "--env-manager venv requested, but neither python3 nor python is available on PATH"
      bootstrap_venv_env "$base_python"
      ;;
  esac
}

require_python_modules_available() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  local -a modules=()
  if [[ "$SKIP_INTERPOLATION" -eq 0 ]]; then
    modules+=(numpy cv2 tifffile tensorflow torch tqdm)
  fi
  if [[ "$SKIP_SEGMENTATION" -eq 0 ]]; then
    modules+=(cellpose)
  fi
  if [[ "$SKIP_TRACKING" -eq 0 ]]; then
    modules+=(numpy tifffile scipy skimage)
  fi
  if [[ "$SKIP_DOWNSAMPLE" -eq 0 ]]; then
    modules+=(numpy tifffile skimage)
  fi
  if [[ "$SKIP_VALIDATION" -eq 0 || "$RUN_EVALUATION" -eq 1 ]]; then
    modules+=(numpy tifffile)
  fi

  if [[ "${#modules[@]}" -eq 0 ]]; then
    return 0
  fi

  local missing
  missing="$("$PYTHON_BIN" -c "
import importlib
import sys

modules = dict.fromkeys(sys.argv[1:])
problems = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        problems.append(f'{module}: {exc.__class__.__name__}: {exc}')
if problems:
    print('\n'.join(problems))
    sys.exit(1)
" "${modules[@]}" 2>/dev/null)" || {
    if [[ "$missing" == *cellpose* ]]; then
      die "Cellpose is not importable with '$PYTHON_BIN'. Re-run with environment bootstrap enabled or inspect requirements.txt."
    fi
    if [[ "$missing" == *tensorflow* ]]; then
      die "TensorFlow is not importable with '$PYTHON_BIN'. If NumPy 2.x is installed, run '$PYTHON_BIN' -m pip install 'numpy<2' or re-run with environment bootstrap enabled. Details: ${missing:-unknown}"
    fi
    die "Required Python modules are not importable with '$PYTHON_BIN'. Details: ${missing:-unknown}. Re-run with environment bootstrap enabled or inspect requirements.txt."
  }
}

has_masks() {
  local folder="$1"
  local pattern="$2"
  local -a files=()
  shopt -s nullglob
  files=("$folder"/$pattern)
  shopt -u nullglob
  [[ "${#files[@]}" -gt 0 ]]
}

resolve_mask_dir() {
  local interpolated_dir="$1"
  local explicit="$2"

  if [[ -n "$explicit" ]]; then
    if [[ "$explicit" = /* || "$explicit" =~ ^[A-Za-z]:[\\/].* ]]; then
      printf '%s\n' "$explicit"
    else
      printf '%s\n' "$interpolated_dir/$explicit"
    fi
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%s\n' "$interpolated_dir/$CELLPOSE_MASK_SUBDIR"
    return 0
  fi

  local candidate
  local -a candidates=(
    "$interpolated_dir/$CELLPOSE_MASK_SUBDIR"
    "$interpolated_dir/masks"
    "$interpolated_dir/cellpose"
    "$interpolated_dir/cp_masks"
    "$interpolated_dir"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]] && has_masks "$candidate" "$SEG_MASK_PATTERN"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  shopt -s nullglob
  for candidate in "$interpolated_dir"/*; do
    if [[ -d "$candidate" ]] && has_masks "$candidate" "$SEG_MASK_PATTERN"; then
      shopt -u nullglob
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  shopt -u nullglob

  die "could not find Cellpose masks under $interpolated_dir using pattern $SEG_MASK_PATTERN"
}

stage_gt_for_sequence() {
  local sequence="$1"
  local gt_src="$DATASET_ROOT/${sequence}_GT"
  local gt_dst="$OUTPUT_ROOT/${sequence}_GT"

  case "$STAGE_GT" in
    none)
      return 0
      ;;
    auto)
      if [[ ! -d "$gt_src" ]]; then
        log "GT not found for $sequence; skipping GT staging"
        return 0
      fi
      ;;
    copy|symlink)
      [[ -d "$gt_src" ]] || die "missing GT folder for $sequence: $gt_src"
      ;;
    *)
      die "--stage-gt must be auto, copy, symlink, or none"
      ;;
  esac

  if [[ "$STAGE_GT" == "symlink" ]]; then
    log "stage GT symlink: $gt_src -> $gt_dst"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      mkdir -p "$(dirname "$gt_dst")"
      if [[ ! -e "$gt_dst" ]]; then
        ln -s "$(cd "$gt_src" && pwd)" "$gt_dst"
      fi
    fi
  else
    log "stage GT copy: $gt_src -> $gt_dst"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      mkdir -p "$gt_dst"
      cp -R "$gt_src"/. "$gt_dst"/
    fi
  fi
}

prepare_sequence_dirs() {
  local sequence="$1"
  local seq_work="$WORK_ROOT/$sequence"
  local final_res="$OUTPUT_ROOT/${sequence}_RES"

  if [[ "$OVERWRITE" -eq 1 ]]; then
    log "overwrite enabled for $sequence"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      rm -rf "$seq_work" "$final_res"
    fi
  fi

  if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$seq_work" "$OUTPUT_ROOT"
  fi
}

parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --dataset-root) DATASET_ROOT="$(path_arg "${2:?}")"; shift 2 ;;
      --work-root) WORK_ROOT="$(path_arg "${2:?}")"; shift 2 ;;
      --output-root) OUTPUT_ROOT="$(path_arg "${2:?}")"; shift 2 ;;
      --film-model) FILM_MODEL="${2:?}"; shift 2 ;;
      --cellpose-model) CELLPOSE_MODEL="${2:?}"; shift 2 ;;
      --sequences)
        read -r -a SEQUENCES <<< "${2:?}"
        shift 2
        ;;
      --interpolation-factor) INTERPOLATION_FACTOR="${2:?}"; shift 2 ;;
      --film-cycles) FILM_CYCLES="${2:?}"; shift 2 ;;
      --film-batch-size) FILM_BATCH_SIZE="${2:?}"; shift 2 ;;
      --film-num-workers) FILM_NUM_WORKERS="${2:?}"; shift 2 ;;
      --film-write-threads) FILM_WRITE_THREADS="${2:?}"; shift 2 ;;
      --film-output-digits) FILM_OUTPUT_DIGITS="${2:?}"; shift 2 ;;
      --python) PYTHON_BIN="${2:?}"; PYTHON_EXPLICIT=1; shift 2 ;;
      --env-manager) ENV_MANAGER="${2:?}"; shift 2 ;;
      --env-name) ENV_NAME="${2:?}"; shift 2 ;;
      --env-dir) ENV_DIR="$(path_arg "${2:?}")"; shift 2 ;;
      --env-python-version) ENV_PYTHON_VERSION="${2:?}"; shift 2 ;;
      --no-env-bootstrap) ENV_MANAGER="none"; shift ;;
      --cellpose-chan) CELLPOSE_CHAN="${2:?}"; shift 2 ;;
      --cellpose-diameter) CELLPOSE_DIAMETER="${2:?}"; shift 2 ;;
      --cellpose-no-gpu) CELLPOSE_USE_GPU=0; shift ;;
      --cellpose-quiet) CELLPOSE_VERBOSE=0; shift ;;
      --cellpose-mask-dir) CELLPOSE_MASK_DIR="${2:?}"; shift 2 ;;
      --cellpose-mask-subdir) CELLPOSE_MASK_SUBDIR="${2:?}"; shift 2 ;;
      --seg-mask-pattern) SEG_MASK_PATTERN="${2:?}"; shift 2 ;;
      --cellpose-extra-arg) CELLPOSE_EXTRA_ARGS+=("${2:?}"); shift 2 ;;
      --tracking-script) TRACKING_SCRIPT="${2:?}"; shift 2 ;;
      --time-series-threshold) TIME_SERIES_THRESHOLD="${2:?}"; shift 2 ;;
      --track-output-digits) TRACK_OUTPUT_DIGITS="${2:?}"; shift 2 ;;
      --output-digits) FINAL_OUTPUT_DIGITS="${2:?}"; shift 2 ;;
      --downsample-offset) TEMPORAL_DOWNSAMPLE_OFFSET="${2:?}"; shift 2 ;;
      --pad-missing-with-empty) PAD_MISSING_WITH_EMPTY=1; shift ;;
      --stage-gt) STAGE_GT="${2:?}"; shift 2 ;;
      --skip-validation) SKIP_VALIDATION=1; shift ;;
      --run-evaluation) RUN_EVALUATION=1; shift ;;
      --ctc-software-dir) CTC_SOFTWARE_DIR="${2:?}"; shift 2 ;;
      --skip-interpolation) SKIP_INTERPOLATION=1; shift ;;
      --skip-segmentation) SKIP_SEGMENTATION=1; shift ;;
      --skip-tracking) SKIP_TRACKING=1; shift ;;
      --skip-downsample) SKIP_DOWNSAMPLE=1; shift ;;
      --dry-run) DRY_RUN=1; shift ;;
      --overwrite) OVERWRITE=1; shift ;;
      --help|-h) usage; exit 0 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
}

validate_env_options() {
  case "$ENV_MANAGER" in
    auto|conda|venv|none)
      ;;
    *)
      die "--env-manager must be auto, conda, venv, or none"
      ;;
  esac

  if [[ -z "$ENV_NAME" ]]; then
    die "--env-name cannot be empty"
  fi
  if [[ -z "$ENV_PYTHON_VERSION" ]]; then
    die "--env-python-version cannot be empty"
  fi
}

validate_args() {
  [[ -n "$DATASET_ROOT" ]] || die "--dataset-root is required"
  [[ -n "$WORK_ROOT" ]] || die "--work-root is required"
  [[ -n "$OUTPUT_ROOT" ]] || die "--output-root is required"
  [[ -n "$CELLPOSE_MODEL" ]] || die "--cellpose-model is required"
  validate_env_options
  [[ -d "$DATASET_ROOT" ]] || die "dataset root does not exist: $DATASET_ROOT"
  [[ -f "$TRACKING_SCRIPT" ]] || die "tracking script does not exist: $TRACKING_SCRIPT"

  if [[ -z "$FILM_CYCLES" ]]; then
    FILM_CYCLES="$(cycles_for_factor "$INTERPOLATION_FACTOR")"
  fi
  if [[ "$INTERPOLATION_FACTOR" -ne $((2 ** FILM_CYCLES)) ]]; then
    die "--interpolation-factor must equal 2^--film-cycles"
  fi
  if [[ "$RUN_EVALUATION" -eq 1 && -z "$CTC_SOFTWARE_DIR" ]]; then
    die "--ctc-software-dir is required with --run-evaluation"
  fi

  if [[ "${#SEQUENCES[@]}" -eq 0 ]]; then
    local discovered
    while IFS= read -r discovered; do
      SEQUENCES+=("$discovered")
    done < <(discover_sequences)
  else
    local -a normalized=()
    local sequence
    for sequence in "${SEQUENCES[@]}"; do
      normalized+=("$(normalize_sequence "$sequence")")
    done
    SEQUENCES=("${normalized[@]}")
  fi
}

run_sequence() {
  local sequence="$1"
  local source_dir="$DATASET_ROOT/$sequence"
  local seq_work="$WORK_ROOT/$sequence"
  local log_dir="$seq_work/logs"
  local interpolated_dir="$seq_work/images_interpolated_${INTERPOLATION_FACTOR}x_tif"
  local tracking_root="$seq_work/tracking"
  local tracking_position="${sequence}_interp"
  local tracking_result_dir="$tracking_root/${tracking_position}_RES"
  local final_result_dir="$OUTPUT_ROOT/${sequence}_RES"

  [[ -d "$source_dir" ]] || die "missing source image sequence folder: $source_dir"
  prepare_sequence_dirs "$sequence"
  stage_gt_for_sequence "$sequence"

  log "===== sequence $sequence ====="
  log "source: $source_dir"
  log "interpolated: $interpolated_dir"
  log "final result: $final_result_dir"

  if [[ "$SKIP_INTERPOLATION" -eq 0 ]]; then
    local -a interpolate_cmd=(
      "$PYTHON_BIN" "$SCRIPT_DIR/interpolate_between_series_rapid.py" \
      --input_dir "$source_dir" \
      --output_dir "$interpolated_dir" \
      --cycles "$FILM_CYCLES" \
      --batch_size "$FILM_BATCH_SIZE" \
      --num_workers "$FILM_NUM_WORKERS" \
      --write_threads "$FILM_WRITE_THREADS" \
      --output_ext tif \
      --output_prefix t \
      --output_digits "$FILM_OUTPUT_DIGITS"
    )
    if [[ -n "$FILM_MODEL" ]]; then
      interpolate_cmd+=(--model_path "$FILM_MODEL")
    fi
    run_cmd "$log_dir/01_interpolate.log" "${interpolate_cmd[@]}"
  else
    log "skip interpolation for $sequence"
  fi

  if [[ "$SKIP_SEGMENTATION" -eq 0 ]]; then
    local -a cellpose_cmd=(
      "$PYTHON_BIN" -m cellpose
      --dir "$interpolated_dir"
      --pretrained_model "$CELLPOSE_MODEL"
      --chan "$CELLPOSE_CHAN"
      --diameter "$CELLPOSE_DIAMETER"
      --save_tif
      --no_npy
    )
    if [[ "$CELLPOSE_USE_GPU" -eq 1 ]]; then
      cellpose_cmd+=(--use_gpu)
    fi
    if [[ "$CELLPOSE_VERBOSE" -eq 1 ]]; then
      cellpose_cmd+=(--verbose)
    fi
    if [[ "${#CELLPOSE_EXTRA_ARGS[@]}" -gt 0 ]]; then
      cellpose_cmd+=("${CELLPOSE_EXTRA_ARGS[@]}")
    fi
    run_cmd "$log_dir/02_cellpose.log" "${cellpose_cmd[@]}"
  else
    log "skip segmentation for $sequence"
  fi

  if [[ "$SKIP_TRACKING" -eq 0 ]]; then
    local mask_dir
    mask_dir="$(resolve_mask_dir "$interpolated_dir" "$CELLPOSE_MASK_DIR")"
    log "segmentation masks: $mask_dir pattern=$SEG_MASK_PATTERN"
    run_cmd "$log_dir/03_tracking.log" \
      "$PYTHON_BIN" "$TRACKING_SCRIPT" \
      --mask-dir "$mask_dir" \
      --mask-pattern "$SEG_MASK_PATTERN" \
      --output-dir "$tracking_root" \
      --position "$tracking_position" \
      --time-series-threshold "$TIME_SERIES_THRESHOLD" \
      --output-digits "$TRACK_OUTPUT_DIGITS"
  else
    log "skip tracking for $sequence"
  fi

  if [[ "$SKIP_DOWNSAMPLE" -eq 0 ]]; then
    local -a downsample_cmd=(
      "$PYTHON_BIN" "$SCRIPT_DIR/temporal_downsample_ctc_results.py"
      --input-result-dir "$tracking_result_dir"
      --output-result-dir "$final_result_dir"
      --source-root "$DATASET_ROOT"
      --sequence "$sequence"
      --factor "$INTERPOLATION_FACTOR"
      --offset "$TEMPORAL_DOWNSAMPLE_OFFSET"
      --output-digits "$FINAL_OUTPUT_DIGITS"
    )
    if [[ "$PAD_MISSING_WITH_EMPTY" -eq 1 ]]; then
      downsample_cmd+=(--pad-missing-with-empty)
    fi
    run_cmd "$log_dir/04_temporal_downsample.log" "${downsample_cmd[@]}"
  else
    log "skip temporal downsample for $sequence"
  fi

  if [[ "$SKIP_VALIDATION" -eq 0 ]]; then
    run_cmd "$log_dir/05_validate.log" \
      "$PYTHON_BIN" "$SCRIPT_DIR/validate_ctc_result_format.py" \
      --dataset-root "$OUTPUT_ROOT" \
      --source-root "$DATASET_ROOT" \
      --sequence "$sequence" \
      --digits "$FINAL_OUTPUT_DIGITS"
  else
    log "skip validation for $sequence"
  fi

  if [[ "$RUN_EVALUATION" -eq 1 ]]; then
    run_cmd "$log_dir/06_evaluate.log" \
      "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_ctc_results.py" \
      --dataset-root "$OUTPUT_ROOT" \
      --source-root "$DATASET_ROOT" \
      --sequence "$sequence" \
      --digits "$FINAL_OUTPUT_DIGITS" \
      --ctc-software-dir "$CTC_SOFTWARE_DIR" \
      --metrics TRA SEG DET
  fi
}

main() {
  if [[ "$#" -eq 0 ]]; then
    configure_ctc_entrypoint_defaults
  fi
  parse_args "$@"
  validate_args
  bootstrap_python_environment
  require_python_modules_available

  if [[ "$CTC_ENTRYPOINT_MODE" -eq 1 ]]; then
    log "CTC entrypoint mode: dataset=$CTC_DATASET_NAME sequence=${SEQUENCES[*]}"
  fi
  log "dataset-root: $DATASET_ROOT"
  log "work-root: $WORK_ROOT"
  log "output-root: $OUTPUT_ROOT"
  log "sequences: ${SEQUENCES[*]}"
  log "interpolation-factor: $INTERPOLATION_FACTOR cycles: $FILM_CYCLES"

  local sequence
  for sequence in "${SEQUENCES[@]}"; do
    run_sequence "$sequence"
  done

  log "workflow complete"
}

main "$@"
