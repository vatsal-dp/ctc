# CTC Environment Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run_ctc_full_workflow.sh` create or reuse a dedicated Python environment, install all workflow dependencies, and run the pipeline with that environment's Python before any expensive stage starts.

**Architecture:** Keep the bootstrap inside `run_ctc_full_workflow.sh` so CTC entrypoint mode and direct CLI mode share one path. Add a small set of shell helpers for manager selection, environment creation, dependency installation, and import preflight. Use tests with fake `mamba`, `python3`, and environment Python executables so unit tests do not create real environments or hit the network.

**Tech Stack:** Bash, Python `unittest`, conda/mamba CLI when present, Python `venv` fallback, `pip install -r requirements.txt`.

---

## File Structure

- Modify `run_ctc_full_workflow.sh`: add CLI options, bootstrap helpers, dependency preflight, and logs.
- Modify `requirements.txt`: add the runtime dependencies used by interpolation, Cellpose, and tracking.
- Modify `test_ctc_full_workflow_script.py`: add fake-command tests for bootstrap selection and preflight behavior.

---

### Task 1: Extend Dependency Manifest And Help Text

**Files:**
- Modify: `requirements.txt`
- Modify: `run_ctc_full_workflow.sh`
- Test: `test_ctc_full_workflow_script.py`

- [ ] **Step 1: Write the failing tests**

Add these methods to `CTCFullWorkflowScriptTests` in `test_ctc_full_workflow_script.py`:

```python
    def test_help_documents_environment_bootstrap_options(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("--env-manager", result.stdout)
        self.assertIn("--env-name", result.stdout)
        self.assertIn("--env-dir", result.stdout)
        self.assertIn("--no-env-bootstrap", result.stdout)

    def test_requirements_include_workflow_runtime_dependencies(self):
        requirements = (
            SCRIPT.parent / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        packages = {line.strip().lower() for line in requirements if line.strip()}

        self.assertIn("numpy", packages)
        self.assertIn("scipy", packages)
        self.assertIn("scikit-image", packages)
        self.assertIn("tifffile", packages)
        self.assertIn("matplotlib", packages)
        self.assertIn("cellpose", packages)
        self.assertIn("opencv-python", packages)
        self.assertIn("tensorflow", packages)
        self.assertIn("torch", packages)
        self.assertIn("tqdm", packages)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -B -m unittest test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_help_documents_environment_bootstrap_options test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_requirements_include_workflow_runtime_dependencies
```

Expected: both tests fail because the help text and dependency list do not yet include the bootstrap additions.

- [ ] **Step 3: Update `requirements.txt`**

Replace the file contents with:

```text
numpy
scipy
scikit-image
tifffile
matplotlib
cellpose
opencv-python
tensorflow
torch
tqdm
```

- [ ] **Step 4: Update help text in `run_ctc_full_workflow.sh`**

In the `Core options:` section, keep the existing `--python` line and add:

```text
  --env-manager MODE       Environment bootstrap: auto, conda, venv, or none. Default: auto.
  --env-name NAME          Conda/mamba environment name. Default: ctc-workflow.
  --env-dir PATH           Python venv directory. Default: WORK_ROOT/.ctc-env.
  --no-env-bootstrap       Alias for --env-manager none.
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
python3 -B -m unittest test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_help_documents_environment_bootstrap_options test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_requirements_include_workflow_runtime_dependencies
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt run_ctc_full_workflow.sh test_ctc_full_workflow_script.py
git commit -m "chore(env): document workflow bootstrap dependencies"
```

---

### Task 2: Parse Bootstrap Options And Preserve Explicit Python Override

**Files:**
- Modify: `run_ctc_full_workflow.sh`
- Test: `test_ctc_full_workflow_script.py`

- [ ] **Step 1: Write the failing test**

Add this method to `CTCFullWorkflowScriptTests`:

```python
    def test_explicit_python_skips_environment_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            dataset_root = root / "BF-C2DL-HSC"
            work_root = root / "work"
            output_root = root / "submission"
            cellpose_model = root / "cellpose_model"
            fake_python = root / "python"
            bootstrap_log = root / "bootstrap.log"
            bin_dir.mkdir()
            for folder in [dataset_root / "01", cellpose_model]:
                folder.mkdir(parents=True)

            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "sys.exit(0)\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            (bin_dir / "mamba").write_text(
                "#!/bin/sh\n"
                f"echo unexpected-bootstrap >> '{bootstrap_log}'\n"
                "exit 99\n",
                encoding="utf-8",
            )
            (bin_dir / "mamba").chmod(0o755)

            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--dataset-root",
                    str(dataset_root),
                    "--work-root",
                    str(work_root),
                    "--output-root",
                    str(output_root),
                    "--cellpose-model",
                    str(cellpose_model),
                    "--python",
                    str(fake_python),
                    "--sequences",
                    "01",
                    "--stage-gt",
                    "none",
                    "--skip-interpolation",
                    "--skip-segmentation",
                    "--skip-tracking",
                    "--skip-downsample",
                    "--skip-validation",
                ],
                check=False,
                env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertFalse(bootstrap_log.exists(), msg=result.stdout)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -B -m unittest test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_explicit_python_skips_environment_bootstrap
```

Expected: fail until option state and explicit Python behavior are implemented.

- [ ] **Step 3: Add bootstrap state variables**

Near the existing `PYTHON_BIN` assignment in `run_ctc_full_workflow.sh`, add:

```bash
PYTHON_EXPLICIT=0
ENV_MANAGER="${CTC_ENV_MANAGER:-auto}"
ENV_NAME="${CTC_ENV_NAME:-ctc-workflow}"
ENV_DIR="${CTC_ENV_DIR:-}"
```

- [ ] **Step 4: Parse new options**

In `parse_args`, replace the existing `--python` case and add the new cases:

```bash
      --python) PYTHON_BIN="${2:?}"; PYTHON_EXPLICIT=1; shift 2 ;;
      --env-manager) ENV_MANAGER="${2:?}"; shift 2 ;;
      --env-name) ENV_NAME="${2:?}"; shift 2 ;;
      --env-dir) ENV_DIR="$(path_arg "${2:?}")"; shift 2 ;;
      --no-env-bootstrap) ENV_MANAGER="none"; shift ;;
```

- [ ] **Step 5: Validate bootstrap option values**

Add this helper before `validate_args`:

```bash
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
}
```

Call it near the top of `validate_args`, after required path checks:

```bash
  validate_env_options
```

- [ ] **Step 6: Run test to verify it passes**

Run:

```bash
python3 -B -m unittest test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_explicit_python_skips_environment_bootstrap
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add run_ctc_full_workflow.sh test_ctc_full_workflow_script.py
git commit -m "feat(env): parse workflow bootstrap options"
```

---

### Task 3: Implement Conda/Mamba Bootstrap

**Files:**
- Modify: `run_ctc_full_workflow.sh`
- Test: `test_ctc_full_workflow_script.py`

- [ ] **Step 1: Write the failing test**

Add this method to `CTCFullWorkflowScriptTests`:

```python
    def test_auto_bootstrap_uses_mamba_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            dataset_root = root / "BF-C2DL-HSC"
            work_root = root / "work"
            output_root = root / "submission"
            cellpose_model = root / "cellpose_model"
            fake_env_python = root / "env-python"
            bootstrap_log = root / "bootstrap.log"
            env_marker = root / "env-created"
            bin_dir.mkdir()
            for folder in [dataset_root / "01", cellpose_model]:
                folder.mkdir(parents=True)

            fake_env_python.write_text(
                "#!/usr/bin/env python3\n"
                "import os, sys\n"
                "with open(os.environ['FAKE_BOOTSTRAP_LOG'], 'a', encoding='utf-8') as handle:\n"
                "    handle.write('env-python ' + ' '.join(sys.argv[1:]) + '\\n')\n"
                "sys.exit(0)\n",
                encoding="utf-8",
            )
            fake_env_python.chmod(0o755)

            (bin_dir / "mamba").write_text(
                "#!/usr/bin/env python3\n"
                "import os, sys\n"
                "from pathlib import Path\n"
                "log = Path(os.environ['FAKE_BOOTSTRAP_LOG'])\n"
                "marker = Path(os.environ['FAKE_CONDA_ENV_CREATED'])\n"
                "with log.open('a', encoding='utf-8') as handle:\n"
                "    handle.write('mamba ' + ' '.join(sys.argv[1:]) + '\\n')\n"
                "args = sys.argv[1:]\n"
                "if args[:2] == ['env', 'list']:\n"
                "    if marker.exists():\n"
                "        print('ctc-workflow * /fake/ctc-workflow')\n"
                "    sys.exit(0)\n"
                "if args[:2] == ['create', '-y']:\n"
                "    marker.write_text('created', encoding='utf-8')\n"
                "    sys.exit(0)\n"
                "if args[:4] == ['run', '-n', 'ctc-workflow', 'python'] and args[4:6] == ['-c', 'import sys; print(sys.executable)']:\n"
                "    print(os.environ['FAKE_ENV_PYTHON'])\n"
                "    sys.exit(0)\n"
                "sys.exit(0)\n",
                encoding="utf-8",
            )
            (bin_dir / "mamba").chmod(0o755)

            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--dataset-root",
                    str(dataset_root),
                    "--work-root",
                    str(work_root),
                    "--output-root",
                    str(output_root),
                    "--cellpose-model",
                    str(cellpose_model),
                    "--sequences",
                    "01",
                    "--stage-gt",
                    "none",
                    "--skip-interpolation",
                    "--skip-segmentation",
                    "--skip-tracking",
                    "--skip-downsample",
                    "--skip-validation",
                ],
                check=False,
                env={
                    "PATH": f"{bin_dir}:/usr/bin:/bin",
                    "FAKE_BOOTSTRAP_LOG": str(bootstrap_log),
                    "FAKE_CONDA_ENV_CREATED": str(env_marker),
                    "FAKE_ENV_PYTHON": str(fake_env_python),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout)
            log = bootstrap_log.read_text(encoding="utf-8")
            self.assertIn("mamba env list", log)
            self.assertIn("mamba create -y -n ctc-workflow python=3.10 pip", log)
            self.assertIn("env-python -m pip install --upgrade pip", log)
            self.assertIn("env-python -m pip install -r", log)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -B -m unittest test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_auto_bootstrap_uses_mamba_when_available
```

Expected: fail because the script does not yet create conda/mamba environments.

- [ ] **Step 3: Add setup command runner**

Add this helper near `run_cmd`:

```bash
run_setup_cmd() {
  local -a cmd=("$@")

  printf '[CTC WORKFLOW] '
  quote_cmd "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry run: setup command was not executed"
    return 0
  fi

  set +e
  "${cmd[@]}"
  local status="$?"
  set -e
  if [[ "$status" -ne 0 ]]; then
    die "environment setup command failed with return code $status. If this is a GPU/CUDA machine, PyTorch or TensorFlow may need site-specific install commands."
  fi
}
```

- [ ] **Step 4: Add conda/mamba helpers**

Add these helpers before `validate_args`:

```bash
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
  "$manager" env list 2>/dev/null | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

resolve_conda_python() {
  local manager="$1"
  local resolved
  if ! resolved="$("$manager" run -n "$ENV_NAME" python -c "import sys; print(sys.executable)" 2>/dev/null)"; then
    die "could not resolve Python executable for conda environment '$ENV_NAME'"
  fi
  resolved="${resolved//$'\r'/}"
  PYTHON_BIN="$resolved"
}

install_python_dependencies() {
  [[ -f "$SCRIPT_DIR/requirements.txt" ]] || die "requirements.txt not found next to workflow script: $SCRIPT_DIR/requirements.txt"
  run_setup_cmd "$PYTHON_BIN" -m pip install --upgrade pip
  run_setup_cmd "$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements.txt"
}

bootstrap_conda_env() {
  local manager="$1"
  log "environment manager: $(basename "$manager")"
  if conda_env_exists "$manager"; then
    log "using existing conda environment: $ENV_NAME"
  else
    log "creating conda environment: $ENV_NAME"
    run_setup_cmd "$manager" create -y -n "$ENV_NAME" python=3.10 pip
  fi
  resolve_conda_python "$manager"
  install_python_dependencies
}
```

- [ ] **Step 5: Add bootstrap dispatcher**

Add this helper before `validate_args`:

```bash
bootstrap_python_env() {
  local manager

  if [[ "$DRY_RUN" -eq 1 || "$PYTHON_EXPLICIT" -eq 1 || "$ENV_MANAGER" == "none" ]]; then
    return 0
  fi

  case "$ENV_MANAGER" in
    auto)
      if manager="$(find_conda_manager)"; then
        bootstrap_conda_env "$manager"
        return 0
      fi
      return 1
      ;;
    conda)
      manager="$(find_conda_manager)" || die "--env-manager conda requested, but neither mamba nor conda is available on PATH"
      bootstrap_conda_env "$manager"
      return 0
      ;;
  esac

  return 1
}
```

Call it in `validate_args` after `validate_env_options`:

```bash
  bootstrap_python_env || true
```

This temporary `|| true` is replaced in Task 4 when `venv` fallback exists.

- [ ] **Step 6: Run test to verify it passes**

Run:

```bash
python3 -B -m unittest test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_auto_bootstrap_uses_mamba_when_available
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add run_ctc_full_workflow.sh test_ctc_full_workflow_script.py
git commit -m "feat(env): bootstrap workflow with conda or mamba"
```

---

### Task 4: Add Python `venv` Fallback

**Files:**
- Modify: `run_ctc_full_workflow.sh`
- Test: `test_ctc_full_workflow_script.py`

- [ ] **Step 1: Write the failing test**

Add this method to `CTCFullWorkflowScriptTests`:

```python
    def test_auto_bootstrap_falls_back_to_venv_without_conda(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            dataset_root = root / "BF-C2DL-HSC"
            work_root = root / "work"
            output_root = root / "submission"
            cellpose_model = root / "cellpose_model"
            env_dir = root / "workflow-env"
            bootstrap_log = root / "bootstrap.log"
            bin_dir.mkdir()
            for folder in [dataset_root / "01", cellpose_model]:
                folder.mkdir(parents=True)

            (bin_dir / "python3").write_text(
                "#!/bin/sh\n"
                "echo \"python3 $*\" >> \"$FAKE_BOOTSTRAP_LOG\"\n"
                "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"venv\" ]; then\n"
                "  env_dir=\"$3\"\n"
                "  mkdir -p \"$env_dir/bin\"\n"
                "  cat > \"$env_dir/bin/python\" <<'PY'\n"
                "#!/bin/sh\n"
                "echo \"env-python $*\" >> \"$FAKE_BOOTSTRAP_LOG\"\n"
                "exit 0\n"
                "PY\n"
                "  chmod +x \"$env_dir/bin/python\"\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "python3").chmod(0o755)

            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--dataset-root",
                    str(dataset_root),
                    "--work-root",
                    str(work_root),
                    "--output-root",
                    str(output_root),
                    "--cellpose-model",
                    str(cellpose_model),
                    "--env-dir",
                    str(env_dir),
                    "--sequences",
                    "01",
                    "--stage-gt",
                    "none",
                    "--skip-interpolation",
                    "--skip-segmentation",
                    "--skip-tracking",
                    "--skip-downsample",
                    "--skip-validation",
                ],
                check=False,
                env={
                    "PATH": f"{bin_dir}:/usr/bin:/bin",
                    "FAKE_BOOTSTRAP_LOG": str(bootstrap_log),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout)
            log = bootstrap_log.read_text(encoding="utf-8")
            self.assertIn(f"python3 -m venv {env_dir}", log)
            self.assertIn("env-python -m pip install --upgrade pip", log)
            self.assertIn("env-python -m pip install -r", log)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -B -m unittest test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_auto_bootstrap_falls_back_to_venv_without_conda
```

Expected: fail because auto mode currently has no `venv` fallback.

- [ ] **Step 3: Add venv helpers**

Add these helpers before `bootstrap_python_env`:

```bash
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
  if [[ -x "$env_dir/bin/python" ]]; then
    PYTHON_BIN="$env_dir/bin/python"
    return 0
  fi
  if [[ -x "$env_dir/Scripts/python.exe" ]]; then
    PYTHON_BIN="$env_dir/Scripts/python.exe"
    return 0
  fi
  if [[ -x "$env_dir/Scripts/python" ]]; then
    PYTHON_BIN="$env_dir/Scripts/python"
    return 0
  fi
  die "could not find Python executable inside venv: $env_dir"
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
    run_setup_cmd "$base_python" -m venv "$ENV_DIR"
  fi

  resolve_venv_python "$ENV_DIR"
  install_python_dependencies
}
```

- [ ] **Step 4: Replace bootstrap dispatcher**

Replace `bootstrap_python_env` with:

```bash
bootstrap_python_env() {
  local manager base_python

  if [[ "$DRY_RUN" -eq 1 || "$PYTHON_EXPLICIT" -eq 1 || "$ENV_MANAGER" == "none" ]]; then
    return 0
  fi

  case "$ENV_MANAGER" in
    auto)
      if manager="$(find_conda_manager)"; then
        bootstrap_conda_env "$manager"
        return 0
      fi
      base_python="$(find_base_python)" || die "could not bootstrap environment: neither mamba/conda nor python3/python is available on PATH"
      bootstrap_venv_env "$base_python"
      return 0
      ;;
    conda)
      manager="$(find_conda_manager)" || die "--env-manager conda requested, but neither mamba nor conda is available on PATH"
      bootstrap_conda_env "$manager"
      return 0
      ;;
    venv)
      base_python="$(find_base_python)" || die "--env-manager venv requested, but neither python3 nor python is available on PATH"
      bootstrap_venv_env "$base_python"
      return 0
      ;;
    none)
      return 0
      ;;
  esac
}
```

In `validate_args`, replace the temporary dispatcher call with:

```bash
  bootstrap_python_env
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
python3 -B -m unittest test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_auto_bootstrap_falls_back_to_venv_without_conda
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add run_ctc_full_workflow.sh test_ctc_full_workflow_script.py
git commit -m "feat(env): fall back to python venv bootstrap"
```

---

### Task 5: Generalize Dependency Preflight

**Files:**
- Modify: `run_ctc_full_workflow.sh`
- Test: `test_ctc_full_workflow_script.py`

- [ ] **Step 1: Keep the existing Cellpose failure test**

Use the existing `test_missing_cellpose_module_fails_before_interpolation` as the regression test. It must still fail before any interpolation command appears when `--python` points to an interpreter without Cellpose.

- [ ] **Step 2: Replace Cellpose-only preflight with module-list preflight**

Replace `require_cellpose_available` with:

```bash
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
  if [[ "$SKIP_VALIDATION" -eq 0 ]]; then
    modules+=(numpy tifffile)
  fi

  if [[ "${#modules[@]}" -eq 0 ]]; then
    return 0
  fi

  local module
  local -a unique=()
  for module in "${modules[@]}"; do
    if [[ " ${unique[*]} " != *" $module "* ]]; then
      unique+=("$module")
    fi
  done

  "$PYTHON_BIN" - "${unique[@]}" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    print(",".join(missing), file=sys.stderr)
    sys.exit(1)
PY
  local status="$?"
  if [[ "$status" -ne 0 ]]; then
    die "required Python modules are not importable with --python '$PYTHON_BIN': ${unique[*]}. Install dependencies in that environment or enable environment bootstrap."
  fi
}
```

In `validate_args`, replace:

```bash
  require_cellpose_available
```

with:

```bash
  require_python_modules_available
```

- [ ] **Step 3: Update the Cellpose test expectation**

In `test_missing_cellpose_module_fails_before_interpolation`, keep:

```python
            self.assertNotEqual(result.returncode, 0, msg=result.stdout)
            self.assertNotIn("interpolate_between_series_rapid.py", result.stdout)
```

Replace the message assertion with:

```python
            self.assertIn("required Python modules are not importable", result.stdout)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -B -m unittest test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_missing_cellpose_module_fails_before_interpolation
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add run_ctc_full_workflow.sh test_ctc_full_workflow_script.py
git commit -m "feat(env): preflight selected python modules"
```

---

### Task 6: Preserve Manual Bootstrap Disable Behavior

**Files:**
- Modify: `run_ctc_full_workflow.sh`
- Test: `test_ctc_full_workflow_script.py`

- [ ] **Step 1: Write the failing test**

Add this method to `CTCFullWorkflowScriptTests`:

```python
    def test_no_env_bootstrap_preserves_manual_python_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            dataset_root = root / "BF-C2DL-HSC"
            work_root = root / "work"
            output_root = root / "submission"
            cellpose_model = root / "cellpose_model"
            bootstrap_log = root / "bootstrap.log"
            bin_dir.mkdir()
            for folder in [dataset_root / "01", cellpose_model]:
                folder.mkdir(parents=True)

            (bin_dir / "mamba").write_text(
                "#!/bin/sh\n"
                f"echo unexpected-bootstrap >> '{bootstrap_log}'\n"
                "exit 99\n",
                encoding="utf-8",
            )
            (bin_dir / "mamba").chmod(0o755)

            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--dataset-root",
                    str(dataset_root),
                    "--work-root",
                    str(work_root),
                    "--output-root",
                    str(output_root),
                    "--cellpose-model",
                    str(cellpose_model),
                    "--sequences",
                    "01",
                    "--stage-gt",
                    "none",
                    "--no-env-bootstrap",
                    "--dry-run",
                ],
                check=False,
                env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertFalse(bootstrap_log.exists(), msg=result.stdout)
            self.assertIn("python -m cellpose", result.stdout)
```

- [ ] **Step 2: Run test to verify it passes**

Run:

```bash
python3 -B -m unittest test_ctc_full_workflow_script.CTCFullWorkflowScriptTests.test_no_env_bootstrap_preserves_manual_python_behavior
```

Expected: pass once `--no-env-bootstrap` parsing and dispatcher behavior are complete.

- [ ] **Step 3: Commit**

```bash
git add run_ctc_full_workflow.sh test_ctc_full_workflow_script.py
git commit -m "test(env): cover disabled bootstrap mode"
```

---

### Task 7: Full Verification

**Files:**
- Verify: `run_ctc_full_workflow.sh`
- Verify: `test_ctc_full_workflow_script.py`
- Verify: `requirements.txt`

- [ ] **Step 1: Run shell syntax check**

Run:

```bash
bash -n run_ctc_full_workflow.sh
```

Expected: exit code 0.

- [ ] **Step 2: Run workflow script tests**

Run:

```bash
python3 -B -m unittest test_ctc_full_workflow_script.py
```

Expected: all workflow script tests pass.

- [ ] **Step 3: Run full local unittest suite**

Run:

```bash
python3 -B -m unittest
```

Expected in a fully provisioned local Python: all tests pass. If this local Mac Python still lacks `numpy`, report the `ModuleNotFoundError` as an environment limitation and include the workflow-specific test result from Step 2.

- [ ] **Step 4: Inspect diff**

Run:

```bash
git diff -- run_ctc_full_workflow.sh test_ctc_full_workflow_script.py requirements.txt
```

Expected: diff only contains environment bootstrap behavior, dependency manifest additions, and related tests.

- [ ] **Step 5: Commit final verification adjustments**

If Task 7 required small fixes, commit them:

```bash
git add run_ctc_full_workflow.sh test_ctc_full_workflow_script.py requirements.txt
git commit -m "chore(env): verify workflow environment bootstrap"
```

If Task 7 required no fixes, do not create an empty commit.
