#!/usr/bin/env python3

import subprocess
import tempfile
import unittest
import shutil
import shlex
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "run_ctc_full_workflow.sh"


class CTCFullWorkflowScriptTests(unittest.TestCase):
    def test_help_documents_core_paths(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("--dataset-root", result.stdout)
        self.assertIn("--film-model", result.stdout)
        self.assertIn("--cellpose-model", result.stdout)
        self.assertIn("--output-root", result.stdout)

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
        self.assertIn("--env-python-version", result.stdout)
        self.assertIn("--no-env-bootstrap", result.stdout)

    def test_requirements_include_workflow_runtime_dependencies(self):
        requirements = (
            SCRIPT.parent / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        packages = {
            line.strip().lower().split("<", 1)[0]
            for line in requirements
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertIn("numpy", packages)
        self.assertIn("scipy", packages)
        self.assertIn("scikit-image", packages)
        self.assertIn("tifffile", packages)
        self.assertIn("matplotlib", packages)
        self.assertIn("cellpose", packages)
        self.assertIn("cellpose<4", [line.strip().lower() for line in requirements])
        self.assertIn("opencv-python", packages)
        self.assertIn("tensorflow", packages)
        self.assertIn("tensorflow<2.11", [line.strip().lower() for line in requirements])
        self.assertIn("torch", packages)
        self.assertIn("tqdm", packages)

    def test_dry_run_auto_detects_numeric_sequences_and_prints_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "BF-C2DL-HSC"
            work_root = root / "work"
            output_root = root / "submission"
            film_model = root / "film_saved_model"
            cellpose_model = root / "cellpose_model"
            for folder in [
                dataset_root / "01",
                dataset_root / "02",
                dataset_root / "01_GT",
                dataset_root / "notes",
                film_model,
                cellpose_model,
            ]:
                folder.mkdir(parents=True)

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
                    "--film-model",
                    str(film_model),
                    "--cellpose-model",
                    str(cellpose_model),
                    "--stage-gt",
                    "none",
                    "--skip-validation",
                    "--dry-run",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertIn("sequences: 01 02", result.stdout)
            self.assertIn("interpolation-factor: 2 cycles: 1", result.stdout)
            self.assertIn("images_interpolated_2x_tif", result.stdout)
            self.assertIn("interpolate_between_series_rapid.py", result.stdout)
            self.assertIn("--batch_size 1", result.stdout)
            self.assertIn("--num_workers 0", result.stdout)
            self.assertIn("python -m cellpose", result.stdout)
            self.assertIn("ram_run_tiptracking_standalone_optimized.py", result.stdout)
            self.assertIn("temporal_downsample_ctc_results.py", result.stdout)
            self.assertNotIn("notes", result.stdout)

    def test_dry_run_can_use_film_runner_default_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "BF-C2DL-HSC"
            work_root = root / "work"
            output_root = root / "submission"
            cellpose_model = root / "cellpose_model"
            for folder in [dataset_root / "01", cellpose_model]:
                folder.mkdir(parents=True)

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
                    "--skip-validation",
                    "--dry-run",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertIn("interpolate_between_series_rapid.py", result.stdout)
            self.assertNotIn("--model_path", result.stdout)

    def test_missing_cellpose_module_fails_before_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "BF-C2DL-HSC"
            work_root = root / "work"
            output_root = root / "submission"
            cellpose_model = root / "cellpose_model"
            fake_python = root / "python"
            for folder in [dataset_root / "01", cellpose_model]:
                folder.mkdir(parents=True)
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "if sys.argv[1:2] == ['-c'] and 'cellpose' in sys.argv[3:]:\n"
                "    print('cellpose')\n"
                "    sys.exit(1)\n"
                "if sys.argv[1:3] == ['-m', 'cellpose']:\n"
                "    sys.stderr.write('No module named cellpose\\n')\n"
                "    sys.exit(1)\n"
                "sys.exit(0)\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

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
                    "--skip-validation",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            self.assertNotEqual(result.returncode, 0, msg=result.stdout)
            self.assertIn("Cellpose is not importable", result.stdout)
            self.assertNotIn("interpolate_between_series_rapid.py", result.stdout)

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
                f"echo unexpected-bootstrap >> {shlex.quote(str(bootstrap_log))}\n"
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

    def test_auto_bootstrap_creates_conda_env_and_installs_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            dataset_root = root / "BF-C2DL-HSC"
            work_root = root / "work"
            output_root = root / "submission"
            cellpose_model = root / "cellpose_model"
            marker = root / "env-created"
            env_python = root / "env-python"
            bootstrap_log = root / "bootstrap.log"
            bin_dir.mkdir()
            for folder in [dataset_root / "01", cellpose_model]:
                folder.mkdir(parents=True)

            env_python.write_text(
                "#!/bin/sh\n"
                f"echo \"env-python PYTHONNOUSERSITE=$PYTHONNOUSERSITE\" >> {shlex.quote(str(bootstrap_log))}\n"
                f"echo \"env-python $@\" >> {shlex.quote(str(bootstrap_log))}\n"
                "exit 0\n",
                encoding="utf-8",
            )
            env_python.chmod(0o755)
            (bin_dir / "mamba").write_text(
                "#!/bin/sh\n"
                f"log={shlex.quote(str(bootstrap_log))}\n"
                f"marker={shlex.quote(str(marker))}\n"
                f"env_python={shlex.quote(str(env_python))}\n"
                "echo \"mamba $@\" >> \"$log\"\n"
                "if [ \"$1\" = \"run\" ]; then\n"
                "  if [ ! -f \"$marker\" ]; then exit 1; fi\n"
                "  echo \"$env_python\"\n"
                "  exit 0\n"
                "fi\n"
                "if [ \"$1\" = \"create\" ]; then\n"
                "  touch \"$marker\"\n"
                "  exit 0\n"
                "fi\n"
                "exit 2\n",
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
                env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertIn("creating conda/mamba environment: ctc-workflow", result.stdout)
            self.assertIn(f"environment python: {env_python}", result.stdout)
            log = bootstrap_log.read_text(encoding="utf-8")
            self.assertIn("env-python PYTHONNOUSERSITE=1", log)
            self.assertIn("mamba create -y -n ctc-workflow python=3.10 pip", log)
            self.assertIn("env-python -m pip install --upgrade pip", log)
            self.assertIn("env-python -m pip install -r", log)

    def test_auto_bootstrap_falls_back_to_venv_without_conda(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            dataset_root = root / "BF-C2DL-HSC"
            work_root = root / "work"
            output_root = root / "submission"
            cellpose_model = root / "cellpose_model"
            env_dir = work_root / ".ctc-env"
            bootstrap_log = root / "bootstrap.log"
            bin_dir.mkdir()
            for folder in [dataset_root / "01", cellpose_model]:
                folder.mkdir(parents=True)

            (bin_dir / "python3").write_text(
                "#!/bin/sh\n"
                f"log={shlex.quote(str(bootstrap_log))}\n"
                "echo \"base-python $@\" >> \"$log\"\n"
                "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"venv\" ]; then\n"
                "  env_dir=\"$3\"\n"
                "  mkdir -p \"$env_dir/bin\"\n"
                "  cat > \"$env_dir/bin/python\" <<PY\n"
                "#!/bin/sh\n"
                "echo \"env-python PYTHONNOUSERSITE=\\$PYTHONNOUSERSITE\" >> \"$log\"\n"
                "echo \"env-python \\$@\" >> \"$log\"\n"
                "exit 0\n"
                "PY\n"
                "  chmod +x \"$env_dir/bin/python\"\n"
                "  exit 0\n"
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
            self.assertIn(f"creating venv: {env_dir}", result.stdout)
            self.assertIn(f"environment python: {env_dir / 'bin' / 'python'}", result.stdout)
            log = bootstrap_log.read_text(encoding="utf-8")
            self.assertIn("env-python PYTHONNOUSERSITE=1", log)
            self.assertIn(f"base-python -m venv {env_dir}", log)
            self.assertIn("env-python -m pip install --upgrade pip", log)
            self.assertIn("env-python -m pip install -r", log)

    def test_no_arg_ctc_entrypoint_infers_dataset_sequence_and_relative_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sw_dir = root / "SW"
            dataset_root = root / "BF-C2DL-HSC"
            entrypoint = sw_dir / "BF-C2DL-HSC-01.sh"

            (dataset_root / "01").mkdir(parents=True)
            (sw_dir / "models" / "film" / "saved_model").mkdir(parents=True)
            (sw_dir / "models" / "CTC_fullscale").mkdir(parents=True)
            shutil.copy2(SCRIPT, entrypoint)
            for name in [
                "interpolate_between_series_rapid.py",
                "ram_run_tiptracking_standalone_optimized.py",
                "temporal_downsample_ctc_results.py",
                "validate_ctc_result_format.py",
                "evaluate_ctc_results.py",
            ]:
                (sw_dir / name).write_text("# placeholder for dry-run\n", encoding="utf-8")

            result = subprocess.run(
                ["bash", str(entrypoint)],
                check=False,
                cwd=sw_dir,
                env={"CTC_DRY_RUN": "1", "PYTHON": "python"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertIn("CTC entrypoint mode: dataset=BF-C2DL-HSC sequence=01", result.stdout)
            self.assertIn(f"dataset-root: {dataset_root}", result.stdout)
            self.assertIn("sequences: 01", result.stdout)
            self.assertIn(str(dataset_root / "01"), result.stdout)
            self.assertIn(str(dataset_root / "01_RES"), result.stdout)
            self.assertIn(str(sw_dir / "models" / "film" / "saved_model"), result.stdout)
            self.assertIn(str(sw_dir / "models" / "CTC_fullscale"), result.stdout)
            self.assertIn("skip validation for 01", result.stdout)


if __name__ == "__main__":
    unittest.main()
