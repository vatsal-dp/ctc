#!/usr/bin/env python3
"""Compatibility wrapper for `ctc_tracking.workflows.run_ctc_training_pipeline`."""

from importlib import import_module
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_MODULE = import_module("ctc_tracking.workflows.run_ctc_training_pipeline")
for _name, _value in vars(_MODULE).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value
__all__ = [
    _name
    for _name in vars(_MODULE)
    if not (_name.startswith("__") and _name.endswith("__"))
]


if __name__ == "__main__":
    raise SystemExit(globals()["main"]())
