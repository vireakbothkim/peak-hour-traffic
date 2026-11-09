"""Run every test file: python tests/run_all.py"""
import importlib.util
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
total = 0
for path in sorted(here.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in sorted(dir(module)):
        if name.startswith("test_"):
            getattr(module, name)()
            total += 1
            print(f"ok  {path.stem}.{name}")
print(f"\n{total} tests passed")
