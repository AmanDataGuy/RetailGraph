"""
eval/run_eval.py

Single entry point for the eval pipeline. Runs collect_results.py in the
app's venv, then score_ragas.py in venv-eval — two separate interpreters,
because RAGAS needs a langchain version that conflicts with the app's
pinned langgraph/langchain (see eval/requirements-eval.txt for the full
explanation). This script just shells out to both in sequence so you don't
have to remember the two-venv dance.

One-time setup (only score_ragas.py's side needs this — collect_results.py
uses packages already in requirements.txt):
    python -m venv venv-eval
    venv-eval/Scripts/pip install -r eval/requirements-eval.txt

Usage (from the project root, with the main venv active or not — this
script calls the venvs by path, not by "current" interpreter):
    python eval/run_eval.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
IS_WINDOWS = sys.platform.startswith("win")

APP_PYTHON = ROOT / ("venv/Scripts/python.exe" if IS_WINDOWS else "venv/bin/python")
EVAL_PYTHON = ROOT / ("venv-eval/Scripts/python.exe" if IS_WINDOWS else "venv-eval/bin/python")


def run(python_bin: Path, script: str, label: str) -> None:
    if not python_bin.exists():
        raise SystemExit(
            f"{python_bin} not found.\n"
            f"{label} needs its own venv — see eval/requirements-eval.txt for setup."
        )
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    result = subprocess.run([str(python_bin), str(ROOT / "eval" / script)])
    if result.returncode != 0:
        raise SystemExit(f"{script} failed (exit {result.returncode})")


def main():
    run(APP_PYTHON, "collect_results.py", "Step 1/2 — collecting agent results (app venv)")
    run(EVAL_PYTHON, "score_ragas.py", "Step 2/2 — scoring with RAGAS (eval venv)")
    print("\nDone. See evaluation/ragas_results.json for the full report.")


if __name__ == "__main__":
    main()
