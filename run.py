"""Run the sample from scratch and leave inspectable outputs in work/."""
import json
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def main():
    subprocess.run([sys.executable, "prepare.py"], cwd=ROOT, check=True)
    notebook = nbformat.read(ROOT / "reference.ipynb", as_version=4)
    NotebookClient(notebook, kernel_name="python3", timeout=120).execute(cwd=str(ROOT))
    nbformat.write(notebook, ROOT / "work/reference.executed.ipynb")
    cmd = [sys.executable, "grade.py", "work/submission.csv"]
    first = subprocess.check_output(cmd, cwd=ROOT)
    second = subprocess.check_output(cmd, cwd=ROOT)
    if first != second:
        raise RuntimeError("Repeated grading returned different JSON.")
    report = json.loads(first)
    (ROOT / "work/report.json").write_bytes(first)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Reference failed a rubric check.")


if __name__ == "__main__":
    main()
