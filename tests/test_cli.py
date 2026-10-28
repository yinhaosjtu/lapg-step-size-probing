"""Check failure propagation and resumability without running the full benchmark."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

MOCK_INTERPRETER = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path
args = sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]
root = Path(option("--results-dir"))
root.mkdir(parents=True, exist_ok=True)
if args[0] == "report.py":
    (root / "report_called.txt").write_text("yes")
    raise SystemExit(0)
command = args[1]
if command == "verify" and os.environ.get("MOCK_FAIL_VERIFY") == "1":
    raise SystemExit(7)
if command == "synth":
    name = "synth_" + option("--m") + "x" + option("--n") + "_" + option("--seeds").replace(":", "_") + ".csv"
elif command == "real":
    name = "real_" + option("--data") + "_" + option("--noise") + "_" + option("--reps").replace(":", "_") + ".csv"
else:
    name = {"verify": "verification.txt", "example": "example1d.json", "ablation": "ablation.csv", "profile": "profile.json"}[command]
(root / name).write_text("completed test output")
"""


@unittest.skipUnless(shutil.which("bash"), "Bash is required for wrapper tests")
class WrapperTests(unittest.TestCase):
    def run_wrapper(self, fail_verify=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        shutil.copy(Path(__file__).resolve().parents[1] / "run_all.sh", root)
        interpreter = root / "mock-python"
        interpreter.write_text(MOCK_INTERPRETER, encoding="utf-8")
        interpreter.chmod(0o755)
        environment = dict(
            os.environ, PYTHON=str(interpreter), JOBS="64", LAPG_RESULTS_DIR=str(root / "results")
        )
        if fail_verify:
            environment["MOCK_FAIL_VERIFY"] = "1"
        result = subprocess.run(
            ["bash", str(root / "run_all.sh")],
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
        return root, result, environment

    def test_success_and_resume(self):
        root, result, environment = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        markers = list((root / "results").glob("*.done"))
        self.assertEqual(len(markers), 27)
        second = subprocess.run(
            ["bash", str(root / "run_all.sh")],
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(second.stdout.count("SKIP"), 27)

    def test_failure_is_not_reported_as_success(self):
        root, result, _ = self.run_wrapper(fail_verify=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((root / "results" / "verification.txt.done").exists())
        self.assertFalse((root / "results" / "report_called.txt").exists())


if __name__ == "__main__":
    unittest.main()
