# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The kernel loop's templates: references, harness and check (orchard/skills/kernel-loop-templates).

Two kinds of test live here. The offline ones import kernel_check.py by path (it is stdlib only) and feed
`classify` text and result files that the simulator and the harness really wrote (tests/fixtures/kernel-loop/
outputs, recorded with tt-lang-sim 1.1.10; machine paths replaced by <ATTEMPT_DIR>, <TEMPLATES> and <TTLANG_VENV>).
The `needs_sim` ones copy the templates and a fixture kernel into tmp_path and run kernel_check.py against the
real simulator in the ttlang venv. They skip, and say so, when that venv is absent.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "kernel-loop-templates"
FIX = Path(__file__).parent / "fixtures" / "kernel-loop"
KERNELS = FIX / "kernels"
OUT = FIX / "outputs"

VENV = Path(os.environ.get("ORCHARD_TTLANG_VENV", Path.home() / "orchard-venvs" / "ttlang"))
SIM = VENV / "bin" / "tt-lang-sim"
needs_sim = pytest.mark.skipif(not SIM.exists(), reason="SKIPPED: no tt-lang-sim in the ttlang venv (run `tt-orchard kernel-env`); "
                                                        "the kernel loop was not run on the simulator")


def _load_check():
    spec = importlib.util.spec_from_file_location(
        "kernel_check_under_test", REPO / "orchard" / "skills" / "kernel-loop-templates" / "kernel_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CHECK = _load_check()


@pytest.mark.parametrize("name", ["numeric_mismatch", "shape_mismatch", "tt_lang_error", "import_error"])
def test_classify_reads_real_recorded_simulator_output(name):
    """Offline: the text and the result file are what the simulator and the harness really wrote (step 4)."""
    text = (OUT / f"{name}.txt").read_text()
    rp = OUT / f"{name}.result.json"
    result = json.loads(rp.read_text()) if rp.exists() else None
    got, line = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=result, bar=0.999)
    assert got == name and line


def test_without_the_diagnostics_file_the_check_reports_the_environment():
    """A tt-lang-sim wheel with no diagnostics.py ends every diagnosed kernel error as FileNotFoundError. That is a fault
    in the environment, so it must not be blamed on the kernel as a tt_lang_error."""
    text = ("Traceback (most recent call last):\n"
            "  File \"<TTLANG_VENV>/lib/python3.12/site-packages/ttl/sim/greenlet_scheduler.py\", line 338, in run\n"
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'<TTLANG_VENV>/lib/python3.12/site-packages/ttl/sim/diagnostics.py'\n")
    result = {"phase": "sim", "sets": [],
              "error": {"phase": "run", "type": "FileNotFoundError",
                        "text": "[Errno 2] No such file or directory: '.../ttl/sim/diagnostics.py'"}}
    got, line = CHECK.classify(kernel_present=True, timed_out=False, returncode=1, text=text, result=result, bar=0.999)
    assert got == "harness_error"
    assert "diagnostics.py" in line
    got, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=1, text=text, result=None, bar=0.999)
    assert got == "harness_error"


def test_classify_phase_hw_uses_the_hardware_names():
    """On hardware a runtime fault is compile_error and a low PCC is hw_numeric_mismatch (Task 9 calls it that way)."""
    text = (OUT / "tt_lang_error.txt").read_text()
    result = json.loads((OUT / "tt_lang_error.result.json").read_text())
    assert CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=result, bar=0.99,
                          phase="hw")[0] == "compile_error"
    result = json.loads((OUT / "numeric_mismatch.result.json").read_text())
    assert CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text="", result=result, bar=0.99,
                          phase="hw")[0] == "hw_numeric_mismatch"


# --- the simulator tests --------------------------------------------------------------------------------------------

def _attempt(tmp_path, kernel_name, op="add", spec_edit=None):
    """A copy of the templates and one fixture kernel, laid out the way an attempt directory is."""
    tdir = tmp_path / "templates"
    shutil.copytree(TEMPLATES, tdir, ignore=shutil.ignore_patterns("__pycache__"))
    adir = tmp_path / "attempt"
    adir.mkdir()
    if kernel_name is not None:
        shutil.copy(KERNELS / f"{kernel_name}.py", adir / "kernel.py")
    spec = json.loads((tdir / "ops" / f"{op}.json").read_text())
    if spec_edit:
        spec_edit(spec)
    (adir / "kernel_spec.json").write_text(json.dumps(spec))
    return tdir, adir


def _run_check(tdir, adir, *extra, timeout=360):
    done = subprocess.run([sys.executable, str(tdir / "kernel_check.py"), "--attempt-dir", str(adir), "--sim", str(SIM),
                           *extra], capture_output=True, text=True, timeout=timeout)
    check = json.loads((adir / "check.json").read_text())
    return done, check


@needs_sim
def test_add_good_passes(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_good")
    done, check = _run_check(tdir, adir)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == "PASS"
    assert check["passed"] is True and check["category"] is None
    assert all(s["pcc"] >= 0.999 for s in check["sets"])


@pytest.mark.parametrize("kernel,category,extra", [
    ("add_wrong_op", "numeric_mismatch", ()),
    ("add_shape_error", "shape_mismatch", ()),
    ("add_deadlock", "tt_lang_error", ()),
    ("add_syntax_error", "import_error", ()),
    ("add_no_operation", "no_kernel", ()),
    ("add_spin", "timeout", ("--timeout-s", "5")),
    ("add_nan", "numeric_mismatch", ()),
])
@needs_sim
def test_each_broken_kernel_gets_its_category(tmp_path, kernel, category, extra):
    tdir, adir = _attempt(tmp_path, kernel)
    done, check = _run_check(tdir, adir, *extra)
    assert done.returncode == 1, done.stdout + done.stderr
    assert done.stdout.startswith(f"FAIL: {category}: ")
    assert check["category"] == category and check["passed"] is False
    assert check["first_line"]


@needs_sim
def test_a_missing_kernel_is_no_kernel(tmp_path):
    tdir, adir = _attempt(tmp_path, None)
    done, check = _run_check(tdir, adir)
    assert done.returncode == 1
    assert check["category"] == "no_kernel" and check["passed"] is False


@needs_sim
def test_a_nan_output_is_recorded_as_not_finite(tmp_path):
    """The harness's `finite` flag must say False when the kernel writes NaN; a PCC of None alone is not enough
    for whoever reads check.json to tell a NaN from a constant output."""
    tdir, adir = _attempt(tmp_path, "add_nan")
    _, check = _run_check(tdir, adir)
    assert check["category"] == "numeric_mismatch"
    assert check["sets"] and all(s["finite"] is False for s in check["sets"])


@needs_sim
def test_the_heavy_tail_sets_run(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_good")
    _, check = _run_check(tdir, adir)
    names = [s["name"] for s in check["sets"]]
    assert "seed0" in names and "seed0-tail" in names
    assert len(names) == 6


@needs_sim
def test_a_spec_cannot_name_other_code(tmp_path):
    """The reference is a key into references.REFERENCES. A spec that names `os.system` must reach no code."""
    def edit(spec):
        spec["reference"] = "os.system"
    tdir, adir = _attempt(tmp_path, "add_good", spec_edit=edit)
    done, check = _run_check(tdir, adir)
    assert done.returncode == 1
    assert check["category"] == "harness_error", check
    assert check["sets"] == []


@needs_sim
def test_check_json_hashes_the_files_that_took_part(tmp_path):
    import hashlib
    tdir, adir = _attempt(tmp_path, "add_good")
    _, check = _run_check(tdir, adir)
    hashes = check["sha256"]
    assert set(hashes) == {"kernel.py", "kernel_harness.py", "references.py", "kernel_check.py", "spec"}
    assert all(isinstance(v, str) and len(v) == 64 for v in hashes.values())
    assert hashes["kernel.py"] == hashlib.sha256((adir / "kernel.py").read_bytes()).hexdigest()
    assert hashes["kernel_harness.py"] == hashlib.sha256((tdir / "kernel_harness.py").read_bytes()).hexdigest()


@needs_sim
def test_references_match_independent_formulas():
    """The references against hand-written loops on a 4x8 input, in the venv's python (this interpreter may lack torch).
    Importing torch there opens no device; the script never imports ttnn or ttl."""
    script = r"""
import math, sys
sys.path.insert(0, sys.argv[1])
import torch
from references import REFERENCES
g = torch.Generator().manual_seed(7)
A = torch.randn(4, 8, generator=g); B = torch.randn(4, 8, generator=g)
S = torch.sigmoid(torch.randn(4, 8, generator=g))
def close(x, y): return abs(x - y) <= 1e-5 * max(1.0, abs(y))
got = REFERENCES["silu_mul"]({"gate": A, "up": B})
for i in range(4):
    for j in range(8):
        g_ = float(A[i, j]); assert close(float(got[i, j]), g_ / (1 + math.exp(-g_)) * float(B[i, j])), ("silu_mul", i, j)
got = REFERENCES["rmsnorm"]({"x": A, "w": B})
for i in range(4):
    ms = sum(float(A[i, j]) ** 2 for j in range(8)) / 8
    for j in range(8):
        assert close(float(got[i, j]), float(A[i, j]) / math.sqrt(ms + 1e-6) * float(B[i, j])), ("rmsnorm", i, j)
got = REFERENCES["linear_scan"]({"a": S, "b": B})
for j in range(8):
    h = 0.0
    for t in range(4):
        h = float(S[t, j]) * h + float(B[t, j])
        assert close(float(got[t, j]), h), ("linear_scan", t, j)
got = REFERENCES["add"]({"a": A, "b": B})
assert torch.equal(got, A + B)
print("REFERENCES OK")
"""
    done = subprocess.run([str(VENV / "bin" / "python"), "-c", script, str(TEMPLATES)], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0 and "REFERENCES OK" in done.stdout, done.stdout + done.stderr


@pytest.mark.parametrize("op", ["silu_mul", "rmsnorm"])
@needs_sim
def test_the_good_kernels_pass_on_the_simulator(tmp_path, op):
    tdir, adir = _attempt(tmp_path, f"{op}_good", op=op)
    done, check = _run_check(tdir, adir)
    assert done.returncode == 0, done.stdout + done.stderr
    assert check["passed"] is True and check["op"] == op
    assert len(check["sets"]) == 6 and all(s["pcc"] >= 0.999 and s["finite"] for s in check["sets"])


def test_the_op_specs_name_only_known_references():
    """Every spec under ops/ names a reference in the table and lists its inputs in the order the kernel takes them."""
    import ast
    tree = ast.parse((TEMPLATES / "references.py").read_text())
    table = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "REFERENCES")
    known = {k.value for k in table.value.keys}
    specs = sorted((TEMPLATES / "ops").glob("*.json"))
    assert {p.stem for p in specs} == {"add", "silu_mul", "rmsnorm", "linear_scan"}
    for p in specs:
        spec = json.loads(p.read_text())
        assert spec["op"] == p.stem and spec["reference"] in known
        assert spec["pcc_bar_sim"] == 0.999 and spec["seeds"] == [0, 1, 2] and spec["heavy_tail"] is True
