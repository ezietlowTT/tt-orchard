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
            "'<TTLANG_VENV>/lib/python3.12/site-packages/ttl/ttl/diagnostics.py'\n")
    result = {"phase": "sim", "sets": [],
              "error": {"phase": "run", "type": "FileNotFoundError",
                        "text": "[Errno 2] No such file or directory: '.../ttl/ttl/diagnostics.py'"}}
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


@pytest.mark.parametrize("kernel,category,extra,op", [
    ("add_wrong_op", "numeric_mismatch", (), "add"),
    ("add_shape_error", "shape_mismatch", (), "add"),
    ("add_deadlock", "tt_lang_error", (), "add"),
    ("add_syntax_error", "import_error", (), "add"),
    ("add_no_operation", "no_kernel", (), "add"),
    ("add_spin", "timeout", ("--timeout-s", "5"), "add"),
    ("add_nan", "numeric_mismatch", (), "add"),
    # correlated with the right answer, so PCC alone passes them
    ("add_scaled", "numeric_mismatch", (), "add"),
    ("rmsnorm_no_norm", "numeric_mismatch", (), "rmsnorm"),
    ("rmsnorm_wrong_n", "numeric_mismatch", (), "rmsnorm"),
    # the static gate, one fixture per rule
    ("forbidden_import", "forbidden_code", (), "add"),
    ("forbidden_name", "forbidden_code", (), "add"),
    ("forbidden_dunder", "forbidden_code", (), "add"),
    ("forbidden_readback", "forbidden_code", (), "add"),
    ("forbidden_undecorated", "forbidden_code", (), "add"),
    # prints the missing-diagnostics text itself; diagnostics.py is present, so the kernel is blamed
    ("add_spoof_diagnostics", "tt_lang_error", (), "add"),
    ("forbidden_in_main", "forbidden_code", (), "add"),
    ("forbidden_readback_import", "forbidden_code", (), "add"),
    ("forbidden_stacked_decorator", "forbidden_code", (), "add"),
    ("forbidden_for_binding", "forbidden_code", (), "add"),
    ("forbidden_walrus", "forbidden_code", (), "add"),
])
@needs_sim
def test_each_broken_kernel_gets_its_category(tmp_path, kernel, category, extra, op):
    tdir, adir = _attempt(tmp_path, kernel, op=op)
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


# --- fix round 1: error bars beyond PCC, the static gate, the setup phase -------------------------------------------

SIM_BARS = {"pcc": 0.999, "rel_l2": 1e-2, "slope": [0.98, 1.02], "max_abs_atol": 1e-2, "max_abs_rtol": 1e-2}
HW_BARS = {"pcc": 0.99, "rel_l2": 5e-2, "slope": [0.95, 1.05], "max_abs_atol": 1e-1, "max_abs_rtol": 5e-2}


def _good_set(**over):
    s = {"name": "seed0", "seed": 0, "heavy_tail": False, "pcc": 0.99999, "max_abs_err": 0.001, "finite": True,
         "rel_l2": 0.001, "slope": 1.0, "exp_abs_max": 4.0}
    s.update(over)
    return s


def _classify_sets(sets, phase="sim", bars=None, **kw):
    args = dict(kernel_present=True, timed_out=False, returncode=0, text="", result={"phase": phase, "sets": sets,
                "error": None}, bar=(bars or (SIM_BARS if phase == "sim" else HW_BARS))["pcc"], phase=phase)
    args.update(kw)
    if bars is not None:
        args["bars"] = bars
    return CHECK.classify(**args)


def test_a_set_inside_every_bar_passes():
    assert _classify_sets([_good_set()], bars=SIM_BARS) == (None, "")


@pytest.mark.parametrize("over,needle", [
    ({"rel_l2": 0.05}, "rel_l2 0.05 above bar 0.01"),
    ({"slope": 2.0}, "slope 2.0 outside [0.98, 1.02]"),
    ({"slope": 0.5}, "slope 0.5 outside [0.98, 1.02]"),
    ({"max_abs_err": 0.2, "exp_abs_max": 4.0}, "max_abs_err 0.2 above"),
])
def test_each_new_bar_fails_a_set_alone(over, needle):
    """PCC passes in every case: only the named field is out of its bar, and the FAIL line names it."""
    category, line = _classify_sets([_good_set(**over)], bars=SIM_BARS)
    assert category == "numeric_mismatch"
    assert needle in line and line.startswith("set seed0: ")


@pytest.mark.parametrize("field", ["rel_l2", "slope", "max_abs_err", "exp_abs_max", "pcc"])
@pytest.mark.parametrize("how", ["missing", "null"])
def test_a_missing_or_null_field_fails(field, how):
    s = _good_set()
    if how == "missing":
        del s[field]
    else:
        s[field] = None
    category, line = _classify_sets([s], bars=SIM_BARS)
    assert category == "numeric_mismatch"
    assert field in line


def test_the_hw_bars_are_wider_and_use_the_hw_name():
    s = _good_set(slope=1.03, rel_l2=0.03)
    assert _classify_sets([s], phase="sim", bars=SIM_BARS)[0] == "numeric_mismatch"
    assert _classify_sets([s], phase="hw", bars=HW_BARS) == (None, "")
    assert _classify_sets([_good_set(slope=1.2)], phase="hw", bars=HW_BARS)[0] == "hw_numeric_mismatch"


def test_classify_without_bars_uses_the_shipped_defaults():
    """Task 9 may call classify with only `bar`: the new bars still apply, from the phase's defaults."""
    assert _classify_sets([_good_set(slope=2.0)])[0] == "numeric_mismatch"
    assert _classify_sets([_good_set()]) == (None, "")
    assert _classify_sets([_good_set(slope=1.03)], phase="hw") == (None, "")


def test_bars_from_spec_reads_each_phase():
    spec = json.loads((TEMPLATES / "ops" / "add.json").read_text())
    assert CHECK.bars_from_spec(spec, "sim") == SIM_BARS
    assert CHECK.bars_from_spec(spec, "hw") == HW_BARS


def test_every_op_spec_has_the_provisional_bars():
    for p in sorted((TEMPLATES / "ops").glob("*.json")):
        spec = json.loads(p.read_text())
        assert spec["provisional"] is True
        assert spec["rel_l2_bar_sim"] == 1e-2 and spec["rel_l2_bar_hw"] == 5e-2, p.name
        assert spec["slope_band_sim"] == [0.98, 1.02] and spec["slope_band_hw"] == [0.95, 1.05], p.name
        assert spec["max_abs_atol_sim"] == 1e-2 and spec["max_abs_rtol_sim"] == 1e-2, p.name
        assert spec["max_abs_atol_hw"] == 1e-1 and spec["max_abs_rtol_hw"] == 5e-2, p.name


def test_classify_timed_out_offline():
    got = CHECK.classify(kernel_present=True, timed_out=True, returncode=None, text="PROGRESS import\n", result=None,
                         bar=0.999)
    assert got[0] == "timeout"


@pytest.mark.parametrize("phase", ["sim", "hw"])
def test_a_setup_phase_error_is_the_harness(phase):
    """The device would not open: not the kernel's fault, in either phase."""
    result = {"phase": phase, "sets": [], "error": {"phase": "setup", "type": "RuntimeError",
                                                    "text": "Failed to open device 0: chip is busy"}}
    text = "PROGRESS import\nPROGRESS open_device\nRuntimeError: Failed to open device 0: chip is busy\n"
    category, line = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=result,
                                    bar=0.999, phase=phase)
    assert category == "harness_error" and "chip is busy" in line


def test_a_nan_set_fails_offline():
    s = _good_set(pcc=None, finite=False, rel_l2=None, slope=None, max_abs_err=float("nan"))
    category, line = _classify_sets([s], bars=SIM_BARS)
    assert category == "numeric_mismatch" and "finite False" in line


def test_a_missing_result_after_the_kernel_started_is_the_kernel():
    text = "PROGRESS import\nPROGRESS open_device\nPROGRESS set seed0\nPROGRESS run seed0\n"
    category, line = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=None,
                                    bar=0.999)
    assert category == "tt_lang_error"
    assert line == "the kernel ended the process without a result"
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=None,
                                 bar=0.99, phase="hw")
    assert category == "compile_error"


def test_a_missing_result_before_the_kernel_loaded_is_the_harness():
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=1, text="Traceback ...\n",
                                 result=None, bar=0.999)
    assert category == "harness_error"


SPOOF = ("FileNotFoundError: [Errno 2] No such file or directory: 'ttl/sim/diagnostics.py'\n"
         "PROGRESS import\n")


def test_spoofed_diagnostics_text_is_the_kernel_when_the_file_is_present():
    result = {"phase": "sim", "sets": [], "error": {"phase": "run", "type": "ValueError", "text": "boom"}}
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=SPOOF, result=result,
                                 bar=0.999, diagnostics_present=True)
    assert category == "tt_lang_error"
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=SPOOF, result=result,
                                 bar=0.999, diagnostics_present=False)
    assert category == "harness_error"


def test_diagnostics_file_lookup(tmp_path):
    """The file the wheel fails to ship is ttl/ttl/diagnostics.py. The wheel's loader, ttl/sim/diagnostics.py, is always
    there and reads the other one (ttl/sim/diagnostics.py lines 35-36; ttlang_env.diagnostics_target)."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    sim = venv / "bin" / "tt-lang-sim"
    sim.write_text("")
    assert CHECK.diagnostics_present(sim) is None                 # no ttl package found: unknown
    ttl = venv / "lib" / "python3.12" / "site-packages" / "ttl"
    (ttl / "ttl").mkdir(parents=True)
    assert CHECK.diagnostics_present(sim) is False
    (ttl / "ttl" / "diagnostics.py").write_text("")
    assert CHECK.diagnostics_present(sim) is True


def test_the_loader_alone_does_not_count_as_the_diagnostics_file(tmp_path):
    """I1. The loader ttl/sim/diagnostics.py is present in every wheel; with the target ttl/ttl/diagnostics.py absent,
    the file is missing."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    sim = venv / "bin" / "tt-lang-sim"
    sim.write_text("")
    ttl = venv / "lib" / "python3.12" / "site-packages" / "ttl"
    (ttl / "sim").mkdir(parents=True)
    (ttl / "sim" / "diagnostics.py").write_text("# the loader\n")
    assert CHECK.diagnostics_present(sim) is False


# the static gate

@pytest.mark.parametrize("kernel,rules", [
    ("forbidden_import", ["import"]),
    ("forbidden_name", ["name"]),
    ("forbidden_dunder", ["dunder"]),
    ("forbidden_readback", ["readback"]),
    ("forbidden_undecorated", ["operation"]),
    ("kernel_exit", ["import", "name"]),                   # `import os`, then `os._exit(0)`
    ("forbidden_in_main", ["name"]),
    ("forbidden_readback_import", ["readback", "readback"]),  # the import, then the bare use
    ("forbidden_stacked_decorator", ["operation"]),
    ("forbidden_for_binding", ["operation"]),
    ("forbidden_walrus", ["operation"]),
])
def test_the_gate_refuses_each_rule(kernel, rules):
    ok, reason = CHECK.check_kernel_source(KERNELS / f"{kernel}.py")
    assert ok is False
    n = len(rules)
    assert reason.startswith(f"{n} violation{'s' if n > 1 else ''}: {rules[0]} at line "), reason
    found = CHECK.kernel_source_violations(KERNELS / f"{kernel}.py")
    assert [v["rule"] for v in found] == rules
    assert all(isinstance(v["line"], int) and v["line"] > 0 and v["remedy"] for v in found)


@pytest.mark.parametrize("kernel", ["add_good", "silu_mul_good", "rmsnorm_good", "add_wrong_op", "add_nan", "add_scaled",
                                    "add_shape_error", "add_deadlock", "add_spin", "rmsnorm_no_norm", "rmsnorm_wrong_n",
                                    "add_spoof_diagnostics", "add_syntax_error", "add_alias_module", "add_alias_ttnn",
                                    "add_alias_function", "add_alias_function_as", "add_boilerplate"])
def test_the_gate_lets_plain_kernels_through(kernel):
    """add_syntax_error does not parse; the gate leaves it to the harness, which reports import_error."""
    assert CHECK.check_kernel_source(KERNELS / f"{kernel}.py") == (True, "")


def test_the_gate_reports_no_operation():
    ok, reason = CHECK.check_kernel_source(KERNELS / "add_no_operation.py")
    assert ok is False and reason.startswith("no-operation")


GATE_HEADER = 'import ttl\nimport ttnn\n\n\n@ttl.operation(grid="full")\ndef OPERATION(a, b, out):\n'


@pytest.mark.parametrize("line,rule", [
    ("    import sys", "import"), ("    import os.path", "import"), ("    import atexit", "import"),
    ("    import builtins", "import"), ("    import subprocess", "import"), ("    import importlib", "import"),
    ("    import references", "import"), ("    import torch", "import"), ("    from . import x", "import"),
    ("    from references import REFERENCES", "import"),
    ("    open('f')", "name"), ("    exec('1')", "name"), ("    eval('1')", "name"), ("    compile('1', 'f', 'exec')", "name"),
    ("    x = __import__", "name"), ("    globals()", "name"), ("    locals()", "name"), ("    setattr(a, 'b', 1)", "name"),
    ("    getattr(a, 'b')", "name"), ("    breakpoint()", "name"), ("    input()", "name"), ("    x = sys", "name"),
    ("    x = os", "name"), ("    x = __builtins__", "name"), ("    x = references", "name"),
    ("    x = a.__dict__", "dunder"), ("    x = ttl.__spec__", "dunder"),
    ("    x = ttnn.to_torch", "readback"), ("    x = ttnn.from_torch", "readback"),
])
def test_the_gate_rules_in_detail(tmp_path, line, rule):
    p = tmp_path / "kernel.py"
    p.write_text(GATE_HEADER + line + "\n")
    ok, reason = CHECK.check_kernel_source(p)
    assert ok is False and reason.startswith(f"1 violation: {rule} at line 7: "), reason


def test_the_gate_allows_math(tmp_path):
    p = tmp_path / "kernel.py"
    p.write_text("import math\n" + GATE_HEADER + "    x = math.sqrt(2.0)\n")
    assert CHECK.check_kernel_source(p) == (True, "")


@pytest.mark.parametrize("src", [
    "import ttl\n\n\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\n@ttl.compute()\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\ndef f(a, b, out):\n    pass\n\n\nOPERATION = ttl.operation(grid='full')(f)\n",
    "import ttl\n\n\ndef outer():\n    @ttl.operation(grid='full')\n    def OPERATION(a, b, out):\n        pass\n\n\nOPERATION = 1\n",
])
def test_operation_must_be_a_decorated_top_level_def(tmp_path, src):
    p = tmp_path / "kernel.py"
    p.write_text(src)
    ok, reason = CHECK.check_kernel_source(p)
    assert ok is False and "operation at line " in reason, reason
    assert {v["rule"] for v in CHECK.kernel_source_violations(p)} == {"operation"}


def test_a_bare_operation_decorator_is_allowed(tmp_path):
    p = tmp_path / "kernel.py"
    p.write_text("import ttl\n\n\n@ttl.operation\ndef OPERATION(a, b, out):\n    pass\n")
    assert CHECK.check_kernel_source(p) == (True, "")


@needs_sim
def test_a_forbidden_kernel_never_reaches_the_simulator(tmp_path):
    tdir, adir = _attempt(tmp_path, "forbidden_import")
    done, check = _run_check(tdir, adir)
    assert done.stdout.startswith("FAIL: forbidden_code: 1 violation: import at line ")
    assert [(v["rule"], v["line"]) for v in check["violations"]] == [("import", 7)]
    assert check["sets"] == [] and "PROGRESS" not in (adir / "harness-output.txt").read_text()
    assert not (adir / "harness-result.json").exists()


@needs_sim
def test_the_new_fixtures_fail_on_the_named_fields(tmp_path):
    """Each correlated-but-wrong kernel is failed by a bar other than PCC, and its PCC is above the PCC bar."""
    for kernel, op in (("add_scaled", "add"), ("rmsnorm_wrong_n", "rmsnorm")):
        tdir, adir = _attempt(tmp_path / kernel, kernel, op=op)
        _, check = _run_check(tdir, adir)
        assert check["category"] == "numeric_mismatch"
        assert all(s["pcc"] >= 0.999 for s in check["sets"]), check["sets"]
        assert any(f in check["first_line"] for f in ("rel_l2", "slope", "max_abs_err")), check["first_line"]
    tdir, adir = _attempt(tmp_path / "no_norm", "rmsnorm_no_norm", op="rmsnorm")
    _, check = _run_check(tdir, adir)
    plain = next(s for s in check["sets"] if s["name"] == "seed0")
    assert plain["pcc"] >= 0.999 and plain["rel_l2"] > 1e-2


@needs_sim
def test_check_json_records_the_bars_and_the_new_fields(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_good")
    _, check = _run_check(tdir, adir)
    assert check["bars"] == SIM_BARS
    for s in check["sets"]:
        assert s["rel_l2"] is not None and s["rel_l2"] <= 1e-2
        assert 0.98 <= s["slope"] <= 1.02
        assert s["exp_abs_max"] > 0


@needs_sim
def test_a_zero_timeout_is_a_timeout(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_good")
    _, check = _run_check(tdir, adir, "--timeout-s", "0")
    assert check["category"] == "timeout"


def _fake_stack(tmp_path):
    """A stand-in ttnn whose open_device raises, and a ttl whose decorators do nothing. No real ttnn is imported."""
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "ttnn.py").write_text(
        "TILE_LAYOUT = 'tile'\nbfloat16 = 'bf16'\nTensor = object\n"
        "def open_device(device_id=0):\n    raise RuntimeError('fake: device 0 is busy')\n"
        "def close_device(device):\n    pass\n")
    (fake / "ttl.py").write_text("def operation(*a, **k):\n    return lambda f: f\n")
    return fake


@needs_sim
def test_a_device_that_will_not_open_is_the_harness(tmp_path):
    fake = _fake_stack(tmp_path)
    out = tmp_path / "result.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(fake)
    done = subprocess.run([str(VENV / "bin" / "python"), str(TEMPLATES / "kernel_harness.py"), "--spec",
                           str(TEMPLATES / "ops" / "add.json"), "--kernel", str(KERNELS / "add_good.py"), "--out",
                           str(out), "--phase", "hw"], capture_output=True, text=True, timeout=120, env=env)
    result = json.loads(out.read_text())
    assert result["error"]["phase"] == "setup", result
    assert "device 0 is busy" in result["error"]["text"]
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=done.returncode,
                                 text=done.stdout + done.stderr, result=result, bar=0.99, phase="hw")
    assert category == "harness_error"


@needs_sim
def test_a_kernel_that_ends_the_process_is_blamed(tmp_path):
    """kernel_exit.py calls os._exit(0) inside OPERATION. The gate refuses it, so this runs the harness directly."""
    out = tmp_path / "result.json"
    done = subprocess.run([str(SIM), str(TEMPLATES / "kernel_harness.py"), "--", "--spec", str(TEMPLATES / "ops" / "add.json"),
                           "--kernel", str(KERNELS / "kernel_exit.py"), "--out", str(out), "--phase", "sim"],
                          capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert not out.exists()
    text = done.stdout + done.stderr
    assert "PROGRESS import" in text
    markers = [l for l in text.splitlines() if l.startswith("PROGRESS ")]
    assert markers[-1] == "PROGRESS run seed0", markers
    category, line = CHECK.classify(kernel_present=True, timed_out=False, returncode=done.returncode, text=text,
                                    result=None, bar=0.999)
    assert (category, line) == ("tt_lang_error", "the kernel ended the process without a result")


@needs_sim
def test_an_input_error_after_a_kernel_ran_is_still_the_harness(tmp_path):
    """The second seed cannot seed a generator. The kernel has already run once, and the fault is still not its own."""
    def edit(spec):
        spec["seeds"] = [0, "not-a-seed"]
    tdir, adir = _attempt(tmp_path, "add_good", spec_edit=edit)
    _, check = _run_check(tdir, adir)
    assert check["category"] == "harness_error", check
    assert [s["name"] for s in check["sets"]] == ["seed0", "seed0-tail"]



# --- fix round 2: every violation at once, harmless boilerplate, aliases, markers ------------------------------------

UPSTREAM = KERNELS / "upstream_eltwise_add.py"     # tt-lang examples/eltwise_add.py, copied byte for byte


def test_the_upstream_example_is_flagged_only_for_torch_and_readback():
    """The example the agent is told to read: its kernel is fine, so only `import torch` and the host code's
    from_torch/to_torch calls may be flagged, each with a remedy. `__name__` and the main block are not flagged."""
    found = CHECK.kernel_source_violations(UPSTREAM)
    assert [v["rule"] for v in found] == ["import", "readback", "readback", "readback", "readback"], found
    assert found[0]["line"] == 4 and "torch" in found[0]["detail"]
    assert all(v["remedy"] for v in found)
    assert all(v["remedy"] == "the harness reads the result back; remove the host code and keep only OPERATION"
               for v in found if v["rule"] == "readback")
    ok, reason = CHECK.check_kernel_source(UPSTREAM)
    assert ok is False and reason.startswith("5 violations: import at line 4: ")


def _src(tmp_path, text):
    p = tmp_path / "kernel.py"
    p.write_text(text)
    return p


def test_the_gate_reports_every_violation(tmp_path):
    p = _src(tmp_path, "import os\nimport sys\n" + GATE_HEADER + "    open('f')\n    x = a.__dict__\n")
    found = CHECK.kernel_source_violations(p)
    assert [(v["rule"], v["line"]) for v in found] == [("import", 1), ("import", 2), ("name", 9), ("dunder", 10)]
    assert all(set(v) >= {"rule", "line", "remedy"} for v in found)
    ok, reason = CHECK.check_kernel_source(p)
    assert reason.startswith("4 violations: import at line 1: ")
    assert reason.count(" at line ") == 4


def test_a_long_list_is_capped_at_eight(tmp_path):
    body = "".join(f"    open('f{i}')\n" for i in range(11))
    ok, reason = CHECK.check_kernel_source(_src(tmp_path, GATE_HEADER + body))
    assert reason.startswith("11 violations: ")
    assert reason.count(" at line ") == 8 and reason.endswith("and 3 more")


def test_every_rule_has_a_remedy():
    assert set(CHECK.REMEDIES) == {"import", "name", "dunder", "readback", "operation"}
    assert all(isinstance(r, str) and r for r in CHECK.REMEDIES.values())


@pytest.mark.parametrize("src", [
    "from __future__ import annotations\n" + GATE_HEADER + "    pass\n",
    "from typing import Any\n" + GATE_HEADER + "    pass\n",
    GATE_HEADER + "    pass\n\n\n__all__ = ['OPERATION']\n",
    GATE_HEADER + "    pass\n\n\nif __name__ == '__main__':\n    print('host code')\n",
])
def test_harmless_boilerplate_passes_the_gate(tmp_path, src):
    assert CHECK.check_kernel_source(_src(tmp_path, src)) == (True, "")


@pytest.mark.parametrize("line", ["    input = a", "    y = a\n    input_blk = y", "    x = input if False else a"])
def test_input_is_a_legal_variable_name(tmp_path, line):
    assert CHECK.check_kernel_source(_src(tmp_path, GATE_HEADER + line + "\n")) == (True, "")


def test_input_as_a_parameter_name_is_legal(tmp_path):
    src = 'import ttl\n\n\n@ttl.operation(grid="full")\ndef OPERATION(input, b, out):\n    x = input\n'
    assert CHECK.check_kernel_source(_src(tmp_path, src)) == (True, "")


@pytest.mark.parametrize("src", [
    "import ttl as tl\n\n\n@tl.operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl as tl\n\n\n@tl.operation\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttl import operation\n\n\n@operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttl import operation as op\n\n\n@op(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttl import operation as op\n\n\n@op\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttnn as tn\nimport ttl\n\n\n@ttl.operation(grid='full')\ndef OPERATION(a: tn.Tensor, b, out):\n    pass\n",
])
def test_the_legal_aliases_pass_the_gate(tmp_path, src):
    assert CHECK.check_kernel_source(_src(tmp_path, src)) == (True, "")


@pytest.mark.parametrize("src", [
    "import ttnn\n\n\n@ttnn.operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttnn as tl\n\n\n@tl.operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\n@ttl.operation_like(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\n@ttl.operations\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\n@my_operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttnn import operation\n\n\n@operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttl import operation\n\n\ndef operation(**k):\n    return lambda f: f\n\n\n@operation(grid='full')\n"
    "def OPERATION(a, b, out):\n    pass\n",
    "import ttl as tl\n\ntl = None\n\n\n@tl.operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttl import operation as op\nop = 1\n\n\n@op(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\n@ttl.compute.operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
])
def test_a_decorator_that_is_not_ttl_operation_is_refused(tmp_path, src):
    found = CHECK.kernel_source_violations(_src(tmp_path, src))
    assert "operation" in {v["rule"] for v in found}, found


@pytest.mark.parametrize("src", [
    "from ttnn import to_torch\n" + GATE_HEADER + "    pass\n",
    "from ttnn import from_torch as ft\n" + GATE_HEADER + "    pass\n",
    "from ttl import from_torch\n" + GATE_HEADER + "    pass\n",
    GATE_HEADER + "    to_torch(a)\n",
])
def test_readback_names_are_refused_however_they_arrive(tmp_path, src):
    assert {v["rule"] for v in CHECK.kernel_source_violations(_src(tmp_path, src))} == {"readback"}


@pytest.mark.parametrize("src", ["from ttnn import *\n" + GATE_HEADER + "    pass\n",
                                 "from ttl import *\n" + GATE_HEADER + "    pass\n"])
def test_star_imports_are_refused(tmp_path, src):
    assert {v["rule"] for v in CHECK.kernel_source_violations(_src(tmp_path, src))} == {"import"}


@pytest.mark.parametrize("tail", [
    "for OPERATION in [OPERATION]:\n    pass\n",
    "with open_ctx() as OPERATION:\n    pass\n",
    "if (OPERATION := OPERATION):\n    pass\n",
    "x = [(OPERATION := y) for y in [1]]\n",
    "if True:\n    OPERATION = 1\n",
    "del OPERATION\n",
    "try:\n    pass\nexcept Exception as OPERATION:\n    pass\n",
    "if True:\n    @ttl.operation(grid='full')\n    def OPERATION(a, b, out):\n        pass\n",
    "import ttl as OPERATION\n",
    "class OPERATION:\n    pass\n",
    "def f():\n    global OPERATION\n    OPERATION = 1\n",
])
def test_operation_bound_any_other_way_is_refused(tmp_path, tail):
    found = CHECK.kernel_source_violations(_src(tmp_path, GATE_HEADER + "    pass\n\n\n" + tail))
    assert [v["rule"] for v in found] == ["operation"], found


@pytest.mark.parametrize("decorators", ["@staticmethod\n@ttl.operation(grid='full')\n",
                                        "@ttl.operation(grid='full')\n@staticmethod\n",
                                        "@ttl.operation(grid='full')\n@ttl.operation(grid='full')\n"])
def test_a_stacked_decorator_is_refused(tmp_path, decorators):
    """Another decorator above or below ttl.operation, or ttl.operation twice: OPERATION would not be the
    operation ttl built."""
    src = "import ttl\n\n\n" + decorators + "def OPERATION(a, b, out):\n    pass\n"
    found = CHECK.kernel_source_violations(_src(tmp_path, src))
    assert [v["rule"] for v in found] == ["operation"]


def test_a_local_name_operation_inside_a_function_is_fine(tmp_path):
    src = GATE_HEADER + "    OPERATION = 1\n"          # a local inside OPERATION's body binds nothing at module scope
    assert CHECK.check_kernel_source(_src(tmp_path, src)) == (True, "")


@pytest.mark.parametrize("text,blamed", [
    ("PROGRESS import\n", True),
    ("PROGRESS import\nPROGRESS open_device\n", False),
    ("PROGRESS import\nPROGRESS open_device\nPROGRESS set seed0\n", False),
    ("PROGRESS import\nPROGRESS open_device\nPROGRESS set seed0\nPROGRESS run seed0\n", True),
    ("PROGRESS import\nPROGRESS open_device\nPROGRESS set seed0\nPROGRESS run seed0\nsome kernel print\n", True),
    ("PROGRESS import\nPROGRESS run seed0\nPROGRESS set seed0-tail\n", False),
    ("", False),
])
def test_the_last_marker_decides_who_ended_the_process(text, blamed):
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=None, bar=0.999)
    assert category == ("tt_lang_error" if blamed else "harness_error")


def test_a_nan_pcc_fails():
    category, line = _classify_sets([_good_set(pcc=float("nan"))], bars=SIM_BARS)
    assert category == "numeric_mismatch" and "pcc nan" in line


def _venv_with_diagnostics(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    sim = venv / "bin" / "tt-lang-sim"
    sim.write_text("#!/bin/sh\n")
    sim.chmod(0o755)
    pkg = venv / "lib" / "python3.12" / "site-packages" / "ttl" / "ttl"
    pkg.mkdir(parents=True)
    (pkg / "diagnostics.py").write_text("")
    return sim


def test_diagnostics_lookup_follows_a_symlink(tmp_path):
    sim = _venv_with_diagnostics(tmp_path)
    links = tmp_path / "links"
    links.mkdir()
    (links / "tt-lang-sim").symlink_to(sim)
    assert CHECK.diagnostics_present(links / "tt-lang-sim") is True


def test_diagnostics_lookup_finds_a_bare_command_on_path(tmp_path, monkeypatch):
    sim = _venv_with_diagnostics(tmp_path)
    monkeypatch.setenv("PATH", str(sim.parent))
    assert CHECK.diagnostics_present("tt-lang-sim") is True
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert CHECK.diagnostics_present("tt-lang-sim") is None


@pytest.mark.parametrize("kernel", ["add_alias_module", "add_alias_ttnn", "add_alias_function", "add_alias_function_as",
                                    "add_boilerplate"])
@needs_sim
def test_the_alias_and_boilerplate_kernels_pass_on_the_simulator(tmp_path, kernel):
    """add_boilerplate's main block raises SystemExit; it passes only because the harness never runs that block."""
    tdir, adir = _attempt(tmp_path, kernel)
    done, check = _run_check(tdir, adir)
    assert done.returncode == 0, done.stdout + done.stderr
    assert check["passed"] is True and check["violations"] == []


@needs_sim
def test_check_json_lists_every_violation(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_good")
    shutil.copy(UPSTREAM, adir / "kernel.py")
    done, check = _run_check(tdir, adir)
    assert check["category"] == "forbidden_code"
    assert check["first_line"].startswith("5 violations: ")
    assert [v["rule"] for v in check["violations"]] == ["import"] + ["readback"] * 4
