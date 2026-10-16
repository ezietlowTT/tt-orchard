#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Check an agent-written kernel.py against its op's reference on the tt-lang simulator.

    python3 kernel_check.py [--attempt-dir DIR] [--phase sim] [--spec FILE] [--sim PATH] [--timeout-s N]
                            [--tt-lang-json JSON] [--out FILE]

Defaults come from DIR/check_config.json: {"sim": "<tt-lang-sim>", "timeout_s": 300, "tt_lang": {...}}. It runs
kernel_harness.py from the directory THIS file is in, so the harness that judges a kernel is the copy next to the check
that was run. Exit 0 and `PASS` when the kernel passed; exit 1 and `FAIL: <category>: <first line>` when it did not;
exit 2 on a usage error. It writes check.json with sha256 values of the files that took part and the bars it applied.

Before the simulator runs, a static gate (`check_kernel_source`) reads kernel.py with `ast` and refuses the shortcuts a
model is likely to try: importing anything but ttl, ttnn and math, calling open/exec/eval and similar, naming dunder
attributes, naming ttnn's to_torch/from_torch, and an OPERATION that is not a top-level def decorated with
ttl.operation. A refused kernel ends `forbidden_code` and never runs. The gate is best effort. The kernel still runs in
the harness's own process, so a determined kernel can find a way around a static check; the gate stops the easy ways.

A kernel passes only when every input set is inside every bar: PCC, relative L2 error, least-squares slope, and
max_abs_err <= atol + rtol * max|expected|. PCC alone does not change when the output is scaled, so a kernel off by a
constant factor would pass on PCC. A missing or null field fails the set.

The category comes from the diagnostic text the simulator prints and the harness's result file, never from an exit code
alone. (The tt-lang-sim wheel without its diagnostics file ends every diagnosed error as FileNotFoundError with a
non-zero exit. That case is reported as harness_error, because it is a fault in the environment. The check looks for
the file next to the simulator, so a kernel that prints the same text cannot use it to shift the blame.)
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPE_RE = re.compile(r"Tensor shape .*does not match Block shape")
STARTED_MARK = "PROGRESS import"      # the harness prints this just before it loads kernel.py

# The provisional bars the op specs ship with. classify falls back to these when its caller passes no `bars`.
DEFAULT_BARS = {
    "sim": {"pcc": 0.999, "rel_l2": 1e-2, "slope": [0.98, 1.02], "max_abs_atol": 1e-2, "max_abs_rtol": 1e-2},
    "hw": {"pcc": 0.99, "rel_l2": 5e-2, "slope": [0.95, 1.05], "max_abs_atol": 1e-1, "max_abs_rtol": 5e-2},
}

# The static gate's tables.
ALLOWED_IMPORTS = {"ttl", "ttnn", "math"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "setattr", "getattr",
                   "delattr", "vars", "breakpoint", "input", "sys", "os", "atexit", "builtins", "subprocess",
                   "importlib", "references"}
READBACK_ATTRS = {"to_torch", "from_torch"}


def sha256_file(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def bars_from_spec(spec: dict, phase: str) -> dict:
    """The bars for one phase, read from an op spec. A missing key raises KeyError."""
    return {"pcc": spec[f"pcc_bar_{phase}"], "rel_l2": spec[f"rel_l2_bar_{phase}"],
            "slope": list(spec[f"slope_band_{phase}"]), "max_abs_atol": spec[f"max_abs_atol_{phase}"],
            "max_abs_rtol": spec[f"max_abs_rtol_{phase}"]}


def diagnostics_present(sim) -> bool | None:
    """Whether tt-lang-sim's diagnostics.py exists in the venv the simulator lives in.

    None when no ttl package is found next to the simulator (the layout is not a venv this code knows); classify then
    falls back to reading the text."""
    venv = Path(sim).parent.parent
    packages = list(venv.glob("lib/python*/site-packages/ttl"))
    if not packages:
        return None
    return any((p / "sim" / "diagnostics.py").is_file() for p in packages)


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _is_ttl_operation(dec: ast.expr) -> bool:
    """`@ttl.operation` or `@ttl.operation(...)`."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    return (isinstance(target, ast.Attribute) and target.attr == "operation" and isinstance(target.value, ast.Name)
            and target.value.id == "ttl")


def _binds_operation(node: ast.stmt) -> bool:
    """A top-level statement other than a def that binds the name OPERATION."""
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return any(isinstance(n, ast.Name) and n.id == "OPERATION" for t in targets for n in ast.walk(t))
    if isinstance(node, ast.ClassDef):
        return node.name == "OPERATION"
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return any((a.asname or a.name) == "OPERATION" for a in node.names)
    return False


def check_kernel_source(path) -> tuple[bool, str]:
    """The static gate. (True, "") when kernel.py breaks no rule; otherwise (False, "<rule> at line N: <detail>").

    Rules: `import` (only ttl, ttnn and math), `name` (FORBIDDEN_NAMES and any dunder name), `dunder` (an attribute
    whose name starts and ends with two underscores), `readback` (the attributes to_torch and from_torch), `operation`
    (OPERATION must be a top-level def decorated with ttl.operation, and nothing else may bind that name). A file with
    no OPERATION at all gives (False, "no-operation: ..."), which the check reports as no_kernel.

    A file that does not parse passes the gate, so the harness reports its SyntaxError as import_error. Best effort:
    this reads the source, so it cannot see what code builds at run time."""
    source = Path(path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True, ""
    found = []                                            # (line, rule, detail)
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    found.append((line, "import", f"import of {alias.name} (allowed: ttl, ttnn, math)"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or module.split(".")[0] not in ALLOWED_IMPORTS:
                found.append((line, "import", f"import from {'.' * node.level}{module} (allowed: ttl, ttnn, math)"))
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or _is_dunder(node.id):
                found.append((line, "name", f"use of {node.id}"))
        elif isinstance(node, ast.Global) and "OPERATION" in node.names:
            found.append((line, "operation", "OPERATION is rebound through a global statement"))
        elif isinstance(node, ast.Attribute):
            if _is_dunder(node.attr):
                found.append((line, "dunder", f"attribute {node.attr}"))
            elif node.attr in READBACK_ATTRS:
                found.append((line, "readback", f"attribute {node.attr}; the harness reads the output back"))
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "OPERATION"]
    others = [n for n in tree.body if _binds_operation(n)]
    for n in others:
        found.append((n.lineno, "operation", "OPERATION is bound by something other than a decorated def"))
    for n in defs:
        if not any(_is_ttl_operation(d) for d in n.decorator_list):
            found.append((n.lineno, "operation", "OPERATION is not decorated with ttl.operation"))
    if found:
        line, rule, detail = min(found)
        return False, f"{rule} at line {line}: {detail}"
    if not defs:
        return False, "no-operation: kernel.py defines no OPERATION"
    return True, ""


def first_line(text: str) -> str:
    """The simulator's first diagnostic line: `error: ...`, a deadlock notice, or the `Error in kernel` header."""
    lines = [l.strip() for l in text.splitlines()]
    for l in lines:
        if l.startswith("error:") or l.startswith("Deadlock detected"):
            return l[:200]
    for l in lines:
        if "Error in kernel" in l:
            return l[:200]
    return ""


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _fmt(v: float) -> float:
    """A value short enough for a FAIL line."""
    return round(v, 6)


def _set_problem(s: dict, bar: float, bars: dict) -> str | None:
    """Why one input set fails, or None when it is inside every bar."""
    pcc, finite = s.get("pcc"), s.get("finite")
    if pcc is None or finite is not True or pcc < bar:
        return f"pcc {pcc}, finite {finite}, bar {bar}"
    for field in ("rel_l2", "slope", "max_abs_err", "exp_abs_max"):
        if not _number(s.get(field)):
            return f"{field} missing or not a number ({s.get(field)!r})"
    if s["rel_l2"] > bars["rel_l2"]:
        return f"rel_l2 {_fmt(s['rel_l2'])} above bar {bars['rel_l2']}"
    lo, hi = bars["slope"]
    if not lo <= s["slope"] <= hi:
        return f"slope {_fmt(s['slope'])} outside [{lo}, {hi}]"
    limit = bars["max_abs_atol"] + bars["max_abs_rtol"] * s["exp_abs_max"]
    if s["max_abs_err"] > limit:
        return (f"max_abs_err {_fmt(s['max_abs_err'])} above {bars['max_abs_atol']} + {bars['max_abs_rtol']} * "
                f"{_fmt(s['exp_abs_max'])} = {_fmt(limit)}")
    return None


def classify(*, kernel_present, timed_out, returncode, text, result, bar, phase="sim", bars=None,
             diagnostics_present=None):
    """(category or None, first line). None means the kernel passed.

    `bar` is the PCC bar. `bars` holds the other bars (bars_from_spec); without it the phase's DEFAULT_BARS apply.
    `diagnostics_present` is what the check measured next to the simulator: True means a missing-diagnostics message
    in the text cannot be the environment's (the kernel printed or raised it), False means it is, None means unknown
    and the text decides."""
    numeric = "hw_numeric_mismatch" if phase == "hw" else "numeric_mismatch"
    runtime = "compile_error" if phase == "hw" else "tt_lang_error"
    bars = bars or DEFAULT_BARS[phase]
    if not kernel_present:
        return "no_kernel", "kernel.py does not exist"
    if timed_out:
        return "timeout", "no result within the time allowed"
    err = (result or {}).get("error")
    blob = text + "\n" + ((err or {}).get("text") or "")
    if diagnostics_present is not True and "FileNotFoundError" in blob and "diagnostics.py" in blob:
        return "harness_error", "tt-lang-sim's diagnostics.py is missing; run `tt-orchard kernel-env`"
    if err and err.get("type") == "NoOperation":
        return "no_kernel", "kernel.py defines no OPERATION"
    if err and err.get("phase") == "setup":         # the spec, the references, the device or the inputs
        first = ((err.get("text") or "").splitlines() or [""])[0]
        return "harness_error", f"{err.get('type')}: {first}"[:200]
    if err and err.get("phase") == "import":
        first = ((err.get("text") or "").splitlines() or [""])[0]
        return "import_error", f"{err.get('type')}: {first}"[:200]
    m = SHAPE_RE.search(blob)
    if m:
        return "shape_mismatch", first_line(blob) or m.group(0)[:200]
    if err or "Deadlock detected" in blob:
        return runtime, first_line(blob) or ((err or {}).get("text") or "")[:200]
    if result is None:
        # After the harness started loading the kernel, only the kernel can end the process without a result.
        # (A crash of the harness itself after that point would also land here.)
        if STARTED_MARK in text:
            return runtime, "the kernel ended the process without a result"
        return "harness_error", f"the harness ended with exit {returncode} and wrote no result"
    sets = result.get("sets", [])
    if not sets:
        return "harness_error", "the harness ran no input sets"
    for s in sets:
        problem = _set_problem(s, bar, bars)
        if problem:
            return numeric, f"set {s.get('name')}: {problem}"
    return None, ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt-dir", default=".")
    ap.add_argument("--phase", choices=("sim",), default="sim")
    ap.add_argument("--spec")
    ap.add_argument("--sim")
    ap.add_argument("--timeout-s", type=float)
    ap.add_argument("--tt-lang-json")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    adir = Path(a.attempt_dir).resolve()
    cfg_path = adir / "check_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    sim = a.sim or cfg.get("sim")
    timeout_s = a.timeout_s if a.timeout_s is not None else cfg.get("timeout_s")
    timeout_s = 300 if timeout_s is None else timeout_s
    spec_path = Path(a.spec) if a.spec else adir / "kernel_spec.json"
    if not sim or not spec_path.is_file():
        print("usage error: need --sim (or check_config.json) and a spec", file=sys.stderr)
        return 2
    tt_lang = json.loads(a.tt_lang_json) if a.tt_lang_json else cfg.get("tt_lang", {})
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    kernel, result_path, log_path = adir / "kernel.py", adir / "harness-result.json", adir / "harness-output.txt"
    result_path.unlink(missing_ok=True)
    timed_out, rc, text = False, None, ""
    start = time.monotonic()
    category = line = None
    bars = None
    try:
        bars = bars_from_spec(spec, "sim")
    except KeyError as exc:
        category, line = "harness_error", f"the spec has no {exc.args[0]}"
    if category is None and kernel.is_file():
        ok, reason = check_kernel_source(kernel)
        if not ok:
            category, line = ("no_kernel", "kernel.py defines no OPERATION") if reason.startswith("no-operation") \
                else ("forbidden_code", reason)
    if category is None and kernel.is_file():
        try:
            done = subprocess.run([sim, str(HERE / "kernel_harness.py"), "--", "--spec", str(spec_path), "--kernel",
                                   str(kernel), "--out", str(result_path), "--phase", "sim"],
                                  capture_output=True, text=True, timeout=timeout_s, cwd=adir)
            rc, text = done.returncode, (done.stdout or "") + "\n" + (done.stderr or "")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            text = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    log_path.write_text(text, encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    if category is None:
        category, line = classify(kernel_present=kernel.is_file(), timed_out=timed_out, returncode=rc, text=text,
                                  result=result, bar=spec["pcc_bar_sim"], phase="sim", bars=bars,
                                  diagnostics_present=diagnostics_present(sim))
    check = {"schema": 1, "op": spec["op"], "phase": "sim", "passed": category is None, "category": category,
             "first_line": line, "sets": (result or {}).get("sets", []), "pcc_bar": spec["pcc_bar_sim"], "bars": bars,
             "provisional": True, "elapsed_s": round(time.monotonic() - start, 2), "tt_lang": tt_lang,
             "sha256": {"kernel.py": sha256_file(kernel), "kernel_harness.py": sha256_file(HERE / "kernel_harness.py"),
                        "references.py": sha256_file(HERE / "references.py"), "kernel_check.py": sha256_file(Path(__file__)),
                        "spec": sha256_file(spec_path)}}
    Path(a.out or adir / "check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print("PASS" if category is None else f"FAIL: {category}: {line}")
    return 0 if category is None else 1


if __name__ == "__main__":
    sys.exit(main())
