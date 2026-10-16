#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Check an agent-written kernel.py against its op's reference on the tt-lang simulator.

    python3 kernel_check.py [--attempt-dir DIR] [--phase sim] [--spec FILE] [--sim PATH] [--timeout-s N]
                            [--tt-lang-json JSON] [--out FILE]

Defaults come from DIR/check_config.json: {"sim": "<tt-lang-sim>", "timeout_s": 300, "tt_lang": {...}}. It runs
kernel_harness.py from the directory THIS file is in, so the harness that judges a kernel is the copy next to the check
that was run. Exit 0 and `PASS` when the kernel passed; exit 1 and `FAIL: <category>: <first line>` when it did not;
exit 2 on a usage error. It writes check.json with sha256 values of the files that took part.

The category comes from the diagnostic text the simulator prints, never from an exit code alone. (The tt-lang-sim
wheel without its diagnostics file ends every diagnosed error as FileNotFoundError with a non-zero exit; that case is
reported as harness_error, because it is a fault in the environment.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPE_RE = re.compile(r"Tensor shape .*does not match Block shape")


def sha256_file(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


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


def classify(*, kernel_present, timed_out, returncode, text, result, bar, phase="sim"):
    """(category or None, first line). None means the kernel passed."""
    numeric = "hw_numeric_mismatch" if phase == "hw" else "numeric_mismatch"
    runtime = "compile_error" if phase == "hw" else "tt_lang_error"
    if not kernel_present:
        return "no_kernel", "kernel.py does not exist"
    if timed_out:
        return "timeout", "no result within the time allowed"
    err = (result or {}).get("error")
    blob = text + "\n" + ((err or {}).get("text") or "")
    if "FileNotFoundError" in blob and "diagnostics.py" in blob:
        return "harness_error", "tt-lang-sim's diagnostics.py is missing; run `tt-orchard kernel-env`"
    if err and err.get("type") == "NoOperation":
        return "no_kernel", "kernel.py defines no OPERATION"
    if err and err.get("phase") == "setup":         # the spec, the references or the environment, before kernel.py loads
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
        return "harness_error", f"the harness ended with exit {returncode} and wrote no result"
    sets = result.get("sets", [])
    if not sets:
        return "harness_error", "the harness ran no input sets"
    bad = [s for s in sets if s["pcc"] is None or not s["finite"] or s["pcc"] < bar]
    if bad:
        s = bad[0]
        return numeric, f"set {s['name']}: pcc {s['pcc']}, finite {s['finite']}, bar {bar}"
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
    timeout_s = a.timeout_s or cfg.get("timeout_s") or 300
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
    if kernel.is_file():
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
    category, line = classify(kernel_present=kernel.is_file(), timed_out=timed_out, returncode=rc, text=text,
                              result=result, bar=spec["pcc_bar_sim"], phase="sim")
    check = {"schema": 1, "op": spec["op"], "phase": "sim", "passed": category is None, "category": category,
             "first_line": line, "sets": (result or {}).get("sets", []), "pcc_bar": spec["pcc_bar_sim"],
             "provisional": True, "elapsed_s": round(time.monotonic() - start, 2), "tt_lang": tt_lang,
             "sha256": {"kernel.py": sha256_file(kernel), "kernel_harness.py": sha256_file(HERE / "kernel_harness.py"),
                        "references.py": sha256_file(HERE / "references.py"), "kernel_check.py": sha256_file(Path(__file__)),
                        "spec": sha256_file(spec_path)}}
    Path(a.out or adir / "check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print("PASS" if category is None else f"FAIL: {category}: {line}")
    return 0 if category is None else 1


if __name__ == "__main__":
    sys.exit(main())
