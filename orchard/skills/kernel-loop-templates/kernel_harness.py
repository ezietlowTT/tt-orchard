#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run an agent-written tt-lang kernel on its op's inputs and write result.json.

    tt-lang-sim kernel_harness.py -- --spec FILE --kernel FILE --out FILE [--phase sim|hw]
    python kernel_harness.py --spec FILE --kernel FILE --out FILE --phase hw      (real stack, under a lease)

The harness opens the device, builds the inputs and reads the output back, so the agent writes only the kernel. It never evaluates anything the spec names: the reference is a key into references.REFERENCES.
Prints `PROGRESS <what>` lines so a watcher can see it is alive.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))      # references.py sits next to this file; tt-lang-sim does not add the script's directory


def progress(msg: str) -> None:
    print(f"PROGRESS {msg}", flush=True)


def make_inputs(spec: dict, seed: int, heavy_tail: bool, torch) -> dict:
    g = torch.Generator().manual_seed(seed)
    ins = {}
    for name, shape in spec["inputs"].items():
        kind = spec.get("gen", {}).get(name, "normal")
        x = torch.randn(shape, generator=g)
        if kind == "sigmoid":
            x = torch.sigmoid(x)
        elif heavy_tail:                                  # a norm on unit-normal inputs is easy: add outliers
            x = torch.where(torch.rand(shape, generator=g) < 0.01, x * 20.0, x)
        ins[name] = x.to(torch.bfloat16)
    return ins


def pcc(got, exp, torch):
    g, e = got.flatten().float(), exp.flatten().float()
    if g.numel() < 2 or float(g.std()) == 0.0 or float(e.std()) == 0.0:
        return None                                       # a constant output has no correlation: counted as a failure
    value = float(torch.corrcoef(torch.stack([g, e]))[0, 1])
    return None if value != value else value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("sim", "hw"), default="sim")
    # tt-lang-sim 1.1.10 hands the `--` that separates its own options from the script's on to the script, and
    # argparse would then read every later flag as a positional. Drop that one leading separator.
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--"]:
        args = args[1:]
    a = ap.parse_args(args)
    result = {"phase": a.phase, "sets": [], "error": None}
    # The phase names who is at fault: `setup` (the spec, the references or the environment, before the kernel is
    # touched), `import` (loading kernel.py) and `run` (calling OPERATION).
    stage = "setup"
    try:
        spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
        import torch
        import ttnn
        from references import REFERENCES
        if spec["reference"] not in REFERENCES:
            raise KeyError(f"reference {spec['reference']!r} is not in references.REFERENCES")
        stage = "import"
        loader = importlib.util.spec_from_file_location("kernel_under_test", a.kernel)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        operation = getattr(module, "OPERATION", None)
        if operation is None:
            result["error"] = {"phase": "import", "type": "NoOperation", "text": "kernel.py defines no OPERATION"}
        else:
            stage = "run"
            progress("open_device")
            device = ttnn.open_device(device_id=0)
            try:
                for seed in spec["seeds"]:
                    for tail in ([False, True] if spec.get("heavy_tail") else [False]):
                        name = f"seed{seed}{'-tail' if tail else ''}"
                        ins = make_inputs(spec, seed, tail, torch)
                        expected = REFERENCES[spec["reference"]]({k: v.float() for k, v in ins.items()})
                        progress(f"set {name}")
                        tt_ins = [ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                                  for v in ins.values()]
                        out = ttnn.from_torch(torch.zeros(spec["output"], dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                              layout=ttnn.TILE_LAYOUT, device=device)
                        operation(*tt_ins, out)
                        got = ttnn.to_torch(out).float()
                        err = (got - expected).abs().max()
                        result["sets"].append({"name": name, "seed": seed, "heavy_tail": tail, "pcc": pcc(got, expected, torch),
                                               "max_abs_err": float(err), "finite": bool(torch.isfinite(got).all())})
            finally:
                ttnn.close_device(device)
    except BaseException as exc:                          # the diagnostic text must reach the check, so record it and print it
        traceback.print_exc()
        result["error"] = {"phase": stage, "type": type(exc).__name__, "text": str(exc)[:2000]}
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
