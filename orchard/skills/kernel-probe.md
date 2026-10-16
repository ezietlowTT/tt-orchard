---
name: kernel-probe
status: draft. For the kernel probe (spec 2026-10-15-kernel-attempt-design.md, section 5). Not yet run with a real coder.
---
# Write a tt-lang kernel

Your directory holds `kernel_spec.json` (the op), `kernel_check.py` (the check), `reference/eltwise_add.py` (a working
tt-lang example) and a few support files. You write one file: `kernel.py`.

Steps:
1. Read `kernel_spec.json`. It lists the inputs, their shapes, the output shape, and what the op computes.
2. Read `reference/eltwise_add.py` once. It shows `@ttl.operation(grid="full")`, the dataflow buffers, and the reader,
   compute and writer functions.
3. Write `kernel.py` with write_file. Define `OPERATION`: it takes the inputs in the order `kernel_spec.json` lists
   them, then the output tensor last.
4. Run `python3 kernel_check.py`. It prints `PASS`, or `FAIL: <category>: <first line>`.
5. Read that line. Change one thing. Run it again.

Advice:
- Copy the example's three thread functions into `OPERATION`: a reader with `@ttl.datamovement()`, a compute
  function with `@ttl.compute()`, and a writer with `@ttl.datamovement()`. Without them the simulator fails with
  `ModuleNotFoundError: ttl.dfb_reset`.
- Delete the example's `main()` and its `if __name__ == "__main__":` block. Do not open a device: the check opens it,
  builds the inputs and reads the result back.

Rules the check enforces. It reads `kernel.py` before running it and answers `FAIL: forbidden_code: ...` with every
rule broken, its line, and what to do instead:
- Imports: import only `ttl`, `ttnn` and `math`. `from __future__ import ...` and `from typing import ...` are also
  allowed. Do not import torch. A star import (`from X import *`) is refused.
- Forbidden names: `open`, `exec`, `eval`, `compile`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`,
  `sys`, `os`, `subprocess`, `importlib`, `builtins`, `breakpoint`, `atexit`, `references`, and calling `input()`.
- Names and attributes with double underscores are refused, except `__name__` and `__all__`.
- Read-back: `to_torch` and `from_torch` are refused, as a name, an attribute or an import.
- `OPERATION`: one plain top-level `def`, with one decorator, `@ttl.operation(...)`. `import ttl as tl` with
  `@tl.operation(...)`, and `from ttl import operation` with `@operation(...)`, are accepted.

The probe also refuses a `kernel.py` that is a link or larger than 256 KiB, when it takes the file after you finish.
Write it with write_file as a plain file.

Other rules:
- Write only `kernel.py`. Your result is judged afterwards by an unedited copy of the check.
- Do not install packages, use the network, or touch any device. The check runs on a simulator.
- Do not read the ledger or the log directory.
- If the same failure comes back three times and you have no new idea, stop and say what you tried.
- Finish with at most five lines: what the kernel does, and what the last check said.
