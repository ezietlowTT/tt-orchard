---
name: kernel-probe
status: draft. Written for the kernel probe (spec 2026-10-15-kernel-attempt-design.md, section 5). Not yet run with a real coder.
---
# Write a tt-lang kernel

You are in a directory that holds `kernel_spec.json` (the op), `kernel_check.py` (the check), `reference/eltwise_add.py` (a
working tt-lang example) and a few support files. You write one file: `kernel.py`.

1. Read `kernel_spec.json`. It lists the inputs and their shapes, the output shape, and what the op computes.
2. Read `reference/eltwise_add.py` once. It shows how a tt-lang operation is declared (`@ttl.operation(grid="full")`), how
   dataflow buffers carry tiles between the reader, compute and writer functions, and how a result is stored.
3. Write `kernel.py`. It must define `OPERATION`, a function decorated with `@ttl.operation` that takes the input tensors in the
   order `kernel_spec.json` lists them, then the output tensor last. At the top level of `kernel.py` put only imports,
   constants and that function.
4. Run `python3 kernel_check.py`. It prints `PASS`, or `FAIL: <category>: <first line>`. The first line is the simulator's own
   message.
5. Read that line. Change one thing. Run it again.

How to write `kernel.py` from the example:

- Keep the three thread functions inside `OPERATION`, as the example has them: a reader decorated with
  `@ttl.datamovement()`, the compute function decorated with `@ttl.compute()`, and a writer decorated with
  `@ttl.datamovement()`. An `OPERATION` without them does not run on the simulator (it fails with
  `ModuleNotFoundError: ttl.dfb_reset`).
- Delete the example's `main()` and the `if __name__ == "__main__":` block. The check opens the device, builds the
  inputs and reads the result back. Your file holds only the operation.

The check reads `kernel.py` before it runs it, and refuses the file with `FAIL: forbidden_code: ...` when it breaks one of
these rules. It prints every rule the file breaks, with the line and what to do instead.

- Import only `ttl, ttnn and math` (`from __future__ import ...` and `from typing import ...` are also allowed). The
  example's `import torch` is host code: do not import torch.
- Do not use `ttnn.from_torch`, `ttnn.to_torch` or `ttnn.open_device`. They belong to the host code you deleted.
- Define `OPERATION` once, as a plain top-level `def` with one decorator, `@ttl.operation(...)`. The forms
  `import ttl as tl` with `@tl.operation(...)`, and `from ttl import operation` with `@operation(...)`, are accepted.
- Do not use `open`, `exec`, `eval`, `compile`, `getattr`, `setattr`, `globals`, `sys`, `os`, `subprocess`,
  `importlib` or names that start and end with two underscores. A kernel does not need them.

Rules:
- Write only `kernel.py`. Do not edit any other file. Your result is judged afterwards by an unedited copy of the check.
- Do not install packages, use the network, or touch any device. The check runs on a simulator.
- Do not read the ledger or the log directory.
- If the same failure comes back three times and you have no new idea, stop and say what you tried.
- Finish with at most five lines: what the kernel does, and what the last check said.
