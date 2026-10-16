---
name: kernel-probe
status: draft. For the kernel probe (spec 2026-10-15-kernel-attempt-design.md, section 5 and section 11, item 28).
---
# Write a tt-lang kernel

Your directory holds `kernel_spec.json` (the op), `kernel_check.py` (the check), reference files in `reference/` and a
few support files. You write one file: `kernel.py`.

Reference files:
- `reference/eltwise_add.py`: a working kernel for the tt-lang the check runs. Start here.
- `reference/eltwise_broadcast_reduce.py`: a simulator example with `ttl.math.reduce_sum`, `ttl.block.broadcast` and
  `ttl.block.fill`. It passes in the simulator. tt-lang's CI marks it `TTLANG_HARDWARE_CI: xfail-compiler`: it is
  expected to fail on the hardware compiler.
- `reference/TTLangSpecification.md`: the specification of that tt-lang (version 0.20). It is long: search it with
  `grep -n "reduce_sum" reference/TTLangSpecification.md` and read the lines around a hit.
- `reference/guide.md` and `reference/examples.md`: the tt-lang skill's guide and worked examples. They were written
  for specification 0.12, so take every name and signature from the two .py files and the specification.

Read first, by kind of op:
- Elementwise (`add`, `silu_mul`): eltwise_add.py, then the specification's row for each function you need
  (`grep -n "ttl.math.silu" reference/TTLangSpecification.md`).
- A reduction (`rmsnorm`): eltwise_broadcast_reduce.py, then the specification's rows for `ttl.math.reduce_sum`,
  `ttl.block.broadcast` and `ttl.block.fill`.
- A recurrence along a dimension (`linear_scan`): both .py files, then the specification's "Block" section. No upstream
  example has a recurrence.

Rules from the specification and the examples:
- A block is used whole. On a compute thread a block is an operand of a block expression (Python operators and
  `ttl.math` functions), and `blk.store(expr)` writes an expression into a block whose shape it matches exactly. On a
  data movement thread a block is the source or destination of `ttl.copy`. Do not index a block element by element
  (`x_blk[i, j]`) or loop over its elements.
- The decorator is `@ttl.operation(grid=...)`. The guide's `@ttl.kernel` is the old name.
- `ttl.make_dataflow_buffer_like(tensor, shape=(R, C), block_count=2)`; the shape counts tiles. The guide's
  `buffer_factor` is the old name.
- `ttl.math.reduce_sum(expr, dims=[...], shape=(...))`, and `ttl.math.reduce_max` the same way: `shape` is the
  result's shape, 1 in each reduced dimension. There is no scaler argument; the guide's `reduce_sum(x, s, dims=...)`
  is the old form.
- `ttl.block.broadcast(expr, dims=[...], shape=(...))` and `ttl.block.fill(value, shape=(...))`. There is no
  `ttl.zeros`: `ttl.block.fill(0.0, shape=(...))` is a block of zeros. The guide's `ttl.math.broadcast` and
  `ttl.math.fill` moved to `ttl.block`.

Steps:
1. Read `kernel_spec.json`. It lists the inputs, their shapes, the output shape, and what the op computes.
2. Read `reference/eltwise_add.py` once, then the files listed above for your kind of op.
3. Write `kernel.py` with write_file. Define `OPERATION`: it takes the inputs in the order `kernel_spec.json` lists
   them, then the output tensor last.
4. Run `python3 kernel_check.py`. It prints `PASS`, or `FAIL: <category>: <first line>`.
5. Read that line. Change one thing. Run it again.

Advice:
- Copy the example's three thread functions into `OPERATION`: a reader with `@ttl.datamovement()`, a compute
  function with `@ttl.compute()`, and a writer with `@ttl.datamovement()`. Without them the simulator fails with
  `ModuleNotFoundError: ttl.dfb_reset`.
- Copy only the `@ttl.operation` function from an example. Write `import ttl` and `import ttnn` at the top: the
  examples' `from sim import ttl, ttnn` is refused, and so are their `import torch` and their host code with
  `ttnn.from_torch` and `ttnn.to_torch`.
- Delete the example's `main()` and its `if __name__ == "__main__":` block. Do not open a device: the check opens it,
  builds the inputs and reads the result back. The guide's advice to add a main block does not apply here.

Rules the check enforces. It reads `kernel.py` before running it and answers `FAIL: forbidden_code: ...` with every
rule broken, its line, and what to do instead:
- Imports: import only `ttl`, `ttnn` and `math`. `from __future__ import annotations` and type names such as
  `from typing import Any, Optional` are also allowed. Do not import torch. A star import (`from X import *`) is refused.
- Forbidden names, refused as a name, an attribute (`ttnn.os`) or an import (also under an alias): `open`, `exec`,
  `eval`, `compile`, `getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `sys`, `os`, `subprocess`,
  `importlib`, `builtins`, `breakpoint`, `atexit`, `references`, and calling `input()`.
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
