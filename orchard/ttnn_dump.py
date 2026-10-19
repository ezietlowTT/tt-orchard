# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard inventory refresh-ttnn`: record the names of the ops the installed ttnn offers.

ttnn's ops are bound from compiled code, so their names are in no Python file, and importing ttnn opens the device.
The import therefore runs under a one-chip gozer lease, once per ttnn version, and the names are written to
orchard/data/ttnn_ops-<version>.json. The operator runs this command. The supervisor never does.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from orchard import opinventory

MARK = "ORCHARD_TTNN_DUMP "
# Public callables at the top of ttnn and one level into the two namespaces that hold model ops.
SCRIPT = r'''
import importlib.metadata as m, json, types
import ttnn

_MISSING = object()
skipped = []


def safe_get(obj, name, label):
    """getattr that survives a lazy or deprecated attribute raising anything. The name is recorded in `skipped`."""
    try:
        return getattr(obj, name)
    except Exception:
        skipped.append(label)
        return _MISSING


def is_op(obj):
    # Classes (Tensor, MemoryConfig, ...) are types and are left out. Functions, builtins and callable objects are ops.
    return obj is not _MISSING and callable(obj) and not isinstance(obj, type)


names = []
for n in sorted(dir(ttnn)):
    if n.startswith("_"):
        continue
    obj = safe_get(ttnn, n, n)
    if is_op(obj):
        names.append(n)
    elif isinstance(obj, types.ModuleType) and n in ("experimental", "transformer"):
        for k in sorted(dir(obj)):
            if not k.startswith("_") and is_op(safe_get(obj, k, f"{n}.{k}")):
                names.append(f"{n}.{k}")
# The leading newline keeps the marker at the start of its line even if ttnn printed text without one.
print("\n" + "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": m.version("ttnn"), "names": names, "skipped": skipped}))
'''


def dump_command(python, who: str = "claude:orchard-inventory") -> list[str]:
    """The full command line. It always goes through `gozer run`, so the import holds a one-chip lease."""
    return ["gozer", "run", "--chips", "1", "--exact", "1", "--who", who,
            "--reason", "dump ttnn op names for tt-orchard inventory", "--", str(python), "-c", SCRIPT]


def parse(stdout: str) -> dict:
    """The dump the script printed on its marked line. Other output (logs, warnings) is ignored. The marker may follow
    banner text on its line, and when several lines carry it the last one wins."""
    found = None
    for line in stdout.splitlines():
        i = line.find(MARK)
        if i >= 0:
            found = line[i + len(MARK):]
    if found is None:
        raise ValueError("the dump printed no ORCHARD_TTNN_DUMP line")
    return json.loads(found)


def write_dump(dump: dict, data_dir) -> Path:
    """Write ttnn_ops-<version>.json with sorted, de-duplicated names and a sha256 of that list."""
    version = dump["ttnn_version"]
    if not isinstance(version, str) or not opinventory.VERSION_RE.match(version):
        raise ValueError(f"ttnn version {version!r} is not a plain version token")
    names = sorted(set(dump["names"]))
    out = Path(data_dir) / f"ttnn_ops-{version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # sha256 = sha256 of json.dumps(names) with default separators, over the sorted, de-duplicated names.
    body = json.dumps({"ttnn_version": version, "names": names, "skipped": sorted(set(dump.get("skipped", []))),
                       "sha256": hashlib.sha256(json.dumps(names).encode()).hexdigest()}, indent=2) + "\n"
    # Write beside the target and rename, so a crash never leaves a half-written file under the final name.
    tmp = out.with_name(out.name + ".tmp")
    try:
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def _real_run(argv, timeout):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return done.returncode, done.stdout, done.stderr


def main(python, *, run=_real_run, data_dir=opinventory.DATA_DIR, say=print) -> int:
    """Show the command, run it, write the file. 0 on success, 2 when the dump failed or could not be read."""
    argv = dump_command(python)
    say("running (it takes a one-chip lease): " + " ".join(argv[:12]) + " -c <script>")
    rc, out, err = run(argv, 1800)
    if rc == 10:
        say("gozer queued the lease (exit 10): run `gozer wait <ticket>` or retry")
        return 2
    if rc != 0:
        say(f"the dump failed (exit {rc}): {(err or out).strip()[-300:]}")
        return 2
    try:
        path = write_dump(parse(out), data_dir)
    except (ValueError, KeyError, OSError) as exc:
        say(f"the dump could not be read: {exc}")
        return 2
    say(f"wrote {path}")
    return 0
