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
import subprocess
from pathlib import Path

from orchard import opinventory

MARK = "ORCHARD_TTNN_DUMP "
# Public callables at the top of ttnn and one level into the two namespaces that hold model ops.
SCRIPT = r'''
import importlib.metadata as m, json, types
import ttnn
names = []
for n in sorted(dir(ttnn)):
    if n.startswith("_"):
        continue
    obj = getattr(ttnn, n, None)
    if callable(obj):
        names.append(n)
    elif isinstance(obj, types.ModuleType) and n in ("experimental", "transformer"):
        names += [f"{n}.{k}" for k in sorted(dir(obj)) if not k.startswith("_") and callable(getattr(obj, k, None))]
print("ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": m.version("ttnn"), "names": names}))
'''


def dump_command(python, who: str = "claude:orchard-inventory") -> list[str]:
    """The full command line. It always goes through `gozer run`, so the import holds a one-chip lease."""
    return ["gozer", "run", "--chips", "1", "--exact", "1", "--who", who,
            "--reason", "dump ttnn op names for tt-orchard inventory", "--", str(python), "-c", SCRIPT]


def parse(stdout: str) -> dict:
    """The dump the script printed on its marked line. Other output (logs, warnings) is ignored."""
    for line in stdout.splitlines():
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    raise ValueError("the dump printed no ORCHARD_TTNN_DUMP line")


def write_dump(dump: dict, data_dir) -> Path:
    """Write ttnn_ops-<version>.json with sorted, de-duplicated names and a sha256 of that list."""
    version = dump["ttnn_version"]
    if not isinstance(version, str) or not opinventory.VERSION_RE.match(version):
        raise ValueError(f"ttnn version {version!r} is not a plain version token")
    names = sorted(set(dump["names"]))
    out = Path(data_dir) / f"ttnn_ops-{version}.json"
    out.write_text(json.dumps({"ttnn_version": version, "names": names,
                               "sha256": hashlib.sha256(json.dumps(names).encode()).hexdigest()}, indent=2) + "\n",
                   encoding="utf-8")
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
    say("running (it takes a one-chip lease): " + " ".join(argv[:11]) + " -c <script>")
    rc, out, err = run(argv, 1800)
    if rc != 0:
        say(f"the dump failed (exit {rc}): {(err or out).strip()[-300:]}")
        return 2
    try:
        path = write_dump(parse(out), data_dir)
    except (ValueError, KeyError) as exc:
        say(f"the dump could not be read: {exc}")
        return 2
    say(f"wrote {path}")
    return 0
