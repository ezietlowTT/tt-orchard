# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Turn a raw op trace into an inventory: which of a model's ops ttnn already has, which it can build, which are missing.

Statuses, and what each claims:

    mapped      the mapping table names a ttnn op and the recorded dump for the installed ttnn contains that name.
                This is a name match. It does not say the op supports the shape, dtype or layout the model needs.
    composable  a decomposition exists and every leaf is `mapped`.
    unmapped    neither.
    unchecked   there is no dump for the installed ttnn version, so nothing is claimed.

The dump comes from `tt-orchard inventory refresh-ttnn` (orchard/ttnn_dump.py), which imports ttnn under a lease,
because ttnn's op names cannot be read without importing it. A dump for another ttnn version than the installed one is
not used.

Standard library only. The torch side is orchard/skills/ops-inventory-templates/ops_inventory.py.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
TEMPLATE = Path(__file__).resolve().parent / "skills" / "ops-inventory-templates" / "ops_inventory.py"
SCHEMA = 1
STATUSES = ("mapped", "composable", "unmapped", "unchecked")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
LABEL = {
    "traced": "measured: the op names, counts and one sample shape per op come from torch.export on the meta device",
    "status": "table plus dump: `mapped` is a name match against the recorded ttnn dump and does not say the op "
              "supports the model's shape, dtype or layout",
    "scaled_count": "estimate: ops inside layers are multiplied by layers in the config / layers traced",
}


def load_table(path) -> dict:
    """Read a mapping table: aten op name to {"ttnn": name}, or (for decompositions) to a list of aten op names."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    for key, entry in data.items():
        if isinstance(entry, dict):
            if not isinstance(entry.get("ttnn"), str):
                raise ValueError(f"{path}: {key!r} has no string `ttnn`")
        elif not (isinstance(entry, list) and all(isinstance(x, str) for x in entry)):
            raise ValueError(f"{path}: {key!r} must be {{\"ttnn\": name}} or a list of aten op names")
    return data


def load_dump(version, data_dir=DATA_DIR):
    """The recorded dump for exactly this ttnn version, or None. A version that is not a plain token is refused,
    so it cannot name a file outside data_dir."""
    if not isinstance(version, str) or not VERSION_RE.match(version):
        return None
    p = Path(data_dir) / f"ttnn_ops-{version}.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return d if d.get("ttnn_version") == version and isinstance(d.get("names"), list) else None


def status_of(op, mapping, decomps, names, seen=()):
    """(status, ttnn name, leaves). `names` is the set of names in the dump, or None for no usable dump."""
    entry = mapping.get(op)
    if names is None:
        return "unchecked", (entry or {}).get("ttnn"), []
    if entry and entry["ttnn"] in names:
        return "mapped", entry["ttnn"], []
    parts = decomps.get(op)
    if parts and op not in seen:
        leaves, ok = [], True
        for p in parts:
            s, _, sub = status_of(p, mapping, decomps, names, seen + (op,))
            if s == "mapped":
                leaves.append(p)
            elif s == "composable":
                leaves.extend(sub)
            else:
                ok = False
        if ok:
            return "composable", None, leaves
    return "unmapped", (entry or {}).get("ttnn"), []


def build(raw, *, mapping, decomps, dump, installed_ttnn=None, against=None) -> dict:
    """The inventory dict for one raw trace. `against` is another inventory (the base model's) to compare with."""
    inv = {"schema": SCHEMA, "model_dir": raw.get("model_dir"), "architectures": raw.get("architectures", []),
           "traced_class": raw.get("traced_class"), "torch": raw.get("torch"), "transformers": raw.get("transformers"),
           "traced": {"graph": "prefill", "seq_len": raw.get("seq_len"), "layers_in_config": raw.get("layers_in_config"),
                      "layers_traced": raw.get("layers_traced"), "note": raw.get("note")},
           "failures": raw.get("failures", []), "label": LABEL,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if "skipped" in raw:
        inv.update(skipped=raw["skipped"], ops=[], summary={}, ttnn_dump=None, ttnn_dump_note=None, scaling=None)
        return inv
    names, note = None, None
    if dump is None:
        note = "no recorded dump for the installed ttnn; nothing is claimed as mapped"
    elif installed_ttnn is None:
        note = "the installed ttnn version is unknown, so the recorded dump is not used"
    elif dump.get("ttnn_version") != installed_ttnn:
        note = f"the recorded dump is for ttnn {dump.get('ttnn_version')} and ttnn {installed_ttnn} is installed; not used"
    else:
        names = set(dump["names"])
    inv["ttnn_dump"] = {"ttnn_version": dump["ttnn_version"], "sha256": dump.get("sha256")} if names is not None else None
    inv["ttnn_dump_note"] = note
    layers, traced = raw.get("layers_in_config"), raw.get("layers_traced")
    factor = layers / traced if layers and traced else None
    inv["scaling"] = (f"in-layer ops scaled by {layers}/{traced}" if factor
                      else "the config has no recognisable layer count: counts are not scaled")
    base_ops = {o["op"] for o in against["ops"]} if against else None
    rows, distinct, scaled_total = [], {s: 0 for s in STATUSES}, {s: 0 for s in STATUSES}
    for o in raw["ops"]:
        status, ttnn_name, leaves = status_of(o["op"], mapping, decomps, names)
        inl = o.get("count_in_layers", 0)
        scaled = round(inl * factor + (o["count"] - inl)) if factor else o["count"]
        row = {"op": o["op"], "count": o["count"], "scaled_count": scaled, "status": status, "ttnn": ttnn_name,
               "leaves": leaves, "sample": o.get("sample"), "modules": o.get("modules", [])}
        if base_ops is not None:
            row["in_base"] = o["op"] in base_ops
        rows.append(row)
        distinct[status] += 1
        scaled_total[status] += scaled
    inv["ops"] = rows
    inv["summary"] = {"distinct": distinct, "scaled_total": scaled_total}
    return inv


def render_md(inv) -> str:
    lines = [f"# Op inventory: {inv.get('traced_class') or inv.get('architectures')}", ""]
    if inv.get("skipped"):
        return "\n".join(lines + [f"Skipped: {inv['skipped']}", ""])
    d = inv.get("ttnn_dump")
    lines += [f"Model directory: `{inv['model_dir']}`. transformers {inv.get('transformers')}, torch {inv.get('torch')}.",
              f"Traced: {inv['traced']['note']}. Layers in config {inv['traced']['layers_in_config']}, "
              f"traced {inv['traced']['layers_traced']}. {inv['scaling']}.", ""]
    if d:
        lines.append(f"`mapped` means the name exists in the dump of ttnn {d['ttnn_version']}. It does not say the op "
                     "supports the shape, dtype or layout this model needs.")
    else:
        lines.append(f"Nothing is claimed as mapped: {inv['ttnn_dump_note']}.")
    lines += ["", "| status | distinct ops |", "| --- | --- |"]
    lines += [f"| {s} | {n} |" for s, n in inv["summary"]["distinct"].items()]
    if inv["failures"]:
        lines += ["", "Trace failures:"] + [f"- {f['stage']}: {f['error']}" for f in inv["failures"]]
    lines += ["", "| op | count | scaled | status | ttnn |", "| --- | --- | --- | --- | --- |"]
    order = {s: i for i, s in enumerate(("unmapped", "unchecked", "composable", "mapped"))}
    for r in sorted(inv["ops"], key=lambda r: (order[r["status"]], -r["scaled_count"], r["op"])):
        lines.append(f"| `{r['op']}` | {r['count']} | {r['scaled_count']} | {r['status']} | {r['ttnn'] or ''} |")
    return "\n".join(lines) + "\n"


def write_outputs(inv, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / "ops-inventory.json", out / "ops-inventory.md"
    j.write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    m.write_text(render_md(inv), encoding="utf-8")
    return j, m


def _real_run(argv, timeout):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return done.returncode, done.stdout, done.stderr


def installed_ttnn_version(run, python):
    """The installed ttnn's version, from package metadata. ttnn itself is not imported: that would open the device."""
    if not python:
        return None
    rc, out, _ = run([str(python), "-c", "import importlib.metadata as m; print(m.version('ttnn'))"], 60)
    return out.strip() or None if rc == 0 else None


def run_inventory(model_dir, *, python, ttnn_python, out_dir, layers=2, seq=8, cls=None, against=None,
                  run=_real_run, data_dir=DATA_DIR, say=print) -> int:
    """Trace, map, write. 0 on success (a skipped model is a success), 2 when the template could not run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw-ops.json"
    argv = [str(python), str(TEMPLATE), "--model-dir", str(model_dir), "--out", str(raw_path),
            "--layers", str(layers), "--seq", str(seq)] + (["--class", cls] if cls else [])
    rc, stdout, stderr = run(argv, 1800)
    if rc not in (0, 3) or not raw_path.is_file():
        say(f"the trace did not run (exit {rc}): {(stderr or stdout).strip()[-300:]}")
        return 2
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    version = installed_ttnn_version(run, ttnn_python)
    base = json.loads(Path(against).read_text(encoding="utf-8")) if against else None
    inv = build(raw, mapping=load_table(Path(data_dir) / "aten_to_ttnn.json"),
                decomps=load_table(Path(data_dir) / "decompositions.json"), dump=load_dump(version, data_dir),
                installed_ttnn=version, against=base)
    j, m = write_outputs(inv, out)
    say(f"wrote {j} and {m}")
    if inv.get("skipped"):
        say(f"skipped: {inv['skipped']}")
    else:
        say("distinct ops by status: " + ", ".join(f"{s} {n}" for s, n in inv["summary"]["distinct"].items()))
    return 0
