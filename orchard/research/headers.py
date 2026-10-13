# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Safetensors headers by HTTP range request, and a class estimate from them.

A safetensors file starts with an 8-byte little-endian length N, then N bytes of JSON that name every
tensor with its dtype and shape. Two range requests read that without fetching any weights. The estimate
applies the stage 0 template's rules (triage_rules) to those headers, so a model can be judged weights-only
or full-port before the download. A sidecar's tensors would need its own download, so the estimate only
flags that one may exist."""
from __future__ import annotations

import json
import re
import struct

from orchard.research import triage_rules
from orchard.research.http import FetchError
from orchard.research.types import Result, Unavailable, clean, valid_id

RAW_URL = "https://huggingface.co/{repo}/resolve/main/{name}"
INDEX_NAME = "model.safetensors.index.json"
SAFE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}")
MAX_SHARDS = 64
MAX_REASONS = 5


def shard_header(fetch, repo: str, name: str) -> dict:
    """{tensor: {"dtype", "shape"}} for one file, from two range requests. Raises FetchError or ValueError."""
    url = RAW_URL.format(repo=repo, name=name)
    first = fetch.get(url, byte_range=(0, 7), max_bytes=8)
    if len(first) != 8:
        raise ValueError("shorter than 8 bytes")
    (n,) = struct.unpack("<Q", first)
    if n <= 0 or n > triage_rules.load().MAX_HEADER:
        raise ValueError(f"header length {n} is not plausible")
    blob = fetch.get(url, byte_range=(8, 8 + n - 1), max_bytes=n)
    if len(blob) != n:
        raise ValueError(f"header is cut short ({len(blob)} of {n} bytes)")
    header = json.loads(blob.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")
    return {k: {"dtype": v.get("dtype"), "shape": v.get("shape")}
            for k, v in header.items() if k != "__metadata__" and isinstance(v, dict)}


def read_model(fetch, repo: str, files: list[str]) -> Result:
    """The backbone's tensors from the headers of its shards. `files` is the repo's file list from the hub.
    The backbone is the shards the index lists, or `model*.safetensors` when there is no index."""
    res = Result(data={"tensors": {}, "shards": [], "extra_safetensors": [], "errors": []})
    if not valid_id(repo):
        res.unavailable.append(Unavailable("headers", f"{clean(repo, 60)!r} is not a model id"))
        return res
    names = sorted(f for f in files if isinstance(f, str) and f.endswith(".safetensors"))
    safe = [n for n in names if SAFE_FILE.fullmatch(n)]
    if not safe:
        res.unavailable.append(Unavailable("headers", f"{repo}: no safetensors files"))
        return res
    shards = None
    if INDEX_NAME in files:
        try:
            index = fetch.json(RAW_URL.format(repo=repo, name=INDEX_NAME))
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if isinstance(weight_map, dict):
                listed = {v for v in weight_map.values() if isinstance(v, str)}
                shards = [n for n in safe if n in listed] or None
            else:
                res.unavailable.append(Unavailable("headers", f"{repo}: the index has no weight_map object"))
        except FetchError:
            shards = None
    if shards is None:
        pattern = triage_rules.load().BACKBONE_SHARD
        shards = [n for n in safe if pattern.fullmatch(n)] or safe
    if len(shards) > MAX_SHARDS:
        res.unavailable.append(Unavailable("headers", f"{repo}: more than {MAX_SHARDS} shards"))
        return res
    res.data["shards"] = shards
    res.data["extra_safetensors"] = [n for n in safe if n not in shards]
    for shard in shards:
        try:
            for name, info in shard_header(fetch, repo, shard).items():
                res.data["tensors"][name] = {**info, "shard": shard}
        except (FetchError, ValueError, UnicodeDecodeError) as exc:
            res.data["errors"].append(f"{shard}: {getattr(exc, 'reason', exc)}")
    return res


def estimate_class(new_cfg: dict, new: dict, cand_cfg: dict, cand: dict) -> dict:
    """weights-only, full-port or unknown, by the stage 0 rules, from configs and headers alone."""
    rules = triage_rules.load()
    c = rules.compare_config_dicts(new_cfg, cand_cfg, [], "new/config.json", "candidate/config.json")
    t = rules.compare_tensor_maps(new["tensors"], cand["tensors"], mshards=new["shards"], bshards=cand["shards"],
                                  merr=new["errors"], berr=cand["errors"], model_path="new", base_path="candidate")
    reasons = rules.decide_path(c, t)
    cls = rules.decide_class(c, t, reasons, {"problems": [], "sidecars": []})
    return {"class": cls, "reasons": [clean(r) for r in reasons[:MAX_REASONS]],
            "sidecar_possible": bool(new["extra_safetensors"]),
            "shared_tensors": t["normalized_name_sets"]["common"]}
