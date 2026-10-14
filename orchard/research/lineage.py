# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where a model came from, from the Hugging Face hub (no token).

A model card names its base in `cardData.base_model` and in tags: `base_model:<id>` and
`base_model:<relation>:<id>` (relation: finetune, quantized, adapter or merge). `chain` follows those upward.
`siblings` lists the other fine-tunes of a base, which is how a bundle built for a sibling is found."""
from __future__ import annotations

from orchard.research.http import FetchError
from orchard.research.types import STATED, Evidence, Result, Unavailable, clean, valid_id

HUB_MODEL = "https://huggingface.co/api/models/{id}"
HUB_LIST = ("https://huggingface.co/api/models?filter=base_model:finetune:{id}"
            "&sort=downloads&direction=-1&limit={limit}")
RELATIONS = ("finetune", "quantized", "adapter", "merge")
MAX_DEPTH = 5
SIBLING_LIMIT = 20


def card_bases(payload: dict) -> list[tuple[str, str]]:
    """[(base id, relation)] from a hub model payload. Tags first (they carry the relation), then the card."""
    found: dict[str, str] = {}
    for tag in payload.get("tags") or []:
        if not isinstance(tag, str) or not tag.startswith("base_model:"):
            continue
        rest = tag[len("base_model:"):]
        relation, _, ident = rest.partition(":")
        if relation in RELATIONS and ident:
            found[ident] = relation
        else:
            found.setdefault(rest, "unknown")
    card = payload.get("cardData")
    card = card if isinstance(card, dict) else {}
    raw = card.get("base_model")
    names = [raw] if isinstance(raw, str) else [b for b in raw if isinstance(b, str)] if isinstance(raw, list) else []
    relation = card.get("base_model_relation")
    for name in names:
        found.setdefault(name, relation if relation in RELATIONS else "unknown")
    return list(found.items())


def chain(fetch, model_id: str, *, max_depth: int = MAX_DEPTH) -> Result:
    """The bases of `model_id`, nearest first. Every base id is checked before it is put in a URL."""
    res = Result(data={"ancestors": [], "edges": []})
    seen = {model_id}
    frontier = [model_id]
    for _ in range(max_depth):
        following = []
        for child in frontier:
            try:
                payload = fetch.json(HUB_MODEL.format(id=child))
            except FetchError as exc:
                res.unavailable.append(Unavailable("lineage", f"{child}: {exc.reason}"))
                continue
            if not isinstance(payload, dict):
                res.unavailable.append(Unavailable("lineage", f"{child}: the answer is not a model"))
                continue
            for parent, relation in card_bases(payload):
                if not valid_id(parent):
                    res.unavailable.append(Unavailable("lineage", f"{child}: {clean(parent, 60)!r} is not a model id"))
                    continue
                if parent == child:
                    continue
                res.evidence.append(Evidence(
                    "hub", clean(f"{child} lists {parent} as its base model (relation: {relation})"),
                    f"https://huggingface.co/{child}", STATED, fetch.now()))
                res.data["edges"].append({"child": child, "parent": parent, "relation": relation})
                if parent not in seen:
                    seen.add(parent)
                    res.data["ancestors"].append(parent)
                    following.append(parent)
        frontier = following
        if not frontier:
            break
    return res


def siblings(fetch, base_id: str, *, exclude=(), limit: int = SIBLING_LIMIT) -> Result:
    """Other fine-tunes of `base_id`, most downloaded first."""
    res = Result(data={"siblings": []})
    try:
        items = fetch.json(HUB_LIST.format(id=base_id, limit=limit))
    except FetchError as exc:
        res.unavailable.append(Unavailable("siblings", f"{base_id}: {exc.reason}"))
        return res
    if not isinstance(items, list):
        res.unavailable.append(Unavailable("siblings", f"{base_id}: the answer is not a list"))
        return res
    for item in items:
        ident = item.get("id") if isinstance(item, dict) else None
        if not valid_id(ident) or ident in exclude:
            continue
        res.data["siblings"].append(ident)
        res.evidence.append(Evidence("hub", clean(f"{ident} is another fine-tune of {base_id}"),
                                     f"https://huggingface.co/{ident}", STATED, fetch.now()))
    return res
