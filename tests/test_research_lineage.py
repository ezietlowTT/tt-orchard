# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Lineage from the hub: the chain of base models and the sibling fine-tunes (orchard/research/lineage.py)."""
import pytest

from orchard.research import lineage
from orchard.research.http import FetchError


class FakeFetch:
    """Answers `.json(url)` from a dict; a missing url is a 404 FetchError."""

    def __init__(self, routes):
        self.routes, self.urls = routes, []

    def json(self, url, **kw):
        self.urls.append(url)
        if url not in self.routes:
            raise FetchError("HTTP 404")
        got = self.routes[url]
        if isinstance(got, Exception):
            raise got
        return got

    def now(self):
        return "2026-10-12T00:00:00Z"


def model(id, *tags, **card):
    return {"id": id, "tags": list(tags), "cardData": card, "config": {"model_type": "gemma4"}}


def url(id):
    return f"https://huggingface.co/api/models/{id}"


def test_the_tags_give_the_base_and_the_relation():
    p = model("a/b", "base_model:x/y", "base_model:finetune:x/y", "license:mit")
    assert lineage.card_bases(p) == [("x/y", "finetune")]


def test_a_plain_tag_alone_has_an_unknown_relation_and_the_card_field_is_a_fallback():
    assert lineage.card_bases(model("a/b", "base_model:x/y")) == [("x/y", "unknown")]
    assert lineage.card_bases(model("a/b", base_model="x/y", base_model_relation="quantized")) == [("x/y", "quantized")]
    assert lineage.card_bases(model("a/b", base_model=["x/y", "p/q"])) == [("x/y", "unknown"), ("p/q", "unknown")]
    assert lineage.card_bases(model("a/b", base_model=7)) == []


def test_the_chain_follows_bases_upward_with_one_evidence_row_per_edge():
    f = FakeFetch({url("h/model"): model("h/model", "base_model:finetune:g/it"),
                   url("g/it"): model("g/it", "base_model:finetune:g/base"),
                   url("g/base"): model("g/base")})
    r = lineage.chain(f, "h/model")
    assert r.data["ancestors"] == ["g/it", "g/base"]
    assert [(e["child"], e["parent"], e["relation"]) for e in r.data["edges"]] == [
        ("h/model", "g/it", "finetune"), ("g/it", "g/base", "finetune")]
    assert [e.confidence for e in r.evidence] == ["stated", "stated"]
    assert r.unavailable == []


def test_a_model_that_names_itself_or_a_loop_does_not_repeat():
    f = FakeFetch({url("a/b"): model("a/b", "base_model:finetune:a/b", "base_model:finetune:c/d"),
                   url("c/d"): model("c/d", "base_model:finetune:a/b")})
    r = lineage.chain(f, "a/b")
    assert r.data["ancestors"] == ["c/d"]
    assert len(f.urls) == 2


def test_the_depth_is_limited():
    routes = {url(f"o/m{i}"): model(f"o/m{i}", f"base_model:finetune:o/m{i + 1}") for i in range(20)}
    r = lineage.chain(FakeFetch(routes), "o/m0", max_depth=3)
    assert r.data["ancestors"] == ["o/m1", "o/m2", "o/m3"]


@pytest.mark.parametrize("hostile", ["x/../../y?z", "a/b c", "a/b\nc", "nogroup", "a/b/../c"])
def test_a_hostile_base_name_never_reaches_a_url(hostile):
    f = FakeFetch({url("a/b"): model("a/b", f"base_model:finetune:{hostile}")})
    r = lineage.chain(f, "a/b")
    assert r.data["ancestors"] == [] and f.urls == [url("a/b")]
    assert any("not a model id" in u.reason for u in r.unavailable)


def test_a_gated_or_missing_base_is_an_unavailable_row_and_the_rest_still_works():
    f = FakeFetch({url("a/b"): model("a/b", "base_model:finetune:g/gated"), url("g/gated"): FetchError("HTTP 401")})
    r = lineage.chain(f, "a/b")
    assert r.data["ancestors"] == ["g/gated"]
    assert [(u.source, u.reason) for u in r.unavailable] == [("lineage", "g/gated: HTTP 401")]


def test_the_new_models_hub_failure_is_unavailable_and_the_data_is_empty():
    r = lineage.chain(FakeFetch({}), "a/b")
    assert r.data["ancestors"] == [] and r.unavailable[0].reason == "a/b: HTTP 404"


def test_the_card_text_in_a_claim_is_cleaned():
    f = FakeFetch({url("a/b"): model("a/b", "base_model:finetune:g/base")})
    claim = lineage.chain(f, "a/b").evidence[0].claim
    assert claim == "a/b lists g/base as its base model (relation: finetune)"


LIST = "https://huggingface.co/api/models?filter=base_model:finetune:g/base&sort=downloads&direction=-1&limit=20"


def test_siblings_are_the_other_fine_tunes_of_the_base():
    f = FakeFetch({LIST: [{"id": "a/b"}, {"id": "s/one"}, {"id": "s/two"}, {"id": "bad id"}, {"x": 1}]})
    r = lineage.siblings(f, "g/base", exclude={"a/b"})
    assert r.data["siblings"] == ["s/one", "s/two"]
    assert all("another fine-tune of g/base" in e.claim for e in r.evidence)


def test_a_failed_sibling_query_is_unavailable():
    r = lineage.siblings(FakeFetch({LIST: FetchError("rate limited")}), "g/base")
    assert r.data["siblings"] == [] and r.unavailable[0].reason == "g/base: rate limited"


def test_a_sibling_answer_that_is_not_a_list_is_unavailable():
    r = lineage.siblings(FakeFetch({LIST: {"error": "x"}}), "g/base")
    assert r.unavailable and r.data["siblings"] == []


@pytest.mark.parametrize("card", [["a", "list"], "text", 7, None])
def test_a_card_data_that_is_not_an_object_is_treated_as_absent(card):
    payload = {"id": "a/b", "tags": ["base_model:finetune:c/d"], "cardData": card}
    assert lineage.card_bases(payload) == [("c/d", "finetune")]
    r = lineage.chain(FakeFetch({url("a/b"): payload, url("c/d"): model("c/d")}), "a/b")
    assert r.data["ancestors"] == ["c/d"]


def test_a_model_that_names_itself_has_no_self_edge():
    f = FakeFetch({url("a/b"): model("a/b", "base_model:finetune:a/b", "base_model:finetune:c/d"),
                   url("c/d"): model("c/d", "base_model:finetune:c/d")})
    r = lineage.chain(f, "a/b")
    assert all(e["child"] != e["parent"] for e in r.data["edges"])
    assert r.data["edges"] == [{"child": "a/b", "parent": "c/d", "relation": "finetune"}]
