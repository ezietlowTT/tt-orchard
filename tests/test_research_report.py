# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The research report: ranking, the offline case, and the files it writes (orchard/research/report.py)."""
import json
import struct
from pathlib import Path

import pytest

pytest.importorskip("tokenizers", reason="the class estimate uses the delta-triage template")

from orchard import nearest, preflight as pf  # noqa: E402
from orchard.nearest import Installed  # noqa: E402
from orchard.research import lineage, report  # noqa: E402
from orchard.research.http import FetchError  # noqa: E402

NOW = "2026-10-12T00:00:00Z"
CFG = {"model_type": "gemma4", "architectures": ["Gemma4ForCausalLM"], "num_hidden_layers": 30,
       "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 262144}
T = {"model.embed_tokens.weight": ("BF16", [100, 64]), "model.layers.0.q.weight": ("BF16", [64, 64])}


def header(tensors):
    blob = json.dumps({k: {"dtype": v[0], "shape": v[1], "data_offsets": [0, 0]} for k, v in tensors.items()}).encode()
    return struct.pack("<Q", len(blob)) + blob


class World:
    """A fake hub, tt-model and GitHub. `json` and `get` answer from tables; a missing url is a 404."""

    def __init__(self, **tables):
        self.json_routes = tables.get("json", {})
        self.files = tables.get("files", {})
        self.offline = tables.get("offline", False)
        self.urls = []

    def json(self, url, **kw):
        self.urls.append(url)
        if self.offline:
            raise FetchError("offline")
        got = self.json_routes.get(url)
        if got is None:
            raise FetchError("HTTP 404")
        if isinstance(got, Exception):
            raise got
        return got

    def get(self, url, *, byte_range=None, **kw):
        self.urls.append(url)
        name = url.rsplit("/", 1)[1]
        if self.offline or name not in self.files:
            raise FetchError("HTTP 404")
        data = self.files[name]
        return data[byte_range[0]:byte_range[1] + 1] if byte_range else data

    def now(self):
        return NOW


def hub_model(id, *tags, config=CFG):
    return {"id": id, "tags": list(tags), "cardData": {}, "config": config, "siblings": []}


def cfg_url(repo):
    return f"https://huggingface.co/{repo}/resolve/main/config.json"


def manifest_url(bundle):
    return f"https://huggingface.co/{bundle}/resolve/main/tt_kernel_manifest.json"


def api(id):
    return f"https://huggingface.co/api/models/{id}"


class Hub:
    def __init__(self, table):
        self.table = table

    def __call__(self, repo):
        files = self.table.get(repo)
        if files is None:
            return None, "HTTP 404"
        return pf.HubInfo(id=repo, sha="abc", private=False, gated=False, license="mit", total_bytes=2_000_000_000,
                          files=files, code_files=[], has_license_file=False), None


def world(extra_json=None, files=None):
    routes = {api("h/new"): hub_model("h/new", "base_model:finetune:g/base"),
              api("g/base"): hub_model("g/base"),
              cfg_url("h/new"): CFG, cfg_url("g/base"): CFG,
              "https://huggingface.co/api/models?filter=base_model:finetune:g/base&sort=downloads&direction=-1&limit=20": []}
    routes.update(extra_json or {})
    f = {"model.safetensors": header(T) + b"\0" * 64}
    f.update(files or {})
    return World(json=routes, files=f)


FILES = ["config.json", "model.safetensors"]
HUBINFO = Hub({"h/new": FILES, "g/base": FILES, "g/pub": FILES, "g/cont": FILES, "g/none": FILES, "g/big": FILES})


def new_hub(*bases):
    card = {"base_model": list(bases)} if bases else {}
    return pf.parse_hub({"id": "h/new", "cardData": card, "siblings": [{"rfilename": n, "size": 1} for n in FILES]})
NO_SEARCH = lambda q: []  # noqa: E731
RUN = lambda *a, **k: (_ for _ in ()).throw(OSError("no subprocess in this test"))  # noqa: E731


def build(w, **over):
    args = dict(model_id="h/new", hub=new_hub("g/base"), base_override=None, installed=[Installed("tt/base-p150", "g/base", "6", 1)], search=NO_SEARCH,
                hub_info=HUBINFO, fetch=w, run=RUN, which=lambda n: None, no_github=True)
    args.update(over)
    return report.build(**args)


def test_the_card_base_with_an_installed_v6_bundle_is_selected_with_its_class():
    r = build(world())
    top = r["candidates"][0]
    assert top["repo"] == "g/base" and top["confidence"] == "stated" and top["group"] == 0
    assert top["class_estimate"]["class"] == "weights-only" and top["download_gb"] == 2.0
    assert r["selected"] == {"repo": "g/base", "bundle": "tt/base-p150", "confidence": "stated", "class": "weights-only"}
    assert r["bundle_choices"][0]["name"] == "tt/base-p150" and r["offline"] is False


def test_every_key_is_always_present():
    r = build(world())
    for key in ("model", "generated_at", "offline", "stated_bases", "candidates", "bundle_choices", "selected",
                "ports", "evidence", "unavailable"):
        assert key in r


def test_a_card_with_no_base_infers_one_from_the_architecture_of_an_installed_bundle():
    w = world({api("h/new"): hub_model("h/new")})
    r = build(w, hub=new_hub())
    top = r["candidates"][0]
    assert top["repo"] == "g/base" and top["confidence"] == "inferred"
    assert r["selected"]["confidence"] == "inferred"


def test_a_candidate_of_another_family_is_left_out():
    other = {**CFG, "model_type": "qwen3"}
    w = world({cfg_url("q/other"): other})
    r = build(w, installed=[Installed("tt/q", "q/other", "6", 1)], hub=new_hub())
    assert all(c["repo"] != "q/other" for c in r["candidates"])


def test_a_weak_candidate_is_listed_but_never_selected():
    w = world({api("h/new"): hub_model("h/new"), cfg_url("g/big"): {**CFG, "num_hidden_layers": 60}})
    r = build(w, hub=new_hub(), installed=[Installed("tt/big", "g/big", "6", 1)])
    big = [c for c in r["candidates"] if c["repo"] == "g/big"][0]
    assert big["confidence"] == "weak" and "num_hidden_layers" in big["differing"]
    assert r["selected"] is None
    assert r["bundle_choices"][0]["confidence"] == "weak"


def test_ranking_puts_installed_v6_before_published_v6_before_container_before_no_bundle():
    w = world({cfg_url("g/pub"): CFG, cfg_url("g/cont"): CFG, cfg_url("g/none"): CFG,
               manifest_url("o/pub"): {"weights": {"repo_id": "g/pub"}, "schema_version": "6"},
               manifest_url("o/cont"): {"weights": {"repo_id": "g/cont"}, "schema_version": "5.1"},
               api("o/pub"): {"siblings": []}, api("o/cont"): {"siblings": []}})
    r = build(w, search=lambda q: [{"name": "o/cont", "installed": False}, {"name": "o/pub", "installed": False}],
              base_override="g/none")
    assert [(c["repo"], c["group"], c["confidence"]) for c in r["candidates"]] == [
        ("g/base", 0, "stated"), ("g/pub", 1, "inferred"), ("g/cont", 2, "inferred"), ("g/none", 3, "stated")]
    assert r["selected"]["repo"] == "g/base"
    assert [c["name"] for c in r["bundle_choices"]] == ["tt/base-p150", "o/pub", "o/cont"]


def test_a_published_only_candidate_is_never_selected():
    w = world({api("h/new"): hub_model("h/new"), cfg_url("g/pub"): CFG,
               manifest_url("o/pub"): {"weights": {"repo_id": "g/pub"}, "schema_version": "6"}, api("o/pub"): {"siblings": []}})
    r = build(w, hub=new_hub(), installed=[], search=lambda q: [{"name": "o/pub", "installed": False}])
    assert r["candidates"][0]["group"] == 1 and r["selected"] is None


def test_a_stale_installed_container_bundle_is_group_2():
    r = build(world(), installed=[Installed("tt/base-container", "g/base", "5.1", 1)])
    assert r["candidates"][0]["group"] == 2 and r["selected"] is None


def test_published_bundle_manifests_are_read_and_a_missing_manifest_is_unavailable():
    w = world({api("o/gone"): {"siblings": []}})
    r = build(w, search=lambda q: [{"name": "o/gone", "installed": False}])
    assert any(u["source"] == "bundle manifest" and "o/gone" in u["reason"] for u in r["unavailable"])


def test_a_hostile_bundle_name_from_the_search_is_never_fetched():
    w = world()
    build(w, search=lambda q: [{"name": "../../x", "installed": False}, {"name": "a b/c", "installed": False}])
    assert not any("x/resolve" in u or "a b" in u for u in w.urls)


def test_headers_that_cannot_be_read_leave_no_estimate_and_say_why():
    w = world()
    w.files = {}                                  # no safetensors served at all
    r = build(w)
    assert r["candidates"][0]["class_estimate"] is None
    assert [u["reason"] for u in r["unavailable"] if u["source"] == "headers"] == [
        "h/new: model.safetensors: HTTP 404", "g/base: model.safetensors: HTTP 404"]


def test_offline_makes_no_request_and_says_so():
    w = world()
    w.offline = True
    r = report.build(model_id="h/new", hub=None, base_override=None, installed=[], search=NO_SEARCH,
                     hub_info=HUBINFO, fetch=w, run=RUN, which=lambda n: None, no_github=True)
    assert r["offline"] is True and r["candidates"] == [] and r["selected"] is None
    assert r["unavailable"] == [{"source": "research", "reason": "--offline"}]
    assert w.urls == []


def test_search_that_raises_is_unavailable_and_the_rest_of_the_report_stands():
    def boom(q):
        raise RuntimeError("tt is broken")
    r = build(world(), search=boom)
    assert r["candidates"] and any(u["source"] == "bundle search" for u in r["unavailable"])


def test_the_report_is_json_and_a_home_path_in_fetched_text_is_replaced():
    home = str(Path.home())
    w = world({cfg_url("g/base"): FetchError(f"HTTP 500 at {home}/secret/x"),
               manifest_url("o/gone"): FetchError(f"HTTP 500 at {home}/y"), api("o/gone"): {"siblings": []}})
    r = build(w, search=lambda q: [{"name": "o/gone", "installed": False}],
              hub_info=lambda repo: (None, f"failed in {home}/z"))
    text = json.dumps(r)
    assert home not in text and "[removed" in text


def test_write_puts_json_and_markdown_in_the_run_directory_only(tmp_path):
    path = report.write(tmp_path, build(world()))
    assert path == tmp_path / "research" / "base-candidates.json"
    assert json.loads(path.read_text())["model"] == "h/new"
    md = (tmp_path / "research" / "base-candidates.md").read_text()
    assert "**g/base** [stated]" in md and "expected class weights-only" in md and "2.0 GB" in md
    assert sorted(p.name for p in (tmp_path / "research").iterdir()) == ["base-candidates.json", "base-candidates.md"]


def test_the_markdown_lists_unavailable_sources_so_a_gap_is_visible():
    r = build(world(), search=lambda q: [{"name": "o/gone", "installed": False}])
    assert "- bundle manifest: o/gone: HTTP 404" in report.markdown(r)


def test_only_the_top_candidates_get_a_class_estimate():
    repos = [f"g/c{i}" for i in range(8)]
    w = world({cfg_url(r): CFG for r in repos})
    table = Hub({**{r: FILES for r in repos}, "g/base": FILES, "h/new": FILES})
    calls = []

    def counting(repo):
        calls.append(repo)
        return table(repo)
    r = build(w, installed=[Installed(f"tt/c{i}", repo, "6", 1) for i, repo in enumerate(repos)], hub_info=counting)
    assert len(calls) == report.MAX_ESTIMATES
    assert sum(1 for c in r["candidates"] if c["class_estimate"]) == report.MAX_ESTIMATES


def test_a_gated_base_is_unavailable_with_its_status_and_the_run_still_gets_an_answer():
    w = world({cfg_url("g/base"): FetchError("HTTP 403")})
    r = build(w)
    assert any(u["source"] == "config" and u["reason"] == "g/base: HTTP 403" for u in r["unavailable"])
    top = r["candidates"][0]
    assert top["repo"] == "g/base" and top["confidence"] == "stated" and top["class_estimate"] is None
    assert r["selected"]["repo"] == "g/base"      # an installed v6 bundle still serves the stated base


def test_build_is_deterministic_for_the_same_answers():  # the cache-resume case is tested with the real Fetcher in Task 7
    first, second = build(world()), build(world())
    assert first == second


def test_a_hub_listing_that_fails_is_unavailable_with_its_reason_and_leaves_no_estimate():
    r = build(world(), hub_info=lambda repo: (None, "HTTP 401"))
    assert {"source": "class estimate", "reason": "g/base: HTTP 401"} in r["unavailable"]
    assert r["candidates"][0]["class_estimate"] is None and r["candidates"][0]["download_gb"] is None
    assert r["selected"]["repo"] == "g/base"


# --- ranking keys: two candidates in one group that differ in exactly one key -----------------------------------

class PerRepo(World):
    """A World whose safetensors header depends on the repo, so two candidates can get different classes."""

    def __init__(self, headers_by_repo, **tables):
        super().__init__(**tables)
        self.by_repo = headers_by_repo

    def get(self, url, *, byte_range=None, **kw):
        repo = url.split("huggingface.co/", 1)[1].rsplit("/resolve", 1)[0] if "huggingface.co/" in url else ""
        if repo in self.by_repo:
            self.urls.append(url)
            data = self.by_repo[repo]
            return data[byte_range[0]:byte_range[1] + 1] if byte_range else data
        return super().get(url, byte_range=byte_range, **kw)


def rank(repos, *, cfgs=None, card=(), order=None, headers_by_repo=None, extra=None, no_estimates=False):
    """Build with every repo in `repos` served by an installed v6 bundle (group 0). `card` are the stated bases."""
    cfgs = cfgs or {}
    routes = {api("h/new"): hub_model("h/new"), cfg_url("h/new"): CFG,
              **{cfg_url(r): cfgs.get(r, CFG) for r in repos}, **(extra or {})}
    w = PerRepo(headers_by_repo or {}, json=routes, files={"model.safetensors": header(T) + b"\0" * 64})
    table = Hub({r: FILES for r in [*repos, "h/new"]})
    names = list(repos if order is None else order)
    return report.build(model_id="h/new", hub=new_hub(*card), base_override=None, search=NO_SEARCH, hub_info=(lambda repo: (None, "HTTP 404")) if no_estimates else table,
                        installed=[Installed(f"tt/{r.split('/')[1]}", r, "6", 1) for r in names], fetch=w, run=RUN,
                        which=lambda n: None, no_github=True)


def order_of(r):
    return [c["repo"] for c in r["candidates"]]


def test_a_stated_candidate_ranks_before_an_inferred_one_in_the_same_group():
    # g/aa sorts first by name, so only the confidence key can put g/zz ahead.
    r = rank(["g/aa", "g/zz"], card=["g/zz"])
    assert [(c["repo"], c["confidence"]) for c in r["candidates"]] == [("g/zz", "stated"), ("g/aa", "inferred")]


def test_fewer_differing_config_keys_rank_first_in_the_same_group():
    # No class estimates here, so the number of differing keys is the only key that separates the two.
    r = rank(["g/aa", "g/zz"], card=["g/aa", "g/zz"], cfgs={"g/aa": {**CFG, "num_hidden_layers": 31}}, no_estimates=True)
    by = {c["repo"]: c for c in r["candidates"]}
    assert len(by["g/aa"]["differing"]) == 1 and by["g/zz"]["differing"] == {}
    assert order_of(r) == ["g/zz", "g/aa"]


def test_a_weights_only_estimate_ranks_before_a_full_port_estimate_in_the_same_group():
    extra_tensor = {**T, "model.layers.1.q.weight": ("BF16", [64, 64])}      # a text tensor the new model lacks
    r = rank(["g/aa", "g/zz"], card=["g/aa", "g/zz"], headers_by_repo={"g/aa": header(extra_tensor) + b"\0" * 64})
    by = {c["repo"]: c["class_estimate"]["class"] for c in r["candidates"]}
    assert by == {"g/aa": "full-port", "g/zz": "weights-only"}
    assert order_of(r) == ["g/zz", "g/aa"]


def test_equal_candidates_end_in_repo_name_order_whatever_the_input_order():
    forward = rank(["g/aa", "g/mm", "g/zz"], card=["g/aa", "g/mm", "g/zz"])
    backward = rank(["g/aa", "g/mm", "g/zz"], card=["g/zz", "g/mm", "g/aa"], order=["g/zz", "g/mm", "g/aa"])
    assert order_of(forward) == order_of(backward) == ["g/aa", "g/mm", "g/zz"]


def test_a_sibling_fine_tune_is_inferred_with_the_sibling_reason():
    url = ("https://huggingface.co/api/models?filter=base_model:finetune:g/base&sort=downloads&direction=-1&limit="
           f"{lineage.SIBLING_LIMIT}")
    w = world({url: [{"id": "g/sib"}], cfg_url("g/sib"): {**CFG, "num_hidden_layers": 60}})
    r = build(w, installed=[Installed("tt/sib", "g/sib", "6", 1)])
    sib = [c for c in r["candidates"] if c["repo"] == "g/sib"][0]
    assert sib["confidence"] == "inferred" and sib["why"] == "another fine-tune of g/base"


def test_error_text_from_a_bundle_manifest_and_hub_listing_is_cleaned():
    junk = "bad\x07line\nbreak " + "y" * 600
    w = world({manifest_url("o/gone"): FetchError(junk), api("o/gone"): {"siblings": []}})
    r = build(w, search=lambda q: [{"name": "o/gone", "installed": False}], hub_info=lambda repo: (None, junk))
    rows = [u for u in r["unavailable"] if u["source"] in ("bundle manifest", "class estimate")]
    assert {u["source"] for u in rows} == {"bundle manifest", "class estimate"}
    for u in rows:
        assert "\x07" not in u["reason"] and "\n" not in u["reason"] and len(u["reason"]) <= 320


def test_a_hub_info_that_raises_becomes_an_unavailable_row():
    def boom(repo):
        raise RuntimeError("hub is down")
    r = build(world(), hub_info=boom)
    assert {"source": "class estimate", "reason": "g/base: RuntimeError"} in r["unavailable"]
    assert r["candidates"][0]["class_estimate"] is None and r["selected"]["repo"] == "g/base"
