# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Existing ports: the catalog, tt-metal listings, GitHub issues and code, and code shipped in bundles."""
import json
import subprocess
from pathlib import Path

import pytest

from orchard.research import ports
from orchard.research.http import FetchError

FIX = Path(__file__).parent / "fixtures" / "research"
NOW = "2026-10-12T00:00:00Z"
CFG = {"model_type": "gemma4_unified", "architectures": ["Gemma4UnifiedForConditionalGeneration"]}


def test_terms_come_from_the_model_type_and_architecture():
    t = ports.terms(CFG)
    assert t["specific"] == ["gemma4"] and t["family"] == ["gemma"]
    assert t["classes"] == ["Gemma4UnifiedForConditionalGeneration"]


@pytest.mark.parametrize("cfg", [{}, {"model_type": None}, {"model_type": 5}, {"architectures": "x"},
                                 {"architectures": ["bad class name", "x" * 200, "a/b", "repo:o/p"]}])
def test_odd_configs_give_no_terms_not_a_crash_and_no_unsafe_class_name(cfg):
    t = ports.terms(cfg)
    assert t["classes"] == [] and all(ports.SAFE_TERM.fullmatch(x) for x in t["specific"] + t["family"])


def test_a_model_type_that_could_change_a_query_gives_no_specific_term():
    # `terms` is the one place a model's own text becomes a search term: the model_type is checked too.
    t = ports.terms({"model_type": "gemma4 repo:other/secret"})
    assert t["specific"] == [] and t["family"] == ["gemma"]
    short = ports.terms({"model_type": "a1"})          # shorter than SAFE_TERM allows
    assert short["specific"] == [] and short["family"] == []


def test_at_most_two_class_names_become_terms():
    t = ports.terms({"architectures": ["AaaForCausalLM", "BbbForCausalLM", "CccForCausalLM"]})
    assert t["classes"] == ["AaaForCausalLM", "BbbForCausalLM"]


# ---- name matching ---------------------------------------------------------------------------------

def test_a_specific_match_is_inferred_and_a_family_only_match_is_weak():
    hits = ports.match_names(["gemma3", "gemma4_it", "llama3", "gemma"], ports.terms(CFG))
    assert hits == [("gemma4_it", "inferred"), ("gemma3", "weak"), ("gemma", "weak")]


# ---- tt-metal ---------------------------------------------------------------------------------------

class FakeFetch:
    def __init__(self, routes):
        self.routes, self.urls = routes, []

    def json(self, url, **kw):
        self.urls.append(url)
        got = self.routes.get(url, FetchError("HTTP 404"))
        if isinstance(got, Exception):
            raise got
        return got

    def get(self, url, **kw):
        self.urls.append(url)
        got = self.routes.get(url, FetchError("HTTP 404"))
        if isinstance(got, Exception):
            raise got
        return got if isinstance(got, bytes) else json.dumps(got).encode()

    def now(self):
        return NOW


DEMOS = "https://api.github.com/repos/tenstorrent/tt-metal/contents/models/demos"
EXPER = "https://api.github.com/repos/tenstorrent/tt-metal/contents/models/experimental"
README = "https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/README.md"


def test_metal_lists_matching_demo_directories_and_readme_rows():
    f = FakeFetch({DEMOS: [{"name": "gemma4", "type": "dir", "html_url": "https://github.com/tenstorrent/tt-metal/tree/main/models/demos/gemma4"},
                           {"name": "llama3", "type": "dir", "html_url": "https://github.com/x/y"}],
                   EXPER: [],
                   README: b"| [Gemma 4 12B (TP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/gemma4) | 32 |\n"
                           b"| [Llama 3.1 (TP=4)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers) | 32 |\n"})
    r = ports.metal(f, ports.terms(CFG))
    claims = [e.claim for e in r.evidence]
    assert any("models/demos/gemma4" in c for c in claims) and any("Gemma 4 12B" in c for c in claims)
    assert not any("llama" in c.lower() for c in claims)
    assert {e.confidence for e in r.evidence} == {"inferred"}


def test_a_demo_listing_link_outside_github_is_replaced_by_the_repo_link():
    f = FakeFetch({DEMOS: [{"name": "gemma4", "type": "dir", "html_url": "https://evil.example/x"}], EXPER: [], README: b""})
    r = ports.metal(f, ports.terms(CFG))
    assert [e.url for e in r.evidence] == ["https://github.com/tenstorrent/tt-metal"]


def test_the_recorded_tt_metal_readme_still_parses_into_rows():
    rows = ports.readme_rows((FIX / "metal-readme-head.md").read_text())
    assert len(rows) == 22
    assert rows[0] == ("Qwen 3 32B", "https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers")


def test_the_recorded_tt_metal_demos_listing_finds_the_gemma4_ports():
    demos = json.loads((FIX / "metal-demos.json").read_text())
    f = FakeFetch({DEMOS: demos, EXPER: [], README: (FIX / "metal-readme-head.md").read_bytes()})
    r = ports.metal(f, ports.terms(CFG))
    assert [(e.claim, e.confidence) for e in r.evidence] == [
        ("tt-metal has models/demos/gemma4", "inferred"), ("tt-metal has models/demos/gemma4_31b_qb2", "inferred"),
        ("tt-metal has models/demos/gemma4_d_p", "inferred")]
    assert r.unavailable == []


def test_a_tt_metal_listing_that_fails_is_unavailable_per_listing():
    r = ports.metal(FakeFetch({DEMOS: FetchError("rate limited")}), ports.terms(CFG))
    assert any(u.source == "tt-metal" and "rate limited" in u.reason for u in r.unavailable)


# ---- GitHub issues -----------------------------------------------------------------------------------

def issues_url(term):
    # The query as GitHub accepts it (checked 2026-10-12: gemma4 gave 444 results).
    return f"https://api.github.com/search/issues?q={term}%20repo:tenstorrent%2Ftt-metal&per_page=5"


def test_issues_become_weak_evidence_with_cleaned_titles_and_checked_links():
    f = FakeFetch({issues_url("gemma4"): {"items": [
        {"number": 12, "title": "Gemma 4 bring-up", "state": "open", "html_url": "https://github.com/tenstorrent/tt-metal/pull/12", "pull_request": {}},
        {"number": 13, "title": "see /home/someone/x", "state": "closed", "html_url": "https://github.com/tenstorrent/tt-metal/issues/13"},
        {"number": 14, "title": "evil", "state": "open", "html_url": "https://evil.example/x"}]}})
    r = ports.issues(f, ports.terms(CFG))
    assert [e.confidence for e in r.evidence] == ["weak", "weak"]
    assert r.evidence[0].claim == "PR #12 (open): Gemma 4 bring-up"
    assert r.evidence[1].claim == "[removed: an absolute home path]"


def test_a_rate_limited_issue_search_is_unavailable():
    r = ports.issues(FakeFetch({issues_url("gemma4"): FetchError("rate limited")}), ports.terms(CFG))
    assert r.evidence == [] and r.unavailable[0].source == "github issues"


# ---- gh code search ------------------------------------------------------------------------------------

class Run:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.out = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self.argv = []

    def __call__(self, argv, **kw):
        self.argv.append(argv)
        return self.out


def which_gh(name):
    return "/usr/bin/gh" if name == "gh" else None


def test_code_search_runs_gh_with_a_checked_term_as_one_argument():
    run = Run(json.dumps({"items": [{"path": "models/demos/gemma4/tt/model.py",
                                     "html_url": "https://github.com/tenstorrent/tt-metal/blob/main/models/demos/gemma4/tt/model.py"}]}))
    r = ports.github_code(ports.terms(CFG), run=run, which=which_gh, now=NOW)
    assert r.evidence[0].claim == "tt-metal code mentions Gemma4UnifiedForConditionalGeneration in models/demos/gemma4/tt/model.py"
    assert run.argv[0][:4] == ["/usr/bin/gh", "api", "-X", "GET"]
    assert "q=Gemma4UnifiedForConditionalGeneration repo:tenstorrent/tt-metal" in run.argv[0]


def test_a_code_hit_with_a_link_outside_github_is_dropped():
    run = Run(json.dumps({"items": [{"path": "a.py", "html_url": "https://evil.example/a"}]}))
    assert ports.github_code(ports.terms(CFG), run=run, which=which_gh, now=NOW).evidence == []


def test_code_search_is_skipped_without_gh_or_when_gh_fails():
    assert "not installed" in ports.github_code(ports.terms(CFG), run=Run(), which=lambda n: None, now=NOW).unavailable[0].reason
    r = ports.github_code(ports.terms(CFG), run=Run(returncode=1, stderr="To get started with GitHub CLI, run: gh auth login"),
                          which=which_gh, now=NOW)
    assert r.evidence == [] and "gh failed" in r.unavailable[0].reason


def test_a_term_that_could_change_the_query_is_never_passed_to_gh():
    run = Run("{}")
    t = ports.terms({"architectures": ["x repo:other/secret"], "model_type": "gemma4"})
    ports.github_code(t, run=run, which=which_gh, now=NOW)
    assert all("other/secret" not in " ".join(a) for a in run.argv)


# ---- the catalog ----------------------------------------------------------------------------------------

def test_the_catalog_search_uses_tt_and_the_catalog_flag():
    run = Run((FIX / "catalog-gemma.json").read_text())
    r = ports.catalog("gemma", run=run, which=lambda n: "/usr/bin/tt" if n == "tt" else None, now=NOW)
    assert run.argv[0] == ["/usr/bin/tt", "model", "search", "gemma", "--catalog", "--json"]
    assert [b["name"] for b in r.data["bundles"]] == ["stisiTT/gemma-4-12b-it-p150"]
    assert r.evidence[0].claim == "the community catalog lists stisiTT/gemma-4-12b-it-p150"


def test_a_missing_tt_or_a_failed_search_is_unavailable():
    assert ports.catalog("gemma", run=Run(), which=lambda n: None, now=NOW).unavailable
    assert ports.catalog("gemma", run=Run(returncode=2), which=lambda n: "/x/tt", now=NOW).unavailable
    assert ports.catalog("gemma", run=Run("not json"), which=lambda n: "/x/tt", now=NOW).unavailable


# ---- code shipped in a bundle -----------------------------------------------------------------------------

def test_a_bundle_that_ships_autoport_code_is_evidence_of_a_port():
    sib = [{"rfilename": "code/models/autoports/google_gemma_4_26b_a4b_it/tt/model.py"},
           {"rfilename": "code/models/common/README.md"},
           {"rfilename": "code/models/demos/gemma4/x.py"},
           {"rfilename": "README.md"}, {"x": 1}]
    ev = ports.bundle_code(sib, "tt-hous/gemma-4-26B-A4B-it_p150", NOW)
    assert [e.claim for e in ev] == [
        "bundle tt-hous/gemma-4-26B-A4B-it_p150 ships code for google_gemma_4_26b_a4b_it",
        "bundle tt-hous/gemma-4-26B-A4B-it_p150 ships code for gemma4"]
    assert {e.confidence for e in ev} == {"inferred"}


def test_the_recorded_gemma_bundle_listing_yields_one_port_row_and_ignores_shared_code():
    payload = json.loads((FIX / "bundle-gemma-4-12b-p150.json").read_text())
    ev = ports.bundle_code(payload["siblings"], "stisiTT/gemma-4-12b-it-p150", NOW)
    assert [e.claim for e in ev] == ["bundle stisiTT/gemma-4-12b-it-p150 ships code for gemma4"]


# ---- collect ------------------------------------------------------------------------------------------------

def test_collect_skips_github_when_asked_and_runs_the_rest():
    f = FakeFetch({DEMOS: [], EXPER: [], README: b""})
    run = Run(json.dumps({"bundles": []}))
    r = ports.collect(f, CFG, family="gemma", run=run, which=which_gh, no_github=True)
    assert not any("search/issues" in u for u in f.urls)
    assert run.argv == []                                     # no gh, and `tt` is not found by which_gh
    assert [u.source for u in r.unavailable] == ["catalog"]   # `tt` is not installed in this test


def test_collect_runs_the_github_searches_by_default():
    f = FakeFetch({DEMOS: [], EXPER: [], README: b""})
    run = Run(json.dumps({"items": []}))
    ports.collect(f, CFG, family="gemma", run=run, which=which_gh)
    assert any("search/issues" in u for u in f.urls)
    assert any("search/code" in argv for argv in run.argv)
