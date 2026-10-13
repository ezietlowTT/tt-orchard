# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The base-model research in the browser UI (orchard/webui.py).

The preflight panel shows the ranked bundles, the existing work and what could not be read. A run's page
shows the report the run wrote, and what stage 0 measured against the estimate. The report holds text from
the internet, so everything the page receives is bounded and redacted, and the report files are served only
from their two allow-listed names.
"""
import json

import pytest

from orchard import webui
from orchard.preflight import BLOCK, OK, WARN, Check
from test_webui import HOME, make_app, root, write_ledger  # noqa: F401  (root is a fixture)

REPORT = {
    "model": "org/new", "generated_at": "2026-10-13T10:00:00Z", "offline": False,
    "stated_bases": ["google/g"],
    "candidates": [{"repo": "google/g-it", "confidence": "inferred", "why": "another fine-tune of google/g",
                    "group": 1, "download_gb": 24.1, "class_estimate": {"class": "weights-only"},
                    "bundles": [{"name": "stisi/g-it-p150", "installed": False, "schema": "6"}], "differing": {}}],
    "bundle_choices": [{"name": "stisi/g-it-p150", "installed": False, "repo": "google/g-it", "schema": "6",
                        "class": "weights-only", "confidence": "inferred", "group": 1}],
    "selected": None,
    "ports": [{"claim": "tt-metal has models/demos/gemma4", "url": "https://github.com/tenstorrent/tt-metal"}],
    "evidence": [], "unavailable": [{"source": "github issues", "reason": "HTTP 403"}],
}


def base_check(report, status=BLOCK):
    return Check("base", status, "no bundle", "nearest-model-missing",
                 {"base": "google/g", "candidates": [], "report": report})


def test_the_preflight_carries_a_bounded_research_view(root):
    app = make_app(root, preflight=lambda m, b: [Check("hub", OK, "found"), base_check(REPORT)])
    out = app.preflight("org/new", None)
    view = out["research"]
    assert view["choices"] == [{"name": "stisi/g-it-p150", "installed": False, "class": "weights-only",
                                "confidence": "inferred", "schema": "6"}]
    assert view["candidates"][0]["repo"] == "google/g-it" and view["candidates"][0]["class"] == "weights-only"
    assert view["ports"][0]["url"].startswith("https://github.com/")
    assert view["unavailable"] == [{"source": "github issues", "reason": "HTTP 403"}]
    assert "report" not in json.dumps(out["checks"])             # the raw report never rides on a check row


def test_a_preflight_without_a_report_has_no_research(root):
    app = make_app(root, preflight=lambda m, b: [Check("hub", OK, "found")])
    assert app.preflight("org/new", None)["research"] is None


def test_the_view_is_redacted_and_bounded(root):
    big = dict(REPORT)
    big["bundle_choices"] = [dict(REPORT["bundle_choices"][0], name=f"o/b{i}") for i in range(40)]
    big["candidates"] = [dict(REPORT["candidates"][0], repo=f"o/c{i}") for i in range(40)]
    big["ports"] = [{"claim": f"{HOME}/secret hf_{'a' * 34}", "url": "https://github.com/x"}] * 40
    big["unavailable"] = [{"source": "s", "reason": "r"}] * 40
    view = make_app(root).research_view(big)
    assert len(view["choices"]) <= 8 and len(view["candidates"]) <= 8
    assert len(view["ports"]) <= 5 and len(view["unavailable"]) <= 10
    text = json.dumps(view)
    assert HOME not in text and "hf_" not in text and "<token>" in text


@pytest.mark.parametrize("junk", [None, [], "x", {"candidates": "x", "bundle_choices": 3, "ports": {"a": 1},
                                                  "unavailable": [1, None], "selected": []}])
def test_odd_report_shapes_do_not_raise(root, junk):
    view = make_app(root).research_view(junk)
    assert view is None or isinstance(view, dict)


def test_a_url_that_is_not_https_is_not_sent_as_a_link(root):
    bad = dict(REPORT, ports=[{"claim": "x", "url": "javascript:alert(1)"}])
    assert make_app(root).research_view(bad)["ports"][0]["url"] is None


def test_a_research_choice_is_validated_and_reaches_the_preflight(root):
    seen = []

    def preflight_for(cfg):
        return lambda m, b, **kw: seen.append(kw) or [Check("hub", OK, "found")]

    app = make_app(root)
    app._preflight_for = preflight_for
    app.preflight("org/new", None, None, "no-github")
    app.preflight("org/new", None, None, "full")
    assert seen == [{"research": "no-github"}, {}]                # the default is asked for as before, with no keyword
    with pytest.raises(webui.BadRequest):
        app.preflight("org/new", None, None, "everything")
    with pytest.raises(webui.BadRequest):
        app.bringup("org/new", None, None, "--evil")


@pytest.mark.parametrize("choice, flags", [("full", []), (None, []), ("no-github", ["--no-github"]),
                                           ("offline", ["--offline"])])
def test_a_bringup_forwards_the_research_choice_as_flags(root, choice, flags):
    app = make_app(root, preflight=lambda m, b: [Check("hub", OK, "found")])
    app._preflight_for = lambda cfg: (lambda m, b, research=None: [Check("hub", OK, "found")])
    app.bringup("org/new", None, None, choice)
    argv = app.spawn.calls[0]["argv"]
    assert [a for a in argv if a in ("--no-github", "--offline")] == flags


def test_the_report_files_are_allowed_listed_and_redacted(root):
    run = write_ledger(root / "org--a")
    d = run / "research"
    d.mkdir()
    (d / "base-candidates.json").write_text(json.dumps(REPORT))
    (d / "base-candidates.md").write_text(f"# Report\n{HOME}/x\n")
    (d / "base-candidates.stage0.json").write_text("{}")
    (d / "cache").mkdir()
    (d / "cache" / "a.json").write_text("{}")
    app = make_app(root)
    listed = app.list_files(run)
    assert "research/base-candidates.json" in listed and "research/base-candidates.md" in listed
    assert not any("cache" in f or "stage0" in f for f in listed)
    assert HOME not in app.read_file("org--a", "research/base-candidates.md")
    for rel in ("research/cache/a.json", "research/base-candidates.stage0.json", "research/../ledger.jsonl"):
        with pytest.raises(webui.Forbidden):
            app.read_file("org--a", rel)


def test_a_run_shows_its_report_and_stage_0s_comparison(root):
    run = write_ledger(root / "org--a")
    (run / "research").mkdir()
    (run / "research" / "base-candidates.json").write_text(json.dumps(REPORT))
    (run / "stages" / "0").mkdir(parents=True)
    (run / "stages" / "0" / "delta.json").write_text(json.dumps(
        {"base_research": {"candidate": "google/g-it", "estimated_class": "weights-only",
                           "measured_class": "weights-only", "agreed": True}}))
    out = make_app(root).run_research("org--a")
    assert out["report"]["choices"][0]["name"] == "stisi/g-it-p150"
    assert out["stage0"] == {"candidate": "google/g-it", "estimated_class": "weights-only",
                             "measured_class": "weights-only", "agreed": True}


def test_a_run_without_research_says_so(root):
    write_ledger(root / "org--a")
    assert make_app(root).run_research("org--a") == {"report": None, "stage0": None}


def test_an_oversized_or_symlinked_report_is_not_read(root, tmp_path):
    run = write_ledger(root / "org--a")
    (run / "research").mkdir()
    (run / "research" / "base-candidates.json").write_text(json.dumps(REPORT) + " " * webui.REPORT_MAX_BYTES)
    assert make_app(root).run_research("org--a")["report"] is None
    outside = tmp_path / "secret.json"
    outside.write_text(json.dumps(REPORT))
    (run / "research" / "base-candidates.json").unlink()
    (run / "research" / "base-candidates.json").symlink_to(outside)
    assert make_app(root).run_research("org--a")["report"] is None


def test_the_stage_0_record_only_passes_known_fields(root):
    run = write_ledger(root / "org--a")
    (run / "stages" / "0").mkdir(parents=True)
    (run / "stages" / "0" / "delta.json").write_text(json.dumps(
        {"base_research": {"candidate": "x/y", "estimated_class": "weights-only", "measured_class": "full-port",
                           "agreed": False, "note": f"{HOME} hf_{'b' * 34}"}}))
    assert "note" not in make_app(root).run_research("org--a")["stage0"]


def _fake_signals(calls, options_seen):
    from orchard.preflight import Signals

    def make(cfg, research_options=None):
        options_seen.append(research_options)
        return Signals(hub_info=None, local_snapshot=None, free_gb=None, same_device=None, credentials=None,
                       port_in_use=None, gozer_status=None, load_tiers=None,
                       research=lambda model_id, hub, override, inst: calls.append((model_id, override)) or REPORT)
    return make


def _ui(monkeypatch):
    from pathlib import Path
    from orchard import cli
    calls, seen, reports = [], [], []
    monkeypatch.setattr(cli.preflight, "default_signals", _fake_signals(calls, seen))
    monkeypatch.setattr(cli.preflight, "run_preflight",
                        lambda cfg, model, signals=None, **k: reports.append(signals.research(model, None, None, [])) or [])
    monkeypatch.setattr(cli.bringup_config, "run_dir", lambda c, m: Path("/nonexistent/x"))
    return cli, calls, seen


def test_the_ui_preflight_asks_for_research_without_keeping_a_cache(monkeypatch):
    cli, calls, seen = _ui(monkeypatch)
    run = cli._ui_preflight(object())
    run("org/new", None, research="no-github")
    assert seen[-1].cache_dir is None and seen[-1].no_github and not seen[-1].offline
    run("org/new", None, research="offline")
    assert seen[-1].offline and seen[-1].no_github
    run("org/new", None)
    assert not seen[-1].offline and not seen[-1].no_github


def test_the_ui_preflight_remembers_an_answer_for_a_while(monkeypatch):
    cli, calls, seen = _ui(monkeypatch)
    clock = [1000.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    run = cli._ui_preflight(object())
    run("org/new", None)
    run("org/new", None)
    assert len(calls) == 1                                        # the second form edit made no request
    run("org/other", None)
    run("org/new", None, research="no-github")
    assert len(calls) == 3                                        # a different model or option asks again
    clock[0] += cli.UI_RESEARCH_TTL_S + 1
    run("org/new", None)
    assert len(calls) == 4
