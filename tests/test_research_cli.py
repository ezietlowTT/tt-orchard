# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard bringup` and the research report: flags, files, and the input handed to the supervisor."""
import json

from test_cli import Pipe, Recorder, run, signals  # noqa: F401  (the existing helpers)
from test_cli import conf  # noqa: F401  (a pytest fixture)

from orchard import cli, preflight

REPORT = {"model": "Cloudflare/clef", "generated_at": "2026-10-12T00:00:00Z", "offline": False, "stated_bases": [],
          "candidates": [], "bundle_choices": [], "selected": None, "ports": [], "evidence": [], "unavailable": []}


def with_report(**over):
    return signals(research=lambda model_id, hub, override, inst: REPORT, **over)


def run_dir_of(tmp_path):
    return tmp_path / "runs" / "cloudflare--clef"


def test_the_new_flags_exist_and_default_off():
    ns = cli._parser().parse_args(["bringup", "a/b"])
    assert (ns.offline, ns.no_github, ns.refresh_research) == (False, False, False)
    ns = cli._parser().parse_args(["bringup", "a/b", "--offline", "--no-github", "--refresh-research"])
    assert (ns.offline, ns.no_github, ns.refresh_research) == (True, True, True)


def spied_main(conf, args, monkeypatch):
    """cli.main with no injected signals, so cli builds them; returns the options it passed on."""
    seen = {}

    def fake_default(cfg, research_options=None):
        seen["options"] = research_options
        return signals()
    monkeypatch.setattr(preflight, "default_signals", fake_default)
    rec = Recorder()
    code = cli.main(["--config", str(conf), *args], env={}, stdout=Pipe(), supervisor_main=rec.supervisor,
                    fetcher=rec.fetcher)
    return seen["options"], code, rec


def test_a_dry_run_builds_signals_with_no_cache_directory(conf, monkeypatch):
    options, code, rec = spied_main(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--offline"], monkeypatch)
    assert options.cache_dir is None and options.offline is True and rec.argv is None


def test_a_real_start_caches_under_the_run_directory_and_passes_the_flags_on(conf, tmp_path, monkeypatch):
    options, code, rec = spied_main(conf, ["bringup", "Cloudflare/clef", "--no-github", "--refresh-research"],
                                    monkeypatch)
    assert options.cache_dir == run_dir_of(tmp_path) / "research" / "cache"
    assert (options.no_github, options.refresh, options.offline) == (True, True, False)


def test_the_report_is_written_and_handed_to_the_supervisor_as_an_input(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=with_report())
    path = run_dir_of(tmp_path) / "research" / "base-candidates.json"
    assert code == 0 and json.loads(path.read_text())["model"] == "Cloudflare/clef"
    i = rec.argv.index(f"base_candidates={path}")
    assert rec.argv[i - 1] == "--input"


def test_a_dry_run_never_writes_the_report_or_names_it_in_the_command(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], sig=with_report())
    assert code == 0 and rec.argv is None
    assert not (run_dir_of(tmp_path) / "research").exists() and "base_candidates" not in out


def test_a_blocked_preflight_writes_no_report_and_starts_nothing(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=with_report(gozer_status=lambda: ""))
    assert code == 2 and rec.argv is None and not (run_dir_of(tmp_path) / "research").exists()


def test_without_a_report_no_input_is_added(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=signals())
    assert code == 0 and not any("base_candidates" in a for a in rec.argv)


def test_a_model_id_that_is_not_org_name_gets_no_report_file(conf, tmp_path):
    # --run-dir skips the slug check, so the report step checks the id itself
    rd = tmp_path / "elsewhere"
    code, out, rec = run(conf, ["bringup", "bad id/x\n# y", "--run-dir", str(rd)], sig=with_report())
    assert not (rd / "research").exists()
    assert rec.argv is None or not any("base_candidates" in a for a in rec.argv)
