# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard bringup` and the research report: flags, files, and the input handed to the supervisor."""
import json
from pathlib import Path

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
    full = run_dir_of(tmp_path) / "research" / "base-candidates.json"
    view = run_dir_of(tmp_path) / "research" / "base-candidates.stage0.json"
    assert code == 0 and json.loads(full.read_text())["model"] == "Cloudflare/clef"
    # Stage 0 gets the candidates-only view. The full report holds text from the internet.
    i = rec.argv.index(f"base_candidates={view}")
    assert rec.argv[i - 1] == "--input"
    assert not any(a == f"base_candidates={full}" for a in rec.argv)
    assert set(json.loads(view.read_text())) == {"model", "selected", "candidates"}


def test_a_dry_run_never_writes_the_report_or_names_it_in_the_command(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], sig=with_report())
    assert code == 0 and rec.argv is None
    assert not (run_dir_of(tmp_path) / "research").exists() and "base_candidates" not in out


def test_a_blocked_preflight_writes_no_report_and_starts_no_supervisor(conf, tmp_path):
    # The research cache (research/cache) is filled by the fetcher during the preflight, so it can exist after a
    # block. This test injects signals that make no fetch, and checks only the report files and the supervisor.
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=with_report(gozer_status=lambda: ""))
    research = run_dir_of(tmp_path) / "research"
    assert code == 2 and rec.argv is None
    assert not list(research.glob("base-candidates*")) if research.exists() else True


def test_without_a_report_no_input_is_added(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=signals())
    assert code == 0 and not any("base_candidates" in a for a in rec.argv)


def test_a_model_id_that_is_not_org_name_gets_no_report_file(conf, tmp_path):
    # --run-dir skips the slug check, so the report step checks the id itself. The preflight passes here (the fake
    # signals answer for any id and the research callable returns a report), so the run starts. The valid_id check
    # at the report step is what keeps the report and the supervisor input out.
    rd = tmp_path / "elsewhere"
    code, out, rec = run(conf, ["bringup", "bad id/x\n# y", "--run-dir", str(rd)], sig=with_report())
    assert code == 0 and rec.argv is not None                    # the run went past the preflight and the report step
    assert not (rd / "research").exists()                         # and wrote no report for the odd id
    assert not any("base_candidates" in a for a in rec.argv)


def test_the_help_for_the_flags_says_what_they_cover():
    helps = {a.option_strings[0]: a.help for a in cli._parser()._subparsers._group_actions[0].choices["bringup"]._actions
             if a.option_strings}
    assert "tt-metal" in helps["--no-github"] and "issue" in helps["--no-github"]
    assert "hub lookup" in helps["--offline"] and "download" in helps["--offline"]


def test_the_readme_says_what_offline_and_no_github_cover():
    text = " ".join((Path(__file__).resolve().parent.parent / "README.md").read_text().split())
    assert "`--offline` skips the research and makes no request for it" in text
    assert "still read" in text and "`--dry-run` does the same lookups as a real run (unless `--offline`)" in text
    assert "can leave that directory behind" in text
