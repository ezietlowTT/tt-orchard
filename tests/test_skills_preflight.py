# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `skills` row of the bringup preflight: is the tenstorrent/skills clone the latest? (Task 14)

Bringup runs unattended, so a stale clone is a warning and never a block. The row comes from the injected
`skills_freshness` signal; no test here touches git or the network."""
from __future__ import annotations

from orchard import preflight as pf
from orchard import skillsfresh as sf
from test_preflight import cfg, signals

ROOT = "/home/op/code/skills"


def fresh(state, **kw):
    detail = kw.pop("detail", f"HEAD is {state}.")
    return sf.Freshness(sf.SOURCE_SKILLS, ROOT, state, detail, **kw)


def test_a_current_clone_is_ok():
    c = pf.check_skills([fresh(sf.CURRENT)])
    assert c.status == pf.OK and ROOT in c.detail and "current" in c.detail


def test_a_clone_behind_main_is_a_warning_with_the_update_command_and_never_a_block():
    c = pf.check_skills([fresh(sf.BEHIND, behind=3, update=f"git -C {ROOT} pull --ff-only origin main")])
    assert c.status == pf.WARN and c.reason is None
    assert "behind" in c.detail and f"git -C {ROOT} pull --ff-only origin main" in c.detail


def test_an_unknown_comparison_is_a_warning_that_says_it_was_not_compared():
    c = pf.check_skills([fresh(sf.UNKNOWN, detail="It could not be compared with upstream (offline).")])
    assert c.status == pf.WARN and "could not be compared" in c.detail


def test_a_current_clone_with_a_dirty_tree_on_a_branch_is_a_warning_that_names_both():
    c = pf.check_skills([fresh(sf.CURRENT, warnings=[f"{ROOT} has uncommitted changes.",
                                                     f"{ROOT} is on branch tsingletaryTT/squeeze; it is compared "
                                                     "with upstream main."])])
    assert c.status == pf.WARN and "uncommitted changes" in c.detail and "tsingletaryTT/squeeze" in c.detail


def test_no_skills_dirs_is_a_warning():
    c = pf.check_skills([])
    assert c.status == pf.WARN and "skills_dirs" in c.detail


def test_the_row_runs_after_the_other_checks_and_a_stale_clone_blocks_nothing(tmp_path):
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(skills_freshness=lambda: [fresh(sf.BEHIND, behind=3, update="u")]))
    assert [c.name for c in out][-1] == "skills"
    assert out[-1].status == pf.WARN and not pf.blocked(out)


def test_a_failing_freshness_signal_is_a_warning_and_never_a_block(tmp_path):
    def boom():
        raise OSError("git missing")
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(skills_freshness=boom))
    row = out[-1]
    assert row.name == "skills" and row.status == pf.WARN and "could not check" in row.detail
    assert not pf.blocked(out)


def test_without_the_signal_there_is_no_row(tmp_path):
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals())
    assert "skills" not in [c.name for c in out]


def test_the_real_signal_checks_the_configured_skills_dirs_read_only_and_reuses_a_recent_check(tmp_path, monkeypatch):
    seen = {}

    def fake(dirs, **kw):
        seen["dirs"] = dirs
        seen.update(kw)
        return []
    monkeypatch.setattr(sf, "check_skills_dirs", fake)
    c = cfg(tmp_path)
    c.skills_dirs = [tmp_path / "skills" / "plugins" / "tt-model-bringup" / "skills"]
    pf.default_signals(c).skills_freshness()
    assert seen["dirs"] == c.skills_dirs and seen["reuse_s"] == pf.SKILLS_REUSE_S and seen["offline"] is False


def test_the_real_signal_is_offline_when_research_is_offline(tmp_path, monkeypatch):
    from orchard.research import Options
    seen = {}
    monkeypatch.setattr(sf, "check_skills_dirs", lambda dirs, **kw: seen.update(kw) or [])
    pf.default_signals(cfg(tmp_path), research_options=Options(offline=True)).skills_freshness()
    assert seen["offline"] is True


def test_the_dry_run_page_shows_the_skills_row(tmp_path):
    import io
    from test_cli import Pipe, Recorder, signals as cli_signals
    from orchard import cli
    conf = tmp_path / "bringup.toml"
    conf.write_text(f'runs_root = "{tmp_path}/runs"\nhf_home = "{tmp_path}/hf"\ncache_root = "{tmp_path}/cache"\n'
                    '[coder]\ntarget = "pkg/coder"\nport = 8000\nchips = 4\n')
    rec, out = Recorder(), Pipe()
    sig = cli_signals(skills_freshness=lambda: [fresh(sf.BEHIND, behind=3,
                                                      update=f"git -C {ROOT} pull --ff-only origin main")])
    code = cli.main(["--config", str(conf), "bringup", "Cloudflare/clef", "--dry-run"], env={}, stdout=out,
                    signals=sig, supervisor_main=rec.supervisor, fetcher=rec.fetcher)
    text = out.getvalue()
    assert code == 0 and rec.argv is None
    assert "skills" in text and "behind" in text and "pull --ff-only" in text
