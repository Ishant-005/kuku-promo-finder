"""run.py stage-2 loop: skip/force, LLM_SLEEP_S, error isolation. No network."""
import pytest

import run
from src import config
from src.models import Beat, BeatKind, Episode, ExtractionResult, Candidate


def _ep(n: int) -> Episode:
    b = Beat(id=f"e{n:03d}-b001", scene_no=1, kind=BeatKind.DIALOGUE,
             start_s=0.0, end_s=1.0, text="x")
    return Episode(number=n, beats=[b], source=f"ep{n}.txt")


def _ok(ep):
    return ExtractionResult(
        episode=ep.number, model="m",
        accepted=[Candidate(candidate_id=f"e{ep.number:03d}-c1")],
        rejected=[Candidate(candidate_id="r1", reject_reason="too_short: 12.0s"),
                  Candidate(candidate_id="r2", reject_reason="duplicate_of:e001-c1")],
    )


def test_sleep_between_calls_only(tmp_path):
    sleeps = []
    stats = run.run_extract([_ep(1), _ep(2), _ep(3)], tmp_path, force=False, sleep_s=4,
                            extract_fn=_ok, sleep_fn=sleeps.append)
    assert sleeps == [4, 4]  # between episodes, not before the first or after the last
    assert stats["done"] == 3 and stats["accepted"] == 3
    assert stats["rejected"] == {"too_short": 3, "duplicate_of": 3}


def test_skip_existing_unless_force(tmp_path):
    run.run_extract([_ep(1)], tmp_path, force=False, sleep_s=0, extract_fn=_ok)
    calls = []

    def counting(ep):
        calls.append(ep.number)
        return _ok(ep)

    s1 = run.run_extract([_ep(1), _ep(2)], tmp_path, force=False, sleep_s=0, extract_fn=counting)
    assert calls == [2] and s1["skipped"] == 1
    s2 = run.run_extract([_ep(1), _ep(2)], tmp_path, force=True, sleep_s=0, extract_fn=counting)
    assert calls == [2, 1, 2] and s2["skipped"] == 0


def test_failure_logged_and_run_continues(tmp_path):
    def flaky(ep):
        if ep.number == 1:
            raise RuntimeError("boom")
        return _ok(ep)

    stats = run.run_extract([_ep(1), _ep(2)], tmp_path, force=False, sleep_s=0, extract_fn=flaky)
    assert stats["failed"] == 1 and stats["done"] == 1
    log = (tmp_path / "logs" / "errors.log").read_text()
    assert "ep001" in log and "RuntimeError: boom" in log
    assert (tmp_path / "episodes" / "ep002.json").exists()


@pytest.mark.parametrize("val,expected", [(None, 4.0), ("0", 0.0), ("2.5", 2.5)])
def test_sleep_seconds_env(monkeypatch, val, expected):
    monkeypatch.setattr(config, "_loaded", True)
    if val is None:
        monkeypatch.delenv("LLM_SLEEP_S", raising=False)
    else:
        monkeypatch.setenv("LLM_SLEEP_S", val)
    assert run.sleep_seconds() == expected


def test_parse_episode_list():
    assert run.parse_episode_list("1, 10,25") == {1, 10, 25}
    assert run.parse_episode_list(None) is None


def test_durations_logged_and_printed(tmp_path, capsys):
    def ex(ep):
        return ExtractionResult(episode=ep.number, accepted=[
            Candidate(candidate_id="e001-c2", duration_s=30.0, declared_duration_s=28.0,
                      duration_delta_s=-2.0)],
            rejected=[Candidate(candidate_id="e001-c1", duration_s=14.5,
                                reject_reason="too_short: 14.5s")])
    stats = run.run_extract([_ep(1)], tmp_path, force=False, sleep_s=0, extract_fn=ex)
    lines = (tmp_path / "logs" / "durations.jsonl").read_text().splitlines()
    rows = [__import__("json").loads(x) for x in lines]
    assert [r["candidate_id"] for r in rows] == ["e001-c1", "e001-c2"]
    assert rows[1]["declared_s"] == 28.0 and rows[1]["delta_s"] == -2.0
    assert rows[0]["declared_s"] is None and rows[0]["actual_s"] == 14.5
    run.print_durations(stats["durations"])
    out = capsys.readouterr().out
    assert "e001-c1" in out and "too_short" in out and "declared missing: 1" in out
    assert "|delta| mean 2.0s  max 2.0s" in out
