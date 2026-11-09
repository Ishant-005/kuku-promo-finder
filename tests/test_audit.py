"""Stage 4 audit + stats. Synthetic data only; no network, no data/."""
import json

import pytest

from src import audit, stats
from src.models import Beat, BeatKind, Episode


def _ep(n=7):
    spec = [(BeatKind.CAMERA, None, "A hall."), (BeatKind.DIALOGUE, "ANNA (cold)", "Leave."),
            (BeatKind.DIALOGUE, "TOM", "No."), (BeatKind.OTHER, None, "Thunder."),
            (BeatKind.ACTION, None, "She slaps him."), (BeatKind.DIALOGUE, "anna", "Go!"),
            (BeatKind.DIALOGUE, "TOM", "Who | are you?")]
    beats = [Beat(id=f"e{n:03d}-b{i + 1:03d}", scene_no=1, kind=k, speaker=sp,
                  start_s=i * 5.0, end_s=i * 5.0 + 5, text=t)
             for i, (k, sp, t) in enumerate(spec)]
    return Episode(number=n, logline="SECRET LOGLINE", beats=beats, source="x.txt")


def _promo(rank=1, segs=((2, 3), (5, 7)), ep=7, standalone=4):
    return {"rank": rank, "candidate_id": f"e{ep:03d}-c{rank}", "episode": ep,
            "standalone_score": standalone,
            "segments": [{"start_id": f"e{ep:03d}-b{a:03d}", "end_id": f"e{ep:03d}-b{b:03d}"}
                         for a, b in segs]}


GOOD = {"what_is_happening": "A woman throws a man out.", "who_are_they": "A couple.",
        "open_question": "Who is he?", "followable_score": 2, "would_tap_next": True}


# --- clip text ------------------------------------------------------------------

def test_clip_text_only_clip_beats_anonymised_with_cuts():
    text, n = audit.build_clip_text(_promo(), _ep())
    assert n == 5
    assert text.splitlines() == [
        "Speaker A: Leave.", "Speaker B: No.", "[cut]",
        "[She slaps him.]", "Speaker A: Go!", "Speaker B: Who | are you?"]
    for leak in ("ANNA", "TOM", "SECRET LOGLINE", "A hall.", "Thunder", "7"):
        assert leak not in text


def test_clip_text_sound_fx_marked():
    text, _ = audit.build_clip_text(_promo(segs=((4, 5),)), _ep())
    assert text.splitlines() == ["[sound: Thunder.]", "[She slaps him.]"]


# --- audit_promo --------------------------------------------------------------------

def _fake(raw, seen=None):
    def f(system, user, **kw):
        if seen is not None:
            seen.update(system=system, user=user, kw=kw)
        audit.llm.last_model, audit.llm.last_response_model = "m", "m-s"
        return raw
    return f


def test_audit_promo_ok_and_delta():
    seen = {}
    it = audit.audit_promo(_promo(), _ep(), complete_fn=_fake(json.dumps(GOOD), seen))
    assert it.error is None and it.followable_score == 2 and it.delta == -2
    assert it.would_tap_next is True and it.model == "m" and it.response_model == "m-s"
    assert seen["system"] == audit.load_prompt()
    assert seen["system"].startswith("You are a stranger scrolling TikTok. You see this clip cold.")
    assert seen["kw"] == {"json_mode": True, "max_tokens": audit.AUDIT_MAX_TOKENS}
    assert it.prompt_hash == audit.llm._prompt_hash(seen["system"], seen["user"])


@pytest.mark.parametrize("raw,frag", [
    (json.dumps({**GOOD, "followable_score": 9}), "malformed: bad fields followable_score"),
    (json.dumps({k: v for k, v in GOOD.items() if k != "who_are_they"}), "who_are_they"),
    ("not json", "ExtractionError"),
])
def test_audit_promo_bad_output_recorded(raw, frag):
    it = audit.audit_promo(_promo(), _ep(), complete_fn=_fake(raw))
    assert it.followable_score is None and frag in it.error


def test_audit_promo_unknown_id_and_missing_episode():
    bad = _promo()
    bad["segments"][0]["start_id"] = "e007-b999"
    assert "unknown beat id" in audit.audit_promo(bad, _ep(), complete_fn=_fake("{}")).error
    assert "not found" in audit.audit_promo(_promo(), None, complete_fn=_fake("{}")).error


# --- audit_all + ranked.md ------------------------------------------------------------

def _write_promos(out, promos):
    out.mkdir(parents=True, exist_ok=True)
    (out / "promos.json").write_text(json.dumps({"promos": promos}))
    (out / "ranked.md").write_text("# Ranked promos\n\n| rank |\n|---|\n| 1 |\n")


def test_audit_all_sleeps_between_writes_and_replaces_md_section(tmp_path):
    _write_promos(tmp_path, [_promo(1), _promo(2, standalone=2), _promo(3, ep=99)])
    sleeps = []
    for _ in range(2):  # rerun must not duplicate the audit table
        res = audit.audit_all(tmp_path, [_ep()], complete_fn=_fake(json.dumps(GOOD)),
                              sleep_s=5, sleep_fn=sleeps.append, log=lambda *_: None)
    assert sleeps == [5, 5] * 2
    assert res["audited"] == 3 and res["ok"] == 2 and res["errors"] == 1
    assert res["big_deltas"] == 1                       # rank 1: 2 - 4 = -2
    data = json.loads((tmp_path / "audit.json").read_text())
    assert [i["rank"] for i in data["items"]] == [1, 2, 3]
    md = (tmp_path / "ranked.md").read_text()
    assert md.startswith("# Ranked promos") and md.count(audit.MD_MARKER) == 1
    assert "| 1 | 4 | 2 | -2 | True | A woman throws a man out. |" in md
    assert "| 2 | 2 | 2 | +0 |" in md and "ERROR: episode 99 not found" in md


@pytest.mark.parametrize("val,expected", [(None, 5.0), ("2.5", 2.5), ("-1", 0.0)])
def test_audit_sleep_env(monkeypatch, val, expected):
    monkeypatch.setattr(audit.config, "_loaded", True)
    if val is None:
        monkeypatch.delenv("AUDIT_SLEEP_S", raising=False)
    else:
        monkeypatch.setenv("AUDIT_SLEEP_S", val)
    assert audit.sleep_seconds() == expected


# --- stats ------------------------------------------------------------------------

def _cand(cid, h, c, s, reason=None):
    return {"candidate_id": cid, "hook_score": h, "cliff_score": c, "standalone_score": s,
            "reject_reason": reason}


def _stats_fixture(out, monkeypatch):
    monkeypatch.setattr(stats.config, "_loaded", True)
    monkeypatch.setenv("LLM_MODEL", "big")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "small")
    d = out / "episodes"
    d.mkdir(parents=True)
    eps = [
        {"episode": 1, "model": "big", "prompt_hash": "h1", "episode_note": "strong",
         "accepted": [_cand("e001-c1", 5, 5, 4), _cand("e001-c2", 5, 4, 3)],
         "rejected": [_cand("e001-c3", 4, 4, 4, "too_short: 12.0s")]},
        {"episode": 2, "model": "small", "prompt_hash": "h2",
         "episode_note": "Mostly exposition | no hook", "accepted": [],
         "rejected": [_cand("e002-c1", 3, 3, 3, "duplicate_of:e002-c2"),
                      _cand("e002-c2", 3, 3, 3, "cuts_mid_beat: x")]},
        {"episode": 3, "model": "big", "prompt_hash": "h3", "episode_note": "ok",
         "accepted": [_cand("e003-c1", 5, 5, 5)], "rejected": []},
    ]
    for e in eps:
        (d / f"ep{e['episode']:03d}.json").write_text(json.dumps(e))
    (out / "promos.json").write_text(json.dumps({"prompt_hash": "hr", "promos": []}))
    (out / "audit.json").write_text(json.dumps({"items": [
        {"prompt_hash": "ha", "followable_score": 3, "delta": -2, "would_tap_next": True},
        {"prompt_hash": "hb", "error": "boom"}]}))
    logs = out / "logs"
    logs.mkdir()
    u = lambda i, o: {"prompt_tokens": i, "completion_tokens": o}  # noqa: E731
    recs = [
        {"ts": "2026-11-08T12:00:00+00:00", "prompt_hash": "h1", "usage": u(1000, 100),
         "latency_ms": 2000},
        {"ts": "2026-11-08T12:00:30+00:00", "prompt_hash": "h2", "usage": None,
         "latency_ms": 100, "error": "429"},
        {"ts": "2026-11-08T12:01:00+00:00", "prompt_hash": "h2", "usage": u(2000, 300),
         "latency_ms": 3000, "fallback_used": True},
        {"ts": "2026-11-08T13:30:00+00:00", "prompt_hash": "ha", "usage": u(500, 50),
         "latency_ms": 900},
        {"ts": "2026-11-07T09:00:00+00:00", "prompt_hash": "dev", "usage": u(9000, 900),
         "latency_ms": 5000},
    ]
    (logs / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\nbroken\n")


def test_stats_numbers(tmp_path, monkeypatch):
    _stats_fixture(tmp_path, monkeypatch)
    s = stats.build_stats(tmp_path, n_parsed=50)
    assert s["n_episodes"] == 3 and s["proposed"] == 6 and s["accepted"] == 3
    assert s["rejected"] == {"too_short": 1, "duplicate_of": 1, "cuts_mid_beat": 1}
    assert s["dist"]["hook_score"] == {5: 3} and s["dist"]["standalone_score"] == {4: 1, 3: 1, 5: 1}
    assert [z["episode"] for z in s["zero"]] == [2]
    assert s["models"] == {"primary": 2, "fallback": 1}
    p, a = s["pipeline"], s["all"]
    assert (p["calls"], p["ok"], p["errors"]) == (4, 3, 1)
    assert (p["prompt_tokens"], p["completion_tokens"]) == (3500, 450)
    assert p["cost_usd"] == pytest.approx(3500 / 1e6 * 0.30 + 450 / 1e6 * 2.50)
    assert p["llm_seconds"] == pytest.approx(6.0) and p["fallback_records"] == 1
    assert a["calls"] == 5 and a["prompt_tokens"] == 12500


def test_stats_markdown(tmp_path, monkeypatch):
    _stats_fixture(tmp_path, monkeypatch)
    path, _ = stats.write_stats(tmp_path, n_parsed=50, stage_seconds=12.3)
    md = path.read_text()
    assert "- screenplays parsed: 50" in md and "- episodes processed (with output): 3" in md
    assert "answered by primary (big): 2" in md and "answered by fallback (small): 1" in md
    assert "| hook_score | 0 | 0 | 0 | 0 | 3 | 5.00 | 100% |" in md
    assert "| 2 | cuts_mid_beat 1, duplicate_of 1 | Mostly exposition / no hook |" in md \
        or "| 2 | duplicate_of 1, cuts_mid_beat 1 | Mostly exposition / no hook |" in md
    assert "| pipeline | 4 | 3 | 1 | 3,500 | 450 |" in md
    assert "1h30m" in md and "this stage-4 invocation: 12s" in md
    assert "- audited: 2 (1 errors)" in md and "|delta| ≥ 2: 1" in md


def test_rank_prompt_hash_rebuilt_from_top30(tmp_path):
    from src import rank
    from src.models import ResolvedSegment
    p = rank.Promo(episode=1, candidate_id="e001-c1", duration_s=30.0, hook="h", cliffhanger="c",
                   why_it_works="w", hook_type="slap",
                   segments_resolved=[ResolvedSegment(start_id="a", end_id="b", start_s=0,
                                                      end_s=30)])
    expected = rank.llm._prompt_hash(rank.load_prompt(), rank.build_cards([p]))
    assert stats.rank_prompt_hash({"top30": [p.model_dump()]}) == expected
    assert stats.rank_prompt_hash({}) is None
