"""Extraction validator tests. Synthetic episode only; no network, no data/."""
import json

import pytest

from src import extract
from src.models import Beat, BeatKind, Episode


def _ep() -> Episode:
    # 12 beats, 5 s each, back to back: b001 0-5 ... b012 55-60.
    kinds = [BeatKind.CAMERA, BeatKind.DIALOGUE, BeatKind.ACTION, BeatKind.DIALOGUE,
             BeatKind.OTHER, BeatKind.DIALOGUE, BeatKind.DIALOGUE, BeatKind.ACTION,
             BeatKind.DIALOGUE, BeatKind.CAMERA, BeatKind.DIALOGUE, BeatKind.DIALOGUE]
    beats = [
        Beat(id=f"e007-b{i + 1:03d}", scene_no=1, kind=k,
             speaker="ANNA" if k is BeatKind.DIALOGUE else None,
             start_s=i * 5.0, end_s=i * 5.0 + 5.0, text=f"line {i + 1}")
        for i, k in enumerate(kinds)
    ]
    return Episode(number=7, logline="A test.", beats=beats, source="ep7.txt")


def _cand(segments, **kw):
    base = {"segments": [{"start_id": f"e007-b{a:03d}", "end_id": f"e007-b{b:03d}"}
                         for a, b in segments],
            "hook": "h", "cliffhanger": "c", "hook_score": 4, "cliff_score": 4,
            "standalone_score": 3, "standalone_reason": "r", "why_it_works": "w",
            "hook_type": "slap"}
    base.update(kw)
    return base


FAKE = {
    "candidates": [
        _cand([(2, 7)]),                                    # c1 ok: 5-35 = 30s
        _cand([(2, 3), (6, 9)]),                            # c2 ok: 10 + 20 = 30s
        {**_cand([(2, 7)]), "segments": [{"start_id": "e007-b002", "end_id": "e007-b099"}]},  # c3
        _cand([(6, 9), (2, 4)]),                            # c4 out of order
        _cand([(2, 6), (5, 9)]),                            # c5 overlap
        _cand([(1, 6)]),                                    # c6 starts on CAMERA: ok, 30s
        _cand([(5, 10)]),                                   # c7 starts on SOUND FX (OTHER)
        _cand([(2, 4)]),                                    # c8 15s too short
        _cand([(2, 12)]),                                   # c9 55s ok
        _cand([(2, 6), (8, 12)]),                           # c10 25 + 25 = 50s ok
        _cand([(7, 4)]),                                    # c11 start after end
        _cand([]),                                          # c12 empty
        {**_cand([(2, 7)]), "hook_score": 9},              # c13 malformed score
        _cand([(2, 7)], hook_type="Shocking Line"),        # c14 ok, coerced
        _cand([(2, 7)], hook_type="weird"),                # c15 ok, -> other
    ],
    "episode_note": "test note",
}


@pytest.fixture
def result():
    return extract.validate_response(json.dumps(FAKE), _ep())


def _by_id(result):
    return {c.candidate_id: c for c in result.accepted + result.rejected}


def test_accepted_resolve_timestamps(result):
    acc = {c.candidate_id for c in result.accepted}
    assert acc == {"e007-c1", "e007-c2", "e007-c6", "e007-c9", "e007-c10", "e007-c14",
                   "e007-c15"}
    c = _by_id(result)
    assert c["e007-c1"].duration_s == 30.0
    assert [(s.start_s, s.end_s) for s in c["e007-c1"].segments_resolved] == [(5.0, 35.0)]
    assert [(s.start_s, s.end_s) for s in c["e007-c2"].segments_resolved] == [
        (5.0, 15.0), (25.0, 45.0)]
    assert c["e007-c2"].duration_s == 30.0
    assert all(x.reject_reason is None for x in result.accepted)
    assert result.episode_note == "test note"


@pytest.mark.parametrize("cid,reason", [
    ("e007-c3", extract.R_UNKNOWN_ID),
    ("e007-c4", extract.R_ORDER),
    ("e007-c5", extract.R_ORDER),
    ("e007-c7", extract.R_FIRST_BEAT),
    ("e007-c8", extract.R_SHORT),
    ("e007-c11", extract.R_BAD_SEGMENT),
    ("e007-c12", extract.R_EMPTY),
    ("e007-c13", extract.R_MALFORMED),
])
def test_each_reject_rule_fires(result, cid, reason):
    c = _by_id(result)[cid]
    assert c in result.rejected
    assert c.reject_reason.startswith(reason), c.reject_reason
    assert not c.segments_resolved or reason in (extract.R_SHORT, extract.R_LONG)


def test_unknown_id_names_the_bad_id(result):
    assert "e007-b099" in _by_id(result)["e007-c3"].reject_reason


def test_malformed_reason_has_no_input_values(result):
    r = _by_id(result)["e007-c13"].reject_reason
    assert "hook_score" in r and "9" not in r


def test_too_long():
    ep = _ep()
    # stretch the last beat so b002..b012 spans 5..75 = 70s
    ep.beats[-1] = ep.beats[-1].model_copy(update={"end_s": 75.0})
    res = extract.validate_response(json.dumps({"candidates": [_cand([(2, 12)])]}), ep)
    assert res.rejected[0].reject_reason.startswith(extract.R_LONG)
    assert res.rejected[0].duration_s == 70.0


def test_bounds_inclusive():
    ep = _ep()
    # exactly 20s (b002-b005 = 5..25) and exactly 60s via stretched end
    ep.beats[4] = ep.beats[4].model_copy(update={"kind": BeatKind.DIALOGUE})
    ep.beats[-1] = ep.beats[-1].model_copy(update={"end_s": 65.0})
    res = extract.validate_response(
        json.dumps({"candidates": [_cand([(2, 5)]), _cand([(2, 12)])]}), ep)
    assert [c.duration_s for c in res.accepted] == [20.0, 60.0]


def test_hook_type_coercion(result):
    c = _by_id(result)
    assert c["e007-c14"].hook_type == "shocking_line"
    assert c["e007-c15"].hook_type == "other"


@pytest.mark.parametrize("wrap", [
    lambda s: s,
    lambda s: f"```json\n{s}\n```",
    lambda s: f"```\n{s}\n```",
    lambda s: f"Here you go:\n{s}\nHope that helps.",
])
def test_tolerant_json(wrap):
    res = extract.validate_response(wrap(json.dumps(FAKE)), _ep())
    assert len(res.accepted) == 7


@pytest.mark.parametrize("raw", ["not json at all", "[1, 2]", '{"candidates": "nope"}'])
def test_unparseable_raises(raw):
    with pytest.raises(extract.ExtractionError):
        extract.validate_response(raw, _ep())


def test_missing_candidates_is_empty_not_error():
    res = extract.validate_response('{"dry_run": true}', _ep())
    assert res.accepted == [] and res.rejected == []


def test_beat_table_format_and_trim():
    ep = _ep()
    ep.beats[1] = ep.beats[1].model_copy(update={"text": "a|b  " + "x" * 300})
    table = extract.build_beat_table(ep).splitlines()
    assert table[0] == "id | kind | speaker | start | end | cum_start | cum_end | text"
    assert len(table) == 1 + len(ep.beats)
    assert table[1] == "e007-b001 | CAMERA | - | 0.0 | 5.0 | 0.0 | 5.0 | line 1"
    row = table[2].split(" | ")
    assert row[:7] == ["e007-b002", "DIALOGUE", "ANNA", "5.0", "10.0", "5.0", "10.0"]
    assert len(row[7]) == extract.TEXT_MAX and row[7].endswith("…")
    assert row[7].startswith("a/b x")


def test_beat_table_cum_is_relative_to_first_beat_one_decimal():
    ep = _ep()
    ep.beats = [b.model_copy(update={"start_s": b.start_s + 38.14, "end_s": b.end_s + 38.14})
                for b in ep.beats]
    rows = [r.split(" | ") for r in extract.build_beat_table(ep).splitlines()[1:]]
    assert rows[0][3:7] == ["38.1", "43.1", "0.0", "5.0"]
    assert rows[11][3:7] == ["93.1", "98.1", "55.0", "60.0"]
    # span b002..b007 = cum_end(b007) - cum_start(b002) = 35 - 5 = 30 = validator's duration
    assert float(rows[6][6]) - float(rows[1][5]) == 30.0


def test_declared_duration_delta_and_rows():
    raw = json.dumps({"candidates": [
        _cand([(2, 7)], declared_duration_s=28.5),   # actual 30 -> delta -1.5
        _cand([(2, 4)], declared_duration_s=21),     # actual 15 -> too_short, delta +6
        _cand([(2, 9)]),                             # no declared -> delta None
        _cand([(2, 99)], declared_duration_s=30),    # unknown id -> actual None
    ]})
    res = extract.validate_response(raw, _ep())
    c = _by_id(res)
    assert c["e007-c1"].declared_duration_s == 28.5 and c["e007-c1"].duration_delta_s == -1.5
    assert c["e007-c2"].reject_reason.startswith(extract.R_SHORT)
    assert c["e007-c2"].duration_delta_s == 6.0
    assert c["e007-c3"].declared_duration_s is None and c["e007-c3"].duration_delta_s is None
    assert c["e007-c4"].duration_s is None and c["e007-c4"].duration_delta_s is None
    rows = extract.duration_rows(res)
    assert [r["candidate_id"] for r in rows] == ["e007-c1", "e007-c2", "e007-c3", "e007-c4"]
    assert rows[1] == {"episode": 7, "candidate_id": "e007-c2", "declared_s": 21.0,
                       "actual_s": 15.0, "delta_s": 6.0,
                       "reject_reason": c["e007-c2"].reject_reason}


def test_mid_beat_reject_still_records_actual_duration():
    ep = _ep()
    ep.beats[7] = ep.beats[7].model_copy(  # DIALOGUE b008 starts before b007 ends
        update={"start_s": 33.0, "kind": BeatKind.DIALOGUE})
    res = extract.validate_response(
        json.dumps({"candidates": [_cand([(2, 7)], declared_duration_s=33)]}), ep)
    (c,) = res.rejected
    assert c.reject_reason.startswith(extract.R_MID_BEAT)
    assert c.duration_s == 30.0 and c.duration_delta_s == 3.0 and not c.segments_resolved


def test_extract_episode_uses_prompt_file_and_writes(monkeypatch, tmp_path):
    seen = {}

    def fake_complete(system, user, **kw):
        seen.update(system=system, user=user, kw=kw)
        extract.llm.last_model = "fake-fallback"
        extract.llm.last_response_model = "fake-fallback-2025"
        return "```json\n" + json.dumps(FAKE) + "\n```"

    monkeypatch.setattr(extract.llm, "complete", fake_complete)
    ep = _ep()
    res = extract.extract_episode(ep)
    assert seen["system"] == extract.load_prompt()
    assert "LOGLINE: A test." in seen["user"] and "e007-b012 |" in seen["user"]
    assert seen["kw"]["json_mode"] is True
    assert res.response_model == "fake-fallback-2025"
    assert res.model == "fake-fallback" and len(res.prompt_hash) == 12

    path = extract.write_result(res, tmp_path)
    assert path == tmp_path / "episodes" / "ep007.json"
    data = json.loads(path.read_text())
    # FAKE's 7 valid candidates all overlap c1 (or share its end beat): dedupe keeps c1.
    assert data["episode"] == 7 and [c["candidate_id"] for c in data["accepted"]] == ["e007-c1"]
    dups = [c for c in data["rejected"] if c["reject_reason"].startswith("duplicate_of:")]
    assert len(dups) == 6 and all(c["reject_reason"] == "duplicate_of:e007-c1" for c in dups)
    assert {"accepted", "rejected", "episode_note", "model"} <= data.keys()


# --- first-beat rule: only SOUND FX (kind OTHER) is banned -------------------

def test_camera_first_beat_allowed_sound_fx_banned():
    res = extract.validate_response(
        json.dumps({"candidates": [_cand([(1, 6)]), _cand([(5, 10)])]}), _ep())
    assert [c.candidate_id for c in res.accepted] == ["e007-c1"]
    assert res.rejected[0].reject_reason.startswith(extract.R_FIRST_BEAT)
    assert "OTHER" in res.rejected[0].reject_reason


# --- cuts_mid_beat ------------------------------------------------------------

def _one(ep, segments):
    res = extract.validate_response(json.dumps({"candidates": [_cand(segments)]}), ep)
    return (res.accepted + res.rejected)[0]


def _shift(ep, n, **upd):
    ep.beats[n - 1] = ep.beats[n - 1].model_copy(update=upd)


def test_mid_beat_dialogue_after_end_id_starts_before_cut():
    # The ep10 case: next line starts 0.2 s before the segment end and runs past it.
    ep = _ep()
    _shift(ep, 8, kind=BeatKind.DIALOGUE, start_s=34.8, end_s=37.0)
    c = _one(ep, [(2, 7)])                       # segment 5.0-35.0
    assert c.reject_reason.startswith(extract.R_MID_BEAT), c.reject_reason
    assert "e007-b008" in c.reject_reason and "end" in c.reject_reason


@pytest.mark.parametrize("kind", [BeatKind.CAMERA, BeatKind.ACTION, BeatKind.OTHER])
def test_non_dialogue_running_past_end_is_allowed(kind):
    ep = _ep()
    _shift(ep, 10, kind=kind, start_s=44.0, end_s=52.0)   # starts inside 5-45, ends after
    c = _one(ep, [(2, 9)])
    assert c.reject_reason is None, c.reject_reason


def test_mid_beat_dialogue_spanning_segment_start():
    ep = _ep()
    _shift(ep, 2, end_s=12.0)                    # DIALOGUE 5-12, segment starts at 10
    c = _one(ep, [(3, 8)])
    assert c.reject_reason.startswith(extract.R_MID_BEAT)
    assert "e007-b002" in c.reject_reason and "start" in c.reject_reason


def test_mid_beat_second_segment_start_checked():
    ep = _ep()
    _shift(ep, 6, start_s=25.0, end_s=37.0)      # DIALOGUE spans b007 start (30)
    c = _one(ep, [(2, 4), (7, 11)])
    assert c.reject_reason.startswith(extract.R_MID_BEAT)
    assert "segment 2 start" in c.reject_reason


@pytest.mark.parametrize("kind", [BeatKind.CAMERA, BeatKind.ACTION, BeatKind.OTHER])
def test_non_dialogue_spanning_segment_start_is_allowed(kind):
    ep = _ep()
    _shift(ep, 1, kind=kind, end_s=12.0)         # 0-12 runs over the 5.0 start
    c = _one(ep, [(2, 7)])
    assert c.reject_reason is None, c.reject_reason


@pytest.mark.parametrize("overrun,ok", [(0.3, True), (0.29, True), (0.31, False)])
def test_dialogue_end_tolerance(overrun, ok):
    ep = _ep()
    _shift(ep, 8, kind=BeatKind.DIALOGUE, start_s=34.0, end_s=35.0 + overrun)  # vs end 35.0
    c = _one(ep, [(2, 7)])
    assert (c.reject_reason is None) is ok, c.reject_reason


@pytest.mark.parametrize("early,ok", [(0.3, True), (0.31, False)])
def test_dialogue_start_tolerance(early, ok):
    ep = _ep()
    _shift(ep, 2, start_s=10.0 - early, end_s=11.0)     # DIALOGUE b002 spans start 10.0
    c = _one(ep, [(3, 8)])                               # segment starts on b003 (10.0)
    assert (c.reject_reason is None) is ok, c.reject_reason


def test_back_to_back_beats_are_not_mid_beat():
    c = _one(_ep(), [(2, 7)])                    # b008 starts exactly at 35.0
    assert c.reject_reason is None


# --- dedupe -------------------------------------------------------------------

def _dedupe(cands):
    res = extract.validate_response(json.dumps({"candidates": cands}), _ep())
    assert not res.rejected  # all valid before dedupe
    return extract.dedupe_accepted(res)


def _reasons(res):
    return {c.candidate_id: c.reject_reason for c in res.rejected}


def test_dedupe_shared_end_beat_keeps_higher_score():
    # c1 0-15 + 35-40 (20s), c2 15-40 (25s): overlap only 5s, but both end on b008.
    res = _dedupe([_cand([(1, 3), (8, 8)], hook_score=3), _cand([(4, 8)], hook_score=5)])
    assert [c.candidate_id for c in res.accepted] == ["e007-c2"]
    assert _reasons(res) == {"e007-c1": "duplicate_of:e007-c2"}


def test_dedupe_overlap_over_half_of_shorter():
    # c1 5-35 (30s), c2 15-45 (30s): overlap 20s > 15s -> dup; tie keeps LLM order.
    res = _dedupe([_cand([(2, 7)]), _cand([(4, 9)])])
    assert [c.candidate_id for c in res.accepted] == ["e007-c1"]
    assert _reasons(res) == {"e007-c2": "duplicate_of:e007-c1"}


def test_dedupe_overlap_exactly_half_is_kept():
    # c1 0-20 (20s), c2 10-40 (30s): overlap 10s == 50% of 20s -> not a dup.
    res = _dedupe([_cand([(1, 4)]), _cand([(3, 8)])])
    assert len(res.accepted) == 2 and not res.rejected


def test_dedupe_counts_multi_segment_overlap():
    # c1 5-15 + 30-45 (25s); c2 10-40 (30s): overlap 5 + 10 = 15s > 12.5s.
    res = _dedupe([_cand([(2, 3), (7, 9)], cliff_score=5), _cand([(3, 8)])])
    assert [c.candidate_id for c in res.accepted] == ["e007-c1"]
    assert _reasons(res) == {"e007-c2": "duplicate_of:e007-c1"}


def test_dedupe_disjoint_kept_and_order_preserved():
    # c1 0-20, c2 35-60, c3 25-45 (overlaps c2 by 10s == 50% of 20s) -> all kept.
    res = _dedupe([_cand([(1, 4)]), _cand([(8, 12)]), _cand([(6, 9)], hook_score=5)])
    assert [c.candidate_id for c in res.accepted] == ["e007-c1", "e007-c2", "e007-c3"]


def test_dedupe_compares_against_kept_only():
    # best c3 (5-35) knocks out c1 (5-30); c2 (35-60) is disjoint from c3 -> kept,
    # even though c1 is gone.
    res = _dedupe([_cand([(2, 6)], hook_score=3), _cand([(8, 12)], hook_score=3),
                   _cand([(2, 7)], hook_score=5)])
    assert [c.candidate_id for c in res.accepted] == ["e007-c2", "e007-c3"]
    assert _reasons(res) == {"e007-c1": "duplicate_of:e007-c3"}
