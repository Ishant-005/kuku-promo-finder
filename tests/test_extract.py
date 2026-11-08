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
        _cand([(1, 6)]),                                    # c6 starts on CAMERA
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
    assert acc == {"e007-c1", "e007-c2", "e007-c9", "e007-c10", "e007-c14", "e007-c15"}
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
    ("e007-c6", extract.R_FIRST_BEAT),
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
    assert len(res.accepted) == 6


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
    assert table[0] == "id | kind | speaker | start | end | text"
    assert len(table) == 1 + len(ep.beats)
    assert table[1] == "e007-b001 | CAMERA | - | 0.00 | 5.00 | line 1"
    row = table[2].split(" | ")
    assert row[:5] == ["e007-b002", "DIALOGUE", "ANNA", "5.00", "10.00"]
    assert len(row[5]) == extract.TEXT_MAX and row[5].endswith("…")
    assert row[5].startswith("a/b x")


def test_extract_episode_uses_prompt_file_and_writes(monkeypatch, tmp_path):
    seen = {}

    def fake_complete(system, user, **kw):
        seen.update(system=system, user=user, kw=kw)
        extract.llm.last_model = "fake-fallback"
        return "```json\n" + json.dumps(FAKE) + "\n```"

    monkeypatch.setattr(extract.llm, "complete", fake_complete)
    ep = _ep()
    res = extract.extract_episode(ep)
    assert seen["system"] == extract.load_prompt()
    assert "LOGLINE: A test." in seen["user"] and "e007-b012 |" in seen["user"]
    assert seen["kw"]["json_mode"] is True
    assert res.model == "fake-fallback" and len(res.prompt_hash) == 12

    path = extract.write_result(res, tmp_path)
    assert path == tmp_path / "episodes" / "ep007.json"
    data = json.loads(path.read_text())
    assert data["episode"] == 7 and len(data["accepted"]) == 6
    assert {"accepted", "rejected", "episode_note", "model"} <= data.keys()
