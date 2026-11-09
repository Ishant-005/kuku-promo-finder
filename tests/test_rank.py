"""Stage 3 ranking: scoring math, per-episode cap, LLM order validation, outputs.
Synthetic candidates only; no network, no data/."""
import csv
import json
from collections import Counter

import pytest

from src import rank
from src.models import ResolvedSegment


def _p(ep, n, h=3, c=3, s=3, dur=30.0, start=10.0, **kw):
    segs = [ResolvedSegment(start_id="a", end_id="b", start_s=start, end_s=start + dur)]
    return rank.Promo(episode=ep, candidate_id=f"e{ep:03d}-c{n}", hook_score=h, cliff_score=c,
                      standalone_score=s, duration_s=dur, segments_resolved=segs,
                      hook=f"hook {ep}.{n}", cliffhanger=f"cliff {ep}.{n}",
                      why_it_works="w", hook_type="slap", **kw)


# --- scoring ------------------------------------------------------------------

def test_base_score_formula_and_duration_bonus():
    p = _p(1, 1, h=5, c=4, s=2, dur=30.0)
    (p,) = rank.score_all([p])
    assert p.base_score == pytest.approx(0.40 * 5 + 0.35 * 4 + 0.25 * 2 + 0.3)
    assert p.penalty == 0.0 and p.score == p.base_score and p.ep_rank == 1


@pytest.mark.parametrize("dur,bonus", [(24.9, 0.0), (25.0, 0.3), (45.0, 0.3), (45.1, 0.0),
                                       (20.0, 0.0), (60.0, 0.0)])
def test_duration_bonus_bounds_inclusive(dur, bonus):
    assert rank.duration_bonus(dur) == bonus


def test_episode_penalty_by_base_score_within_episode():
    ps = [_p(1, 1, h=1), _p(1, 2, h=5), _p(1, 3, h=3), _p(1, 4, h=2), _p(1, 5, h=4)]
    rank.score_all(ps)
    by = {p.candidate_id: p for p in ps}
    # base order: c2 (h5) > c5 (h4) > c3 (h3) > c4 (h2) > c1 (h1)
    assert [by[f"e001-c{i}"].ep_rank for i in (2, 5, 3, 4, 1)] == [1, 2, 3, 4, 5]
    assert [by[f"e001-c{i}"].penalty for i in (2, 5, 3, 4, 1)] == [0.0, 0.5, 1.0, 2.0, 2.0]
    assert by["e001-c5"].score == pytest.approx(by["e001-c5"].base_score - 0.5)


def test_sort_desc_ties_by_episode_then_id():
    ps = [_p(3, 1), _p(2, 2), _p(2, 1, h=5)]
    ranked = rank.score_all(ps)
    # e002-c1 best; e003-c1 (ep_rank 1) ties e002-c2 base but e002-c2 is ep_rank 2 (-0.5)
    assert [p.candidate_id for p in ranked] == ["e002-c1", "e003-c1", "e002-c2"]


# --- per-episode cap ---------------------------------------------------------

def test_cap_three_per_episode_and_backfill_to_n():
    # ep1 has 6 very strong candidates; 30 weaker ones from eps 2..31.
    ps = [_p(1, i, h=5, c=5, s=5) for i in range(1, 7)]
    ps += [_p(ep, 1, h=2, c=2, s=2) for ep in range(2, 32)]
    top, dropped = rank.select_top(rank.score_all(ps), n=30, cap=3)
    assert len(top) == 30
    assert Counter(p.episode for p in top)[1] == 3
    # ep1's 4th (score 5.3 - 2.0 = 3.3) outranks the 2.3s from other eps, so it is
    # hit by the cap, not by the cutoff
    assert {p.candidate_id for p in dropped} >= {"e001-c4"}
    assert all(p.cap_dropped_from_top30 for p in dropped)
    assert all(not p.cap_dropped_from_top30 for p in top)
    # filled back from the next-best: 27 of the 30 single-candidate episodes
    assert sum(1 for p in top if p.episode != 1) == 27


def test_cap_fewer_candidates_than_n():
    top, dropped = rank.select_top(rank.score_all([_p(1, 1), _p(1, 2), _p(2, 1)]), n=30)
    assert len(top) == 3 and dropped == []


# --- LLM ordering ----------------------------------------------------------------

IDS = {"e001-c1", "e002-c1", "e003-c1"}


def test_parse_ranking_orders_by_rank_field():
    raw = json.dumps({"ranking": [
        {"rank": 2, "candidate_id": "e002-c1", "reason": "b"},
        {"rank": 1, "candidate_id": "e003-c1", "reason": "a"}]})
    picks, issues = rank.parse_ranking(raw, IDS)
    assert picks == [("e003-c1", "a"), ("e002-c1", "b")] and not issues


def test_parse_ranking_drops_unknown_and_repeats():
    raw = "```json\n" + json.dumps({"ranking": [
        {"rank": 1, "candidate_id": "e009-c9"},
        {"rank": 2, "candidate_id": "e001-c1"},
        {"rank": 3, "candidate_id": "e001-c1"}]}) + "\n```"
    picks, issues = rank.parse_ranking(raw, IDS)
    assert [c for c, _ in picks] == ["e001-c1"]
    assert issues == {rank.I_UNKNOWN: 1, rank.I_REPEAT: 1}


@pytest.mark.parametrize("raw", ["nope", '{"x": 1}', "{]", '{"a": [], "b": []}'])
def test_parse_ranking_unparseable(raw):
    picks, issues = rank.parse_ranking(raw, IDS)
    assert picks == [] and issues == {rank.I_UNPARSEABLE: 1}


def test_parse_ranking_accepts_bare_list():
    picks, _ = rank.parse_ranking(json.dumps([{"candidate_id": "e002-c1"}]), IDS)
    assert picks == [("e002-c1", "")]


def test_final_order_llm_first_then_backfill():
    top = rank.score_all([_p(1, 1, h=5), _p(2, 1, h=4), _p(3, 1, h=3)])
    issues = Counter()
    final = rank.final_order(top, [("e003-c1", "why")], issues, n=3)
    assert [p.candidate_id for p in final] == ["e003-c1", "e001-c1", "e002-c1"]
    assert [p.rank for p in final] == [1, 2, 3]
    assert [p.source for p in final] == ["llm", "backfill", "backfill"]
    assert final[0].rank_reason == "why" and issues[rank.I_BACKFILL] == 2


def test_final_order_truncates_extra_llm_picks():
    top = rank.score_all([_p(e, 1) for e in range(1, 5)])
    picks = [(p.candidate_id, "") for p in reversed(top)]
    final = rank.final_order(top, picks, Counter(), n=2)
    assert [p.candidate_id for p in final] == [picks[0][0], picks[1][0]]


# --- formatting and end to end -----------------------------------------------------

def test_format_segments_mmss():
    p = _p(1, 1, dur=20.0, start=58.25)
    p.segments_resolved.append(ResolvedSegment(start_id="c", end_id="d", start_s=90.0,
                                               end_s=95.5))
    assert rank.format_segments(p) == "00:58.2–01:18.2 + 01:30.0–01:35.5"


def _write_episode(out, ep, cands):
    d = out / "episodes"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"ep{ep:03d}.json").write_text(json.dumps({
        "episode": ep, "accepted": [c.model_dump(exclude={"episode"}) for c in cands],
        "rejected": []}))


def test_rank_all_end_to_end(tmp_path, monkeypatch):
    _write_episode(tmp_path, 1, [_p(1, i, h=5) for i in range(1, 6)])
    for ep in range(2, 20):
        _write_episode(tmp_path, ep, [_p(ep, 1)])
    seen = {}

    def fake_complete(system, user, **kw):
        seen.update(system=system, user=user, kw=kw)
        rank.llm.last_model, rank.llm.last_response_model = "m", "m-served"
        return json.dumps({"ranking": [{"rank": 1, "candidate_id": "e019-c1", "reason": "r1"},
                                       {"rank": 2, "candidate_id": "bogus"}]})

    res = rank.rank_all(tmp_path, complete_fn=fake_complete)
    assert seen["system"] == rank.load_prompt() and seen["kw"]["json_mode"] is True
    cards = seen["user"].splitlines()
    assert cards[0].startswith("candidate_id | episode") and len(cards) == 1 + 21
    assert sum(1 for c in cards[1:] if c.startswith("e001-")) == 3     # cap applied
    assert res["loaded"] == 23 and res["top"] == 21 and res["cap_dropped"] == 2
    assert res["final"] == 15 and res["issues"] == {"unknown_id": 1, "backfilled": 14}

    data = json.loads((tmp_path / "promos.json").read_text())
    assert data["promos"][0]["candidate_id"] == "e019-c1" and data["promos"][0]["rank"] == 1
    assert data["response_model"] == "m-served" and len(data["top30"]) == 21
    rows = list(csv.DictReader((tmp_path / "promos.csv").open()))
    assert list(rows[0].keys()) == rank.CSV_COLUMNS and len(rows) == 15
    assert rows[0]["episode"] == "19" and rows[0]["duration"] == "30.0"
    md = (tmp_path / "ranked.md").read_text()
    assert "| rank | episode | segments |" in md and md.count("\n| ") == 16  # header + 15
