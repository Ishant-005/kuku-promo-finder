"""Stage 3: cross-episode ranking.

Code does the math: base score, duration bonus, per-episode diversity penalty,
a hard cap of PER_EP_CAP per episode while filling the top TOP_N, then ONE LLM
call (prompts/rank.md) that orders the final FINAL_N. The LLM only picks and
orders candidate ids; code rejects unknown / repeated ids and backfills from
the numeric order if it returns too few.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from . import config, extract, llm
from .models import Candidate

PROMPT_PATH = config.ROOT / "prompts" / "rank.md"
DUR_BONUS = 0.3
DUR_BONUS_RANGE = (25.0, 45.0)            # inclusive
EP_PENALTIES = (0.0, 0.5, 1.0)            # 1st, 2nd, 3rd best in an episode
EP_PENALTY_LATER = 2.0                    # 4th onward
TOP_N = 30
PER_EP_CAP = 3
FINAL_N = 15
CARD_FIELD_MAX = 200

CSV_COLUMNS = ["rank", "episode", "segments", "duration", "hook_type", "hook",
               "cliffhanger", "why", "h", "c", "s", "score"]

# Ranking issues (stable keys, printed as counts)
I_UNPARSEABLE = "unparseable_response"
I_UNKNOWN = "unknown_id"
I_REPEAT = "repeated_id"
I_BACKFILL = "backfilled"


class Promo(Candidate):
    episode: int
    base_score: float = 0.0
    duration_bonus: float = 0.0
    ep_rank: int = 0                       # 1 = best (by base score) in its episode
    penalty: float = 0.0
    score: float = 0.0
    cap_dropped_from_top30: bool = False
    rank: int | None = None
    source: str | None = None              # "llm" | "backfill"
    rank_reason: str = ""


# --- scoring ------------------------------------------------------------------

def load_accepted(episodes_dir: Path) -> list[Promo]:
    promos: list[Promo] = []
    for path in sorted(episodes_dir.glob("ep*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for c in data.get("accepted", []):
            promos.append(Promo(episode=data["episode"], **c))
    return promos


def duration_bonus(duration_s: float | None) -> float:
    lo, hi = DUR_BONUS_RANGE
    d = duration_s or 0.0
    return DUR_BONUS if lo - extract.EPS <= d <= hi + extract.EPS else 0.0


def _tie_key(p: Promo) -> tuple:
    return (p.episode, p.candidate_id)


def score_all(promos: list[Promo]) -> list[Promo]:
    """Set base_score, duration_bonus, ep_rank, penalty, score. Returns promos
    sorted by final score (desc), ties by episode then candidate id."""
    for p in promos:
        p.duration_bonus = duration_bonus(p.duration_s)
        p.base_score = round(extract.candidate_score(p) + p.duration_bonus, 4)
    by_ep: dict[int, list[Promo]] = {}
    for p in promos:
        by_ep.setdefault(p.episode, []).append(p)
    for group in by_ep.values():
        group.sort(key=lambda p: (-p.base_score, p.candidate_id))
        for i, p in enumerate(group):
            p.ep_rank = i + 1
            p.penalty = EP_PENALTIES[i] if i < len(EP_PENALTIES) else EP_PENALTY_LATER
            p.score = round(p.base_score - p.penalty, 4)
    return sorted(promos, key=lambda p: (-p.score, *_tie_key(p)))


def select_top(ranked: list[Promo], n: int = TOP_N, cap: int = PER_EP_CAP
               ) -> tuple[list[Promo], list[Promo]]:
    """Walk the score order; skip the (cap+1)th and later candidates of any
    episode; stop at n. Returns (top, cap_dropped). Skipped candidates that
    would otherwise have made the top n are flagged cap_dropped_from_top30."""
    top: list[Promo] = []
    dropped: list[Promo] = []
    per_ep: Counter = Counter()
    for p in ranked:
        if len(top) >= n:
            break
        if per_ep[p.episode] >= cap:
            p.cap_dropped_from_top30 = True
            dropped.append(p)
            continue
        per_ep[p.episode] += 1
        top.append(p)
    return top, dropped


# --- LLM ordering ---------------------------------------------------------------

def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _cell(text: str) -> str:
    return extract._trim(text or "-", CARD_FIELD_MAX)


def build_cards(top: list[Promo]) -> str:
    lines = ["candidate_id | episode | duration_s | hook_type | hook | cliffhanger | why"]
    for p in top:
        lines.append(f"{p.candidate_id} | {p.episode} | {p.duration_s:.1f} | {p.hook_type} | "
                     f"{_cell(p.hook)} | {_cell(p.cliffhanger)} | {_cell(p.why_it_works)}")
    return "\n".join(lines)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def _load_ranking(raw: str) -> list:
    """Return the list of ranking entries. Accepts {"ranking": [...]}, any
    single list-valued key, or a bare list. Raises ValueError otherwise."""
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        i = min((k for k in (text.find("{"), text.find("[")) if k != -1), default=-1)
        j = max(text.rfind("}"), text.rfind("]"))
        if i == -1 or j <= i:
            raise ValueError("no JSON in response") from None
        try:
            data = json.loads(text[i : j + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("ranking"), list):
            return data["ranking"]
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    raise ValueError("no ranking list in response")


def parse_ranking(raw: str, valid_ids: set[str]) -> tuple[list[tuple[str, str]], Counter]:
    """-> ([(candidate_id, reason)] in the LLM's order, issues). Entries are
    ordered by their "rank" field when every entry has an int rank, else by
    list position. Unknown and repeated ids are dropped and counted."""
    issues: Counter = Counter()
    try:
        entries = _load_ranking(raw)
    except ValueError:
        issues[I_UNPARSEABLE] += 1
        return [], issues
    entries = [e for e in entries if isinstance(e, dict)]
    if entries and all(isinstance(e.get("rank"), int) for e in entries):
        entries = sorted(entries, key=lambda e: e["rank"])  # stable for equal ranks
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for e in entries:
        cid = str(e.get("candidate_id", "")).strip()
        if cid not in valid_ids:
            issues[I_UNKNOWN] += 1
            continue
        if cid in seen:
            issues[I_REPEAT] += 1
            continue
        seen.add(cid)
        out.append((cid, str(e.get("reason", "") or "")))
    return out, issues


def final_order(top: list[Promo], picks: list[tuple[str, str]], issues: Counter,
                n: int = FINAL_N) -> list[Promo]:
    """LLM order first (capped at n), then backfill from the numeric order."""
    by_id = {p.candidate_id: p for p in top}
    final: list[Promo] = []
    for cid, reason in picks[:n]:
        p = by_id[cid]
        p.source, p.rank_reason = "llm", reason
        final.append(p)
    chosen = {p.candidate_id for p in final}
    for p in top:
        if len(final) >= min(n, len(top)):
            break
        if p.candidate_id not in chosen:
            p.source = "backfill"
            final.append(p)
            issues[I_BACKFILL] += 1
    for i, p in enumerate(final, 1):
        p.rank = i
    return final


# --- output -------------------------------------------------------------------

def _mmss(s: float) -> str:
    m, sec = divmod(s, 60)
    return f"{int(m):02d}:{sec:04.1f}"


def format_segments(p: Promo) -> str:
    return " + ".join(f"{_mmss(r.start_s)}–{_mmss(r.end_s)}" for r in p.segments_resolved)


def csv_row(p: Promo) -> dict:
    return {
        "rank": p.rank, "episode": p.episode, "segments": format_segments(p),
        "duration": f"{p.duration_s:.1f}", "hook_type": p.hook_type, "hook": p.hook,
        "cliffhanger": p.cliffhanger, "why": p.why_it_works, "h": p.hook_score,
        "c": p.cliff_score, "s": p.standalone_score, "score": f"{p.score:.2f}",
    }


def _md(text) -> str:
    return " ".join(str(text).split()).replace("|", "/")


def render_markdown(final: list[Promo], meta: dict) -> str:
    lines = [
        "# Ranked promos",
        "",
        f"- generated: {meta['generated_at']}",
        f"- ranking model: {meta['model']} (served: {meta['response_model']})",
        f"- accepted candidates loaded: {meta['loaded']}; top {TOP_N} sent to LLM "
        f"(max {PER_EP_CAP} per episode; {meta['cap_dropped']} skipped by the cap)",
        f"- ranking issues: {meta['issues'] or 'none'}",
        "",
        "| " + " | ".join(CSV_COLUMNS + ["source", "rank_reason"]) + " |",
        "|" + "---|" * (len(CSV_COLUMNS) + 2),
    ]
    for p in final:
        row = csv_row(p)
        cells = [_md(row[k]) for k in CSV_COLUMNS] + [p.source or "", _md(p.rank_reason)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(out: Path, final: list[Promo], top: list[Promo], dropped: list[Promo],
                  meta: dict) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "promos.json", "csv": out / "promos.csv", "md": out / "ranked.md"}
    payload = {
        **meta,
        "weights": {"hook_cliff_standalone": list(extract.SCORE_W), "duration_bonus": DUR_BONUS,
                    "duration_bonus_range": list(DUR_BONUS_RANGE),
                    "ep_penalties": list(EP_PENALTIES), "ep_penalty_later": EP_PENALTY_LATER,
                    "top_n": TOP_N, "per_ep_cap": PER_EP_CAP, "final_n": FINAL_N},
        "promos": [p.model_dump() for p in final],
        "top30": [p.model_dump() for p in top],
        "cap_dropped": [p.model_dump() for p in dropped],
    }
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for p in final:
            w.writerow(csv_row(p))
    paths["md"].write_text(render_markdown(final, meta), encoding="utf-8")
    return paths


# --- entry point ----------------------------------------------------------------

def rank_all(out: Path, *, complete_fn=None) -> dict:
    """Load <out>/episodes, score, cap, ONE LLM call, write outputs. Returns a
    summary dict (counts only, no script text)."""
    complete_fn = complete_fn or llm.complete
    promos = load_accepted(out / "episodes")
    ranked = score_all(promos)
    top, dropped = select_top(ranked)
    issues: Counter = Counter()
    picks: list[tuple[str, str]] = []
    model = response_model = phash = None
    if top:
        system, user = load_prompt(), build_cards(top)
        phash = llm._prompt_hash(system, user)
        raw = complete_fn(system, user, json_mode=True)
        model, response_model = llm.last_model, llm.last_response_model
        picks, issues = parse_ranking(raw, {p.candidate_id for p in top})
    final = final_order(top, picks, issues)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model, "response_model": response_model, "prompt_hash": phash,
        "loaded": len(promos),
        "cap_dropped": len(dropped), "issues": dict(issues),
    }
    paths = write_outputs(out, final, top, dropped, meta)
    return {**meta, "top": len(top), "final": len(final),
            "top_per_episode": Counter(p.episode for p in top), "paths": paths,
            "final_promos": final}
