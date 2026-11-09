"""Per-episode promo extraction.

Code builds the beat table, calls the LLM once per episode, then resolves the
LLM's beat-id proposals to timestamps and rejects anything invalid. The LLM
never supplies timestamps.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from . import config, llm
from .models import (
    BeatKind, Candidate, Episode, ExtractionResult, LLMCandidate, ResolvedSegment,
)

PROMPT_PATH = config.ROOT / "prompts" / "extract.md"
ID_RETRY_PROMPT_PATH = config.ROOT / "prompts" / "extract_id_retry.md"
TEXT_MAX = 160
MIN_S, MAX_S = 20.0, 60.0
# Only OTHER (SOUND FX in the source format) may not open a clip. CAMERA beats
# are allowed: in these scripts they often carry the physical event.
BANNED_FIRST_KINDS = {BeatKind.OTHER}
# Only spoken lines may not be cut by a segment boundary. CAMERA, ACTION and
# SOUND FX beats may run across a segment start or end.
NO_SPLIT_KINDS = {BeatKind.DIALOGUE}
# A dialogue beat may stick out past a boundary by up to this much (timing slop
# in the source timestamps; 0.3 s is below a syllable).
MID_BEAT_TOL_S = 0.3
EPS = 1e-6

# Reject reasons (stable strings: run.py summarises by these)
R_MALFORMED = "malformed"
R_EMPTY = "empty_segments"
R_UNKNOWN_ID = "unknown_id"
R_BAD_SEGMENT = "bad_segment"
R_ORDER = "out_of_order_or_overlap"
R_FIRST_BEAT = "bad_first_beat"
R_MID_BEAT = "cuts_mid_beat"
R_SHORT = "too_short"
R_LONG = "too_long"
R_DUPLICATE = "duplicate_of"

# Dedupe: same weights as the ranking score, so "higher-scoring" means the same
# thing everywhere. Ties keep the candidate the LLM listed first.
SCORE_W = (0.40, 0.35, 0.25)  # hook, cliff, standalone
DUP_OVERLAP_FRAC = 0.5


class ExtractionError(Exception):
    """The LLM response could not be parsed at all."""


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def load_id_retry_note() -> str:
    return ID_RETRY_PROMPT_PATH.read_text(encoding="utf-8").strip()


def needs_id_retry(result: ExtractionResult) -> bool:
    """Zero accepted candidates and at least one unknown_id rejection."""
    return not result.accepted and any(
        (c.reject_reason or "").startswith(R_UNKNOWN_ID) for c in result.rejected
    )


def _trim(text: str, n: int = TEXT_MAX) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def build_beat_table(episode: Episode) -> str:
    """One line per beat: id | kind | speaker | start | end | cum_start | cum_end | text.

    Times are seconds, one decimal. cum_* = seconds elapsed since the first
    beat's start, so a clip's span = cum_end(last beat) - cum_start(first beat)."""
    lines = ["id | kind | speaker | start | end | cum_start | cum_end | text"]
    t0 = min((b.start_s for b in episode.beats), default=0.0)
    for b in episode.beats:
        lines.append(
            f"{b.id} | {b.kind.value} | {b.speaker or '-'} | "
            f"{b.start_s:.1f} | {b.end_s:.1f} | {b.start_s - t0:.1f} | {b.end_s - t0:.1f} | "
            f"{_trim(b.text)}"
        )
    return "\n".join(lines)


def build_user_message(episode: Episode) -> str:
    return (
        f"EPISODE {episode.number}\n"
        f"LOGLINE: {episode.logline or '(none)'}\n\n"
        f"BEATS:\n{build_beat_table(episode)}\n"
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def _load_json(raw: str) -> dict:
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i == -1 or j <= i:
            raise ExtractionError("no JSON object in response") from None
        try:
            data = json.loads(text[i : j + 1])
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"invalid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ExtractionError(f"expected a JSON object, got {type(data).__name__}")
    return data


def parse_response(raw: str) -> tuple[list[LLMCandidate | str], str]:
    """Return (items, episode_note). Each item is an LLMCandidate, or an error
    string for a candidate that failed schema validation (kept, later rejected)."""
    data = _load_json(raw)
    cands = data.get("candidates", [])
    if not isinstance(cands, list):
        raise ExtractionError("'candidates' is not a list")
    items: list[LLMCandidate | str] = []
    for c in cands:
        try:
            items.append(LLMCandidate.model_validate(c))
        except ValidationError as exc:
            # Field locations only: never echo input values (could be script text).
            locs = sorted({".".join(map(str, e["loc"])) or "<root>" for e in exc.errors()})
            items.append(f"{R_MALFORMED}: bad fields {', '.join(locs)}")
    return items, str(data.get("episode_note", "") or "")


def validate_candidate(cand: LLMCandidate, episode: Episode, candidate_id: str) -> Candidate:
    """Resolve beat ids to timestamps; set reject_reason on the first failed rule."""
    out = Candidate(candidate_id=candidate_id, **cand.model_dump())

    def reject(reason: str) -> Candidate:
        out.reject_reason = reason
        return out

    if not cand.segments:
        return reject(R_EMPTY)

    index = {b.id: i for i, b in enumerate(episode.beats)}
    unknown = [sid for s in cand.segments for sid in (s.start_id, s.end_id) if sid not in index]
    if unknown:
        return reject(f"{R_UNKNOWN_ID}: {', '.join(unknown)}")

    spans = [(index[s.start_id], index[s.end_id]) for s in cand.segments]
    for k, (i, j) in enumerate(spans):
        if i > j:
            return reject(f"{R_BAD_SEGMENT}: segment {k + 1} start after end")
    for k in range(1, len(spans)):
        if spans[k][0] <= spans[k - 1][1]:
            return reject(f"{R_ORDER}: segment {k + 1} starts at or before end of segment {k}")

    first = episode.beats[spans[0][0]]
    if first.kind in BANNED_FIRST_KINDS:
        return reject(f"{R_FIRST_BEAT}: {first.kind.value} (SOUND FX)")

    resolved: list[ResolvedSegment] = []
    for k, (i, j) in enumerate(spans):
        beats = episode.beats[i : j + 1]
        start = min(b.start_s for b in beats)
        end = max(b.end_s for b in beats)
        if end <= start:  # timestamps restarting mid-span
            return reject(f"{R_BAD_SEGMENT}: segment {k + 1} has non-positive duration")
        resolved.append(ResolvedSegment(
            start_id=episode.beats[i].id, end_id=episode.beats[j].id,
            start_s=start, end_s=end,
        ))
    for k in range(1, len(resolved)):
        if resolved[k].start_s < resolved[k - 1].end_s:
            return reject(f"{R_ORDER}: segment {k + 1} overlaps segment {k} in time")

    # Duration is recorded even if the mid-beat rule rejects, so the declared vs
    # actual comparison covers every candidate that resolved to timestamps.
    out.duration_s = round(sum(r.end_s - r.start_s for r in resolved), 2)
    mid = _mid_beat_cut(resolved, episode)
    if mid:
        return reject(f"{R_MID_BEAT}: {mid}")

    out.segments_resolved = resolved
    if out.duration_s < MIN_S:
        return reject(f"{R_SHORT}: {out.duration_s:.1f}s")
    if out.duration_s > MAX_S:
        return reject(f"{R_LONG}: {out.duration_s:.1f}s")
    return out


def _mid_beat_cut(resolved: list[ResolvedSegment], episode: Episode) -> str | None:
    """Time-based check against ALL DIALOGUE beats in the episode (not just the
    id range). A dialogue beat that overlaps a segment must not stick out past
    the segment's end, or before its start, by more than MID_BEAT_TOL_S."""
    tol = MID_BEAT_TOL_S + EPS
    for k, seg in enumerate(resolved, 1):
        for b in episode.beats:
            if b.kind not in NO_SPLIT_KINDS:
                continue
            if seg.start_s - EPS <= b.start_s < seg.end_s - EPS and b.end_s > seg.end_s + tol:
                return (f"{b.id} ({b.kind.value} {b.start_s:.2f}-{b.end_s:.2f}) runs past "
                        f"segment {k} end {seg.end_s:.2f}")
            if b.start_s < seg.start_s - tol and b.end_s > seg.start_s + EPS:
                return (f"{b.id} ({b.kind.value} {b.start_s:.2f}-{b.end_s:.2f}) is cut by "
                        f"segment {k} start {seg.start_s:.2f}")
    return None


def candidate_score(c: Candidate) -> float:
    wh, wc, ws = SCORE_W
    return wh * (c.hook_score or 0) + wc * (c.cliff_score or 0) + ws * (c.standalone_score or 0)


def _overlap_s(a: Candidate, b: Candidate) -> float:
    return sum(
        max(0.0, min(x.end_s, y.end_s) - max(x.start_s, y.start_s))
        for x in a.segments_resolved for y in b.segments_resolved
    )


def _duplicates(a: Candidate, b: Candidate) -> bool:
    """Same end beat, or time overlap > 50% of the shorter clip's duration."""
    if a.segments_resolved[-1].end_id == b.segments_resolved[-1].end_id:
        return True
    shorter = min(a.duration_s or 0.0, b.duration_s or 0.0)
    return shorter > 0 and _overlap_s(a, b) > DUP_OVERLAP_FRAC * shorter + EPS


def dedupe_accepted(result: ExtractionResult) -> ExtractionResult:
    """Within one episode, keep the higher-scoring of any duplicate pair; the
    other moves to rejected with reason "duplicate_of:<kept id>". In place."""
    order = sorted(range(len(result.accepted)),
                   key=lambda i: (-candidate_score(result.accepted[i]), i))
    kept: list[Candidate] = []
    for i in order:
        c = result.accepted[i]
        dup_of = next((k for k in kept if _duplicates(c, k)), None)
        if dup_of is None:
            kept.append(c)
        else:
            c.reject_reason = f"{R_DUPLICATE}:{dup_of.candidate_id}"
            result.rejected.append(c)
    kept_ids = {c.candidate_id for c in kept}
    result.accepted = [c for c in result.accepted if c.candidate_id in kept_ids]
    return result


def validate_response(raw: str, episode: Episode) -> ExtractionResult:
    items, note = parse_response(raw)
    result = ExtractionResult(episode=episode.number, source=episode.source, episode_note=note)
    for n, item in enumerate(items, 1):
        cid = f"e{episode.number:03d}-c{n}"
        if isinstance(item, str):
            result.rejected.append(Candidate(candidate_id=cid, reject_reason=item))
            continue
        cand = validate_candidate(item, episode, cid)
        if cand.declared_duration_s is not None and cand.duration_s is not None:
            cand.duration_delta_s = round(cand.declared_duration_s - cand.duration_s, 2)
        (result.rejected if cand.reject_reason else result.accepted).append(cand)
    return result


def extract_episode(episode: Episode, *, max_tokens: int | None = None) -> ExtractionResult:
    """One LLM call for the episode, then code-side validation and dedupe.

    If nothing was accepted and at least one candidate used an unknown beat id,
    re-ask once with the same prompt plus the id-retry note appended to the
    system prompt; the retry's result replaces the first (id_retry=True).
    prompt_hash stays the hash of the original prompt.
    max_tokens=None uses LLM_MAX_TOKENS (default 3000)."""
    system, user = load_prompt(), build_user_message(episode)
    kw = {} if max_tokens is None else {"max_tokens": max_tokens}
    raw = llm.complete(system, user, json_mode=True, **kw)
    result = validate_response(raw, episode)
    if needs_id_retry(result):
        raw = llm.complete(f"{system}\n\n{load_id_retry_note()}", user, json_mode=True, **kw)
        result = validate_response(raw, episode)
        result.id_retry = True
    result = dedupe_accepted(result)
    result.model = llm.last_model  # actual responder (may be the fallback)
    result.response_model = llm.last_response_model
    result.prompt_hash = llm._prompt_hash(system, user)
    return result


def duration_rows(result: ExtractionResult) -> list[dict]:
    """Declared (LLM) vs actual (code) duration per candidate. No script text."""
    return [
        {"episode": result.episode, "candidate_id": c.candidate_id,
         "declared_s": c.declared_duration_s, "actual_s": c.duration_s,
         "delta_s": c.duration_delta_s, "reject_reason": c.reject_reason}
        for c in sorted(result.accepted + result.rejected, key=_cand_order)
    ]


def _cand_order(c: Candidate) -> tuple[int, str]:
    """LLM order: 'e001-c3' -> 3. Unparseable ids sort last, by id."""
    tail = c.candidate_id.rsplit("-c", 1)[-1]
    return (int(tail), "") if tail.isdigit() else (10**9, c.candidate_id)


def write_result(result: ExtractionResult, out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else config.ROOT / "outputs"
    path = out / "episodes" / f"ep{result.episode:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path
