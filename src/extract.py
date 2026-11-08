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
TEXT_MAX = 160
MIN_S, MAX_S = 20.0, 60.0
HOOK_KINDS = {BeatKind.ACTION, BeatKind.DIALOGUE}

# Reject reasons (stable strings: run.py summarises by these)
R_MALFORMED = "malformed"
R_EMPTY = "empty_segments"
R_UNKNOWN_ID = "unknown_id"
R_BAD_SEGMENT = "bad_segment"
R_ORDER = "out_of_order_or_overlap"
R_FIRST_BEAT = "bad_first_beat"
R_SHORT = "too_short"
R_LONG = "too_long"


class ExtractionError(Exception):
    """The LLM response could not be parsed at all."""


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _trim(text: str, n: int = TEXT_MAX) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def build_beat_table(episode: Episode) -> str:
    """One line per beat: id | kind | speaker | start | end | text."""
    lines = ["id | kind | speaker | start | end | text"]
    for b in episode.beats:
        lines.append(
            f"{b.id} | {b.kind.value} | {b.speaker or '-'} | "
            f"{b.start_s:.2f} | {b.end_s:.2f} | {_trim(b.text)}"
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
    if first.kind not in HOOK_KINDS:
        return reject(f"{R_FIRST_BEAT}: {first.kind.value}")

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

    out.segments_resolved = resolved
    out.duration_s = round(sum(r.end_s - r.start_s for r in resolved), 2)
    if out.duration_s < MIN_S:
        return reject(f"{R_SHORT}: {out.duration_s:.1f}s")
    if out.duration_s > MAX_S:
        return reject(f"{R_LONG}: {out.duration_s:.1f}s")
    return out


def validate_response(raw: str, episode: Episode) -> ExtractionResult:
    items, note = parse_response(raw)
    result = ExtractionResult(episode=episode.number, source=episode.source, episode_note=note)
    for n, item in enumerate(items, 1):
        cid = f"e{episode.number:03d}-c{n}"
        if isinstance(item, str):
            result.rejected.append(Candidate(candidate_id=cid, reject_reason=item))
            continue
        cand = validate_candidate(item, episode, cid)
        (result.rejected if cand.reject_reason else result.accepted).append(cand)
    return result


def extract_episode(episode: Episode, *, max_tokens: int = 8000) -> ExtractionResult:
    """One LLM call for the episode, then code-side validation."""
    system, user = load_prompt(), build_user_message(episode)
    raw = llm.complete(system, user, json_mode=True, max_tokens=max_tokens)
    result = validate_response(raw, episode)
    result.model = llm.last_model  # actual responder (may be the fallback)
    result.prompt_hash = llm._prompt_hash(system, user)
    return result


def write_result(result: ExtractionResult, out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else config.ROOT / "outputs"
    path = out / "episodes" / f"ep{result.episode:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path
