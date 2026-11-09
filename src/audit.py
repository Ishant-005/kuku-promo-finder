"""Stage 4a: cold-viewer audit of the final promos.

For each promo in <out>/promos.json (the top FINAL_N), send ONLY the beat texts
inside the clip to the LLM (prompts/audit.md): no logline, no episode number,
no notes, no extraction output. Speakers are relabelled "Speaker A/B/..." in
order of first appearance, since a cold viewer sees faces, not name tags.
Results go to <out>/audit.json and an audit table appended to <out>/ranked.md
(replaced, not duplicated, on rerun).
"""
from __future__ import annotations

import json
import re
import string
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from . import config, extract, llm
from .models import BeatKind, Episode

PROMPT_PATH = config.ROOT / "prompts" / "audit.md"
AUDIT_MAX_TOKENS = 1200       # small JSON answer; keeps requests well under TPM limits
DEFAULT_SLEEP_S = 5.0         # audit prompts are small (env AUDIT_SLEEP_S overrides)
CUT_MARKER = "[cut]"
MD_MARKER = "<!-- audit -->"


class AuditAnswer(BaseModel):
    what_is_happening: str
    who_are_they: str
    open_question: str
    followable_score: int = Field(ge=1, le=5)
    would_tap_next: bool


class AuditItem(BaseModel):
    rank: int
    candidate_id: str
    episode: int
    standalone_score: int | None = None     # from extraction
    followable_score: int | None = None     # from this audit
    delta: int | None = None                # followable - standalone
    would_tap_next: bool | None = None
    what_is_happening: str = ""
    who_are_they: str = ""
    open_question: str = ""
    n_beats: int = 0
    model: str | None = None
    response_model: str | None = None
    prompt_hash: str | None = None
    error: str | None = None


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def sleep_seconds() -> float:
    raw = config.get("AUDIT_SLEEP_S", str(DEFAULT_SLEEP_S))
    try:
        return max(0.0, float(raw))
    except ValueError:
        raise RuntimeError(f"AUDIT_SLEEP_S must be a number, got {raw!r}") from None


def _speaker_label(i: int) -> str:
    letters = string.ascii_uppercase
    return f"Speaker {letters[i]}" if i < 26 else f"Speaker {i + 1}"


def _speaker_key(speaker: str | None) -> str:
    """'ANNA (cold)' and 'anna' are the same person."""
    return " ".join(re.sub(r"\(.*?\)", " ", speaker or "?").split()).upper() or "?"


def build_clip_text(promo: dict, episode: Episode) -> tuple[str, int]:
    """Beat texts inside the clip's segments (by id range, file order), with
    anonymised speakers and [cut] between segments. -> (text, n_beats).
    Raises KeyError on an id not in the episode."""
    index = {b.id: i for i, b in enumerate(episode.beats)}
    speakers: dict[str, str] = {}
    blocks: list[str] = []
    n = 0
    for seg in promo["segments"]:
        i, j = index[seg["start_id"]], index[seg["end_id"]]
        lines = []
        for b in episode.beats[i : j + 1]:
            text = " ".join(b.text.split())
            if b.kind is BeatKind.DIALOGUE:
                key = _speaker_key(b.speaker)
                if key not in speakers:
                    speakers[key] = _speaker_label(len(speakers))
                lines.append(f"{speakers[key]}: {text}")
            elif b.kind is BeatKind.OTHER:
                lines.append(f"[sound: {text}]")
            else:
                lines.append(f"[{text}]")
            n += 1
        blocks.append("\n".join(lines))
    return f"\n{CUT_MARKER}\n".join(blocks), n


def parse_answer(raw: str) -> AuditAnswer:
    """Raises extract.ExtractionError / ValidationError on bad output."""
    return AuditAnswer.model_validate(extract._load_json(raw))


def audit_promo(promo: dict, episode: Episode | None, *, complete_fn=None) -> AuditItem:
    complete_fn = complete_fn or llm.complete
    item = AuditItem(rank=promo["rank"], candidate_id=promo["candidate_id"],
                     episode=promo["episode"], standalone_score=promo.get("standalone_score"))
    if episode is None:
        item.error = f"episode {promo['episode']} not found in --input"
        return item
    try:
        clip, item.n_beats = build_clip_text(promo, episode)
    except KeyError as exc:
        item.error = f"unknown beat id {exc}"
        return item
    system = load_prompt()
    item.prompt_hash = llm._prompt_hash(system, clip)
    try:
        raw = complete_fn(system, clip, json_mode=True, max_tokens=AUDIT_MAX_TOKENS)
        item.model, item.response_model = llm.last_model, llm.last_response_model
        ans = parse_answer(raw)
    except ValidationError as exc:
        locs = sorted({".".join(map(str, e["loc"])) or "<root>" for e in exc.errors()})
        item.error = f"malformed: bad fields {', '.join(locs)}"
        return item
    except Exception as exc:  # noqa: BLE001 - recorded per promo, audit continues
        item.error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
        return item
    item.followable_score = ans.followable_score
    item.would_tap_next = ans.would_tap_next
    item.what_is_happening = ans.what_is_happening
    item.who_are_they = ans.who_are_they
    item.open_question = ans.open_question
    if item.standalone_score is not None:
        item.delta = ans.followable_score - item.standalone_score
    return item


def _md(text) -> str:
    return " ".join(str(text).split()).replace("|", "/")


def render_audit_table(items: list[AuditItem]) -> str:
    lines = [
        MD_MARKER,
        "## Audit (cold viewer, clip beats only)",
        "",
        "| rank | extraction standalone_score | audit followable_score | delta | "
        "would_tap_next | what_is_happening |",
        "|---|---|---|---|---|---|",
    ]
    for it in items:
        def fmt(v):
            return "-" if v is None else str(v)
        delta = "-" if it.delta is None else f"{it.delta:+d}"
        what = _md(it.what_is_happening) if not it.error else f"ERROR: {_md(it.error)}"
        lines.append(f"| {it.rank} | {fmt(it.standalone_score)} | {fmt(it.followable_score)} | "
                     f"{delta} | {fmt(it.would_tap_next)} | {what} |")
    return "\n".join(lines) + "\n"


def append_to_ranked(md_path: Path, items: list[AuditItem]) -> None:
    """Replace any previous audit section, then append the new one."""
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    if MD_MARKER in text:
        text = text[: text.index(MD_MARKER)]
    text = text.rstrip("\n") + "\n\n" + render_audit_table(items)
    md_path.write_text(text, encoding="utf-8")


def audit_all(out: Path, episodes: list[Episode], *, complete_fn=None,
              sleep_s: float | None = None, sleep_fn=time.sleep, log=print) -> dict:
    promos = json.loads((out / "promos.json").read_text(encoding="utf-8"))["promos"]
    by_num = {e.number: e for e in episodes}
    sleep_s = sleep_seconds() if sleep_s is None else sleep_s
    items: list[AuditItem] = []
    for k, p in enumerate(promos):
        if k and sleep_s:
            sleep_fn(sleep_s)
        it = audit_promo(p, by_num.get(p["episode"]), complete_fn=complete_fn)
        items.append(it)
        status = (f"followable={it.followable_score} delta={it.delta:+d}"
                  if it.error is None and it.delta is not None else f"ERROR {it.error}")
        log(f"  audit #{it.rank:<2} {it.candidate_id} {status}")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "items": [it.model_dump() for it in items],
    }
    (out / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    append_to_ranked(out / "ranked.md", items)
    ok = [it for it in items if it.error is None]
    return {"audited": len(items), "ok": len(ok), "errors": len(items) - len(ok),
            "big_deltas": sum(1 for it in ok if it.delta is not None and abs(it.delta) >= 2),
            "items": items}
