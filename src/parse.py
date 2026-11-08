"""Deterministic screenplay parser: one .txt per episode -> Episode.

Line-based state machine. Label matching is case-insensitive because real files
vary (e.g. "Scene start:" vs "SCENE START:"). Anything unparseable is skipped and
recorded in Episode.warnings (line numbers / tag names only, never script text).
"""
from __future__ import annotations

import re
from pathlib import Path

from .models import Beat, BeatKind, Episode

_HEADER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z ]*?)\s*:\s*(.*?)\s*$")
_SCRIPT_START_RE = re.compile(r"^\s*-{2,}\s*SCRIPT\s+START\s*-{2,}", re.I)
_SCRIPT_END_RE = re.compile(r"^\s*-{2,}\s*SCRIPT\s+END\s*-{2,}", re.I)
_SCENE_RE = re.compile(r"^\s*SCENE\s+(\d+)\s*$", re.I)
_SCENE_META_RE = re.compile(
    r"^\s*(scene\s+heading|scene\s+start|scene\s+end|actions\s+and\s+dialogue)\s*:", re.I
)
_TAG_RE = re.compile(r"^\s*\*\*(.+?):\*\*\s*$")
_TIMING_RE = re.compile(
    r"^\s*start\s*:\s*\(?\s*([\d:.]+)\s*\)?\s*\|\s*end\s*:\s*\(?\s*([\d:.]+)\s*\)?\s*$", re.I
)
_TS_RE = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")
_PAREN_RE = re.compile(r"\(([^)]*)\)")
_QUOTES = "\"'“”‘’"


def parse_timestamp(s: str) -> float:
    """'MM:SS.cc' (or 'H:MM:SS.cc') -> seconds."""
    m = _TS_RE.match(s.strip())
    if not m:
        raise ValueError(f"bad timestamp: {s!r}")
    h, mm, ss = m.groups()
    return (int(h) if h else 0) * 3600 + int(mm) * 60 + float(ss)


def _classify(tag: str) -> tuple[BeatKind, str | None, str | None]:
    """Tag text (between ** and :**) -> (kind, speaker, delivery)."""
    tag = tag.strip()
    if "(" in tag:
        speaker = tag.split("(", 1)[0].strip()
        parens = [p.strip() for p in _PAREN_RE.findall(tag) if p.strip()]
        return BeatKind.DIALOGUE, speaker or None, (parens[-1] if parens else None)
    upper = tag.upper()
    if upper == "CAMERA":
        return BeatKind.CAMERA, None, None
    if upper == "ACTION":
        return BeatKind.ACTION, None, None
    return BeatKind.OTHER, None, None


def _episode_number(header_value: str | None, path: Path) -> int:
    for src in (header_value or "", path.stem):
        m = re.search(r"(\d+)", src)
        if m:
            return int(m.group(1))
    raise ValueError(f"{path.name}: no episode number in header or filename")


def _title_guess(path: Path) -> str | None:
    name = re.sub(r"[-_ ]*scripts?\b.*$", "", path.parent.name, flags=re.I).strip()
    return name or None


def parse_episode(path: str | Path) -> Episode:
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    header: dict[str, str] = {}
    warnings: list[str] = []
    pending: list[dict] = []   # raw beat blocks before validation
    scene_no = 0
    cur: dict | None = None
    in_body = False

    def close() -> None:
        nonlocal cur
        if cur is not None:
            pending.append(cur)
            cur = None

    for lineno, line in enumerate(lines, 1):
        if not in_body:
            if _SCRIPT_START_RE.match(line):
                in_body = True
                continue
            if _SCENE_RE.match(line):   # file without a SCRIPT START marker
                in_body = True
            else:
                m = _HEADER_RE.match(line)
                if m:
                    header[m.group(1).upper()] = m.group(2)
                continue

        if _SCRIPT_END_RE.match(line):
            close()
            break
        if not line.strip():
            continue
        m = _SCENE_RE.match(line)
        if m:
            close()
            scene_no = int(m.group(1))
            continue
        if _SCENE_META_RE.match(line):
            close()
            continue
        m = _TAG_RE.match(line)
        if m:
            close()
            kind, speaker, delivery = _classify(m.group(1))
            cur = {"line": lineno, "tag": kind.value, "scene_no": scene_no, "kind": kind,
                   "speaker": speaker, "delivery": delivery, "times": None, "text": []}
            continue
        m = _TIMING_RE.match(line)
        if m and cur is not None and cur["times"] is None and not cur["text"]:
            try:
                cur["times"] = (parse_timestamp(m.group(1)), parse_timestamp(m.group(2)))
            except ValueError:
                warnings.append(f"line {lineno}: unparseable timestamp")
            continue
        if cur is not None:
            cur["text"].append(line.strip())
        else:
            warnings.append(f"line {lineno}: unrecognised line outside any beat")
    close()

    number = _episode_number(header.get("EPISODE NUMBER"), path)
    beats: list[Beat] = []
    last_start: dict[int, float] = {}
    for blk in pending:
        where = f"line {blk['line']} ({blk['tag']})"
        if blk["times"] is None:
            warnings.append(f"{where}: skipped, no timing")
            continue
        start, end = blk["times"]
        if end <= start:
            warnings.append(f"{where}: skipped, end <= start")
            continue
        text = " ".join(blk["text"]).strip()
        if blk["kind"] is BeatKind.DIALOGUE:
            text = text.strip(_QUOTES).strip()
        if not text:
            warnings.append(f"{where}: skipped, empty text")
            continue
        sc = blk["scene_no"]
        if sc in last_start and start < last_start[sc]:
            warnings.append(f"{where}: start goes backwards within scene {sc} (kept in file order)")
        last_start[sc] = start
        beats.append(Beat(
            id=f"e{number:03d}-b{len(beats) + 1:03d}",
            scene_no=sc, kind=blk["kind"], speaker=blk["speaker"], delivery=blk["delivery"],
            start_s=start, end_s=end, text=text,
        ))

    if not in_body:
        warnings.append("no SCRIPT START marker or SCENE line found")

    return Episode(
        number=number,
        title_guess=_title_guess(path),
        logline=header.get("EPISODE LOGLINE", ""),
        continuity_notes=header.get("CONTINUITY NOTES", ""),
        duration_s=max((b.end_s for b in beats), default=0.0),
        beats=beats,
        source=path.name,
        warnings=warnings,
    )


def parse_show(directory: str | Path) -> list[Episode]:
    """Parse every .txt in a folder, sorted by episode number."""
    files = sorted(p for p in Path(directory).glob("*.txt") if p.is_file())
    return sorted((parse_episode(p) for p in files), key=lambda e: e.number)
