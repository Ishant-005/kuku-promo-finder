"""Pydantic data model for parsed screenplays."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class BeatKind(str, Enum):
    CAMERA = "CAMERA"
    ACTION = "ACTION"
    DIALOGUE = "DIALOGUE"
    OTHER = "OTHER"


class Beat(BaseModel):
    id: str                      # "e001-b017": 1-based index in file order
    scene_no: int
    kind: BeatKind
    speaker: str | None = None   # DIALOGUE only, parentheticals stripped
    delivery: str | None = None  # last parenthetical of a dialogue tag, e.g. "Gloating, Smug"
    start_s: float
    end_s: float
    text: str

    @model_validator(mode="after")
    def _check_times(self) -> "Beat":
        if self.end_s <= self.start_s:
            raise ValueError(f"{self.id}: end_s {self.end_s} <= start_s {self.start_s}")
        return self


class Episode(BaseModel):
    number: int
    title_guess: str | None = None   # show title guessed from the parent folder name
    logline: str = ""
    continuity_notes: str = ""
    duration_s: float = 0.0          # max beat end_s
    beats: list[Beat]
    source: str = ""                 # filename
    warnings: list[str] = []         # skipped blocks / ordering anomalies (no script text)


# ---------------------------------------------------------------------------
# Extraction (LLM proposes by beat id; code resolves and validates)
# ---------------------------------------------------------------------------
HookType = Literal["slap", "reveal", "shocking_line", "confrontation", "threat",
                   "humiliation", "other"]
_HOOK_TYPES = set(HookType.__args__)


class Segment(BaseModel):
    start_id: str
    end_id: str


class LLMCandidate(BaseModel):
    """One candidate exactly as the LLM returns it (see prompts/extract.md)."""
    segments: list[Segment]
    hook: str = ""
    cliffhanger: str = ""
    hook_score: int = Field(ge=1, le=5)
    cliff_score: int = Field(ge=1, le=5)
    standalone_score: int = Field(ge=1, le=5)
    standalone_reason: str = ""
    why_it_works: str = ""
    hook_type: HookType = "other"

    @field_validator("hook_type", mode="before")
    @classmethod
    def _coerce_hook_type(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in _HOOK_TYPES else "other"


class ResolvedSegment(BaseModel):
    start_id: str
    end_id: str
    start_s: float
    end_s: float


class Candidate(BaseModel):
    candidate_id: str                      # "e001-c1": episode + 1-based index in LLM order
    segments: list[Segment] = []
    hook: str = ""
    cliffhanger: str = ""
    hook_score: int | None = None
    cliff_score: int | None = None
    standalone_score: int | None = None
    standalone_reason: str = ""
    why_it_works: str = ""
    hook_type: str = "other"
    segments_resolved: list[ResolvedSegment] = []
    duration_s: float | None = None
    reject_reason: str | None = None       # None = accepted


class ExtractionResult(BaseModel):
    episode: int
    source: str = ""
    accepted: list[Candidate] = []
    rejected: list[Candidate] = []
    episode_note: str = ""
    model: str | None = None
    prompt_hash: str | None = None
