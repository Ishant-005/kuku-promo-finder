"""Pydantic data model for parsed screenplays."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, model_validator


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
