from pathlib import Path

import pytest

from src.models import BeatKind
from src.parse import parse_episode, parse_show, parse_timestamp

ROOT = Path(__file__).resolve().parent.parent
EP1_MATCHES = sorted(ROOT.glob("data/*/episode-001*.txt"))
needs_data = pytest.mark.skipif(not EP1_MATCHES, reason="confidential data/ not present")


@pytest.fixture(scope="module")
def ep1():
    return parse_episode(EP1_MATCHES[0])


# ---- episode-001 (real data) ----

@needs_data
def test_beats_non_empty(ep1):
    assert ep1.number == 1
    assert len(ep1.beats) > 0


@needs_data
def test_every_start_before_end(ep1):
    assert all(b.start_s < b.end_s for b in ep1.beats)


@needs_data
def test_starts_non_decreasing_within_scene(ep1):
    for prev, cur in zip(ep1.beats, ep1.beats[1:]):
        if prev.scene_no == cur.scene_no:
            assert cur.start_s >= prev.start_s, (prev.id, cur.id)


@needs_data
def test_first_beat_starts_at_zero(ep1):
    assert ep1.beats[0].start_s == 0.0


@needs_data
def test_some_dialogue_has_speaker(ep1):
    assert any(b.kind is BeatKind.DIALOGUE and b.speaker for b in ep1.beats)


@needs_data
def test_ids_and_header(ep1):
    assert ep1.beats[0].id == "e001-b001"
    assert len({b.id for b in ep1.beats}) == len(ep1.beats)
    assert ep1.logline and ep1.continuity_notes
    assert ep1.duration_s == max(b.end_s for b in ep1.beats)
    assert ep1.warnings == []


# ---- synthetic fixture: format variants (no confidential data) ----

SYNTHETIC = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A test logline: with a colon.
CONTINUITY NOTES: None.

--- SCRIPT START ---

SCENE 01
SCENE HEADING: INT. ROOM – DAY
SCENE START: 00:00.00
SCENE END: 00:20.00

ACTIONS AND DIALOGUE:

**CAMERA:**
Start: (00:00.00) | End: (00:03.00)
Wide shot of a room.

**BOB (O.S.) (Angry):**
Start: (00:01.00) | End: (00:02.50)
"Get out."

**SOUND FX:**
Start: (00:02.00) | End: (00:04.00)
A door slams.

**ACTION:**
Start: (00:05.00) | End: (00:04.00)
Bad timing, should be skipped.

**ACTION:**
Start: (00:03.00) | End: (00:06.00)
Earlier start than previous kept beat? No - equal order check.

SCENE 02
Scene heading: EXT. STREET – NIGHT
Scene start: 00:20.00
Scene end: 01:05.00

**ALICE (Whispering):**
Start: (01:00.00) | End: (01:04.50)
"Who's there?"

--- SCRIPT END ---
"""


@pytest.fixture
def synth(tmp_path):
    show = tmp_path / "Some Show-scripts (2)"
    show.mkdir()
    p = show / "episode-007-Episode 999.txt"
    p.write_text(SYNTHETIC, encoding="utf-8")
    return parse_episode(p)


def test_synthetic_header_and_title(synth):
    assert synth.number == 7
    assert synth.logline == "A test logline: with a colon."
    assert synth.title_guess == "Some Show"


def test_synthetic_kinds_and_speakers(synth):
    kinds = [b.kind for b in synth.beats]
    assert kinds == [BeatKind.CAMERA, BeatKind.DIALOGUE, BeatKind.OTHER,
                     BeatKind.ACTION, BeatKind.DIALOGUE]
    bob = synth.beats[1]
    assert (bob.speaker, bob.delivery, bob.text) == ("BOB", "Angry", "Get out.")
    assert synth.beats[4].speaker == "ALICE" and synth.beats[4].text == "Who's there?"


def test_synthetic_scenes_and_uppercase_labels(synth):
    assert [b.scene_no for b in synth.beats] == [1, 1, 1, 1, 2]
    assert synth.duration_s == 64.5


def test_synthetic_skips_bad_timing_with_warning(synth):
    assert len(synth.beats) == 5
    assert [b.id for b in synth.beats] == [f"e007-b00{i}" for i in range(1, 6)]
    assert len(synth.warnings) == 1 and "end <= start" in synth.warnings[0]


def test_backwards_start_is_warned_not_dropped(tmp_path):
    text = SYNTHETIC.replace("Start: (00:03.00) | End: (00:06.00)",
                             "Start: (00:01.50) | End: (00:06.00)")
    p = tmp_path / "episode-007.txt"
    p.write_text(text, encoding="utf-8")
    ep = parse_episode(p)
    assert len(ep.beats) == 5
    assert any("backwards" in w for w in ep.warnings)


def test_parse_show_sorts_by_number(tmp_path):
    for n in (10, 2):
        (tmp_path / f"episode-{n:03d}.txt").write_text(
            SYNTHETIC.replace("Episode 7", f"Episode {n}"), encoding="utf-8")
    assert [e.number for e in parse_show(tmp_path)] == [2, 10]


def test_parse_timestamp():
    assert parse_timestamp("02:18.00") == 138.0
    assert parse_timestamp("1:00:01.5") == 3601.5
    with pytest.raises(ValueError):
        parse_timestamp("abc")
