"""Parse every screenplay under data/*/ and print one summary line per episode.

Usage: python scripts/inspect.py [DATA_DIR]   (default: data)
Prints counts only, never script text.
"""
import sys
from pathlib import Path

# This file is named inspect.py; running it puts scripts/ first on sys.path, which
# would shadow the stdlib `inspect` module (imported by pydantic). Swap it for the repo root.
_HERE = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
sys.path.insert(0, str(_HERE.parent))

from src.models import BeatKind  # noqa: E402
from src.parse import parse_show  # noqa: E402


def main() -> int:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE.parent / "data"
    shows = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not shows:
        print(f"no show folders under {data_dir}")
        return 1
    total = 0
    for show in shows:
        print(f"== {show.name}")
        for ep in parse_show(show):
            n_dlg = sum(b.kind is BeatKind.DIALOGUE for b in ep.beats)
            line = f"{ep.source:<40} beats={len(ep.beats):>4}  dialogue={n_dlg:>3}  duration_s={ep.duration_s:7.2f}"
            if ep.warnings:
                line += f"  PARSE WARNINGS: {len(ep.warnings)}"
            print(line)
            total += 1
    print(f"{total} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
