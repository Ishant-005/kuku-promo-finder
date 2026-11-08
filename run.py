"""Promo finder CLI.

    python run.py --input "data/<show dir>" --out outputs [--episodes 1,10,25] [--force]

Stage 1: parse every screenplay in --input.
Stage 2: extract each episode (one LLM call each), writing <out>/episodes/epNNN.json.
         Existing outputs are skipped unless --force. LLM_SLEEP_S (default 4)
         seconds between LLM calls. Per-episode failures go to
         <out>/logs/errors.log and the run continues.
Prints counts and reject reasons only, never script text.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src import config, extract
from src.models import Episode
from src.parse import parse_show

ERROR_MSG_MAX = 500


def parse_episode_list(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    try:
        return {int(x) for x in spec.split(",") if x.strip()}
    except ValueError:
        raise argparse.ArgumentTypeError(f"--episodes must be comma-separated ints, got {spec!r}")


def sleep_seconds() -> float:
    raw = config.get("LLM_SLEEP_S", "4")
    try:
        return max(0.0, float(raw))
    except ValueError:
        raise RuntimeError(f"LLM_SLEEP_S must be a number, got {raw!r}") from None


def reason_key(reason: str) -> str:
    """'unknown_id: e001-b999' -> 'unknown_id'; 'duplicate_of:e001-c2' -> 'duplicate_of'."""
    return reason.split(":", 1)[0].strip()


def _log_error(out: Path, ep: Episode, exc: BaseException) -> None:
    path = out / "logs" / "errors.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    msg = " ".join(f"{type(exc).__name__}: {exc}".split())[:ERROR_MSG_MAX]
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{ts} ep{ep.number:03d} {ep.source} {msg}\n")


def _log_durations(out: Path, rows: list[dict], stats: dict) -> None:
    """Append declared-vs-actual rows to <out>/logs/durations.jsonl and stats."""
    path = out / "logs" / "durations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"ts": ts, **r}) + "\n")
    stats["durations"].extend(rows)


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.1f}"


def print_durations(rows: list[dict]) -> None:
    print("\ndeclared vs actual duration (s)")
    print(f"  {'candidate':<10} {'declared':>8} {'actual':>8} {'delta':>7}  result")
    for r in rows:
        print(f"  {r['candidate_id']:<10} {_fmt(r['declared_s']):>8} {_fmt(r['actual_s']):>8} "
              f"{_fmt(r['delta_s']):>7}  {reason_key(r['reject_reason'] or 'accepted')}")
    deltas = [abs(r["delta_s"]) for r in rows if r["delta_s"] is not None]
    missing = sum(1 for r in rows if r["declared_s"] is None)
    if deltas:
        print(f"  |delta| mean {sum(deltas) / len(deltas):.1f}s  max {max(deltas):.1f}s  "
              f"n={len(deltas)}  within 1s: {sum(d <= 1.0 for d in deltas)}")
    print(f"  declared missing: {missing}")


def run_extract(episodes: list[Episode], out: Path, *, force: bool, sleep_s: float,
                extract_fn=extract.extract_episode, sleep_fn=time.sleep) -> dict:
    stats = {"done": 0, "skipped": 0, "failed": 0, "accepted": 0, "rejected": Counter(),
             "durations": []}
    called = False
    for ep in episodes:
        target = out / "episodes" / f"ep{ep.number:03d}.json"
        if target.exists() and not force:
            stats["skipped"] += 1
            print(f"ep{ep.number:03d} skip (exists)", flush=True)
            continue
        if called and sleep_s:
            sleep_fn(sleep_s)
        called = True
        t0 = time.monotonic()
        try:
            res = extract_fn(ep)
            extract.write_result(res, out)
        except Exception as exc:  # noqa: BLE001 - logged, run continues
            stats["failed"] += 1
            _log_error(out, ep, exc)
            print(f"ep{ep.number:03d} FAILED {type(exc).__name__} "
                  f"({time.monotonic() - t0:.1f}s, see logs/errors.log)", flush=True)
            continue
        _log_durations(out, extract.duration_rows(res), stats)
        stats["done"] += 1
        stats["accepted"] += len(res.accepted)
        stats["rejected"].update(reason_key(c.reject_reason or "") for c in res.rejected)
        print(f"ep{ep.number:03d} ok accepted={len(res.accepted)} rejected={len(res.rejected)} "
              f"model={res.model} served={res.response_model} "
              f"({time.monotonic() - t0:.1f}s)", flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, type=Path, help="folder of episode .txt files")
    ap.add_argument("--out", default=config.ROOT / "outputs", type=Path)
    ap.add_argument("--episodes", type=parse_episode_list, default=None,
                    help='subset, e.g. "1,10,25"')
    ap.add_argument("--force", action="store_true", help="re-extract existing outputs")
    args = ap.parse_args(argv)

    if not args.input.is_dir():
        print(f"--input is not a directory: {args.input}", file=sys.stderr)
        return 2

    # Stage 1
    episodes = parse_show(args.input)
    print(f"stage 1: parsed {len(episodes)} episodes from {args.input.name}")
    if args.episodes is not None:
        missing = sorted(args.episodes - {e.number for e in episodes})
        episodes = [e for e in episodes if e.number in args.episodes]
        if missing:
            print(f"warning: episodes not found: {missing}")

    # Stage 2
    sleep_s = sleep_seconds()
    print(f"stage 2: extracting {len(episodes)} episodes (sleep {sleep_s:g}s between calls)")
    stats = run_extract(episodes, args.out, force=args.force, sleep_s=sleep_s)

    if stats["durations"]:
        print_durations(stats["durations"])

    print("\nsummary")
    print(f"  episodes done:    {stats['done']}")
    print(f"  episodes skipped: {stats['skipped']}")
    print(f"  episodes failed:  {stats['failed']}")
    print(f"  accepted:         {stats['accepted']}")
    print(f"  rejected:         {sum(stats['rejected'].values())}")
    for reason, n in stats["rejected"].most_common():
        print(f"    {reason:<26} {n}")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
