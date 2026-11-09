"""Stage 4b: outputs/stats.md from the saved outputs and outputs/logs/calls.jsonl.

All numbers are computed from files; nothing is show-specific. No script text:
the only free text included is each zero-accepted episode's episode_note
(the LLM's one-line judgement, not a quote).
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import config, llm, rank

PRICE_IN_PER_M = 0.30    # USD per 1M prompt tokens
PRICE_OUT_PER_M = 2.50   # USD per 1M completion tokens
SCORE_FIELDS = ("hook_score", "cliff_score", "standalone_score")


def reason_key(reason: str) -> str:
    return (reason or "").split(":", 1)[0].strip()


def load_episodes(out: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((out / "episodes").glob("ep*.json"))]


def load_calls(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    recs = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pipeline_hashes(episodes: list[dict], promos: dict, audit: dict) -> set[str]:
    """prompt_hashes of the LLM calls behind the current outputs."""
    hs = {e.get("prompt_hash") for e in episodes}
    hs.add(promos.get("prompt_hash") or rank_prompt_hash(promos))
    hs |= {it.get("prompt_hash") for it in audit.get("items", [])}
    return {h for h in hs if h}


def rank_prompt_hash(promos: dict) -> str | None:
    """For promos.json written before prompt_hash was stored: rebuild the
    ranking prompt from the saved top30 (same order, same card format)."""
    top = promos.get("top30")
    if not top:
        return None
    cards = rank.build_cards([rank.Promo(**p) for p in top])
    return llm._prompt_hash(rank.load_prompt(), cards)


def call_totals(recs: list[dict]) -> dict:
    ok = [r for r in recs if r.get("usage") is not None]
    pin = sum((r["usage"] or {}).get("prompt_tokens") or 0 for r in ok)
    pout = sum((r["usage"] or {}).get("completion_tokens") or 0 for r in ok)
    ts = sorted(r["ts"] for r in recs if r.get("ts"))
    return {
        "calls": len(recs), "ok": len(ok), "errors": len(recs) - len(ok),
        "prompt_tokens": pin, "completion_tokens": pout,
        "cost_usd": pin / 1e6 * PRICE_IN_PER_M + pout / 1e6 * PRICE_OUT_PER_M,
        "llm_seconds": sum(r.get("latency_ms") or 0 for r in recs) / 1000,
        "first_ts": ts[0] if ts else None, "last_ts": ts[-1] if ts else None,
        "fallback_records": sum(1 for r in recs if r.get("fallback_used")),
    }


def score_distribution(accepted: list[dict]) -> dict[str, Counter]:
    return {f: Counter(c.get(f) for c in accepted if c.get(f) is not None) for f in SCORE_FIELDS}


def model_split(episodes: list[dict]) -> Counter:
    """'primary' / 'fallback' / 'other' / 'none' per episode, by the model that
    answered vs LLM_MODEL / LLM_FALLBACK_MODEL in the environment."""
    primary, fallback = config.get("LLM_MODEL"), config.get("LLM_FALLBACK_MODEL")
    out: Counter = Counter()
    for e in episodes:
        m = e.get("model")
        out["none" if m is None else "primary" if m == primary
            else "fallback" if m == fallback else "other"] += 1
    return out


def _span(first: str | None, last: str | None) -> str:
    if not first or not last:
        return "-"
    secs = (datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds()
    h, rem = divmod(int(secs), 3600)
    return f"{h}h{rem // 60:02d}m ({first[:16]} → {last[:16]} UTC)"


def build_stats(out: Path, *, n_parsed: int | None = None,
                stage_seconds: float | None = None) -> dict:
    episodes = load_episodes(out)
    promos = _read_json(out / "promos.json")
    audit = _read_json(out / "audit.json")
    calls = load_calls(out / "logs" / "calls.jsonl")
    accepted = [c for e in episodes for c in e.get("accepted", [])]
    rejected = [c for e in episodes for c in e.get("rejected", [])]
    hashes = pipeline_hashes(episodes, promos, audit)
    return {
        "n_parsed": n_parsed,
        "n_episodes": len(episodes),
        "proposed": len(accepted) + len(rejected),
        "accepted": len(accepted),
        "rejected": Counter(reason_key(c.get("reject_reason")) for c in rejected),
        "dist": score_distribution(accepted),
        "zero": [
            {"episode": e["episode"], "note": e.get("episode_note", ""),
             "rejected": Counter(reason_key(c.get("reject_reason")) for c in e.get("rejected", []))}
            for e in episodes if not e.get("accepted")
        ],
        "models": model_split(episodes),
        "model_names": Counter(e.get("model") for e in episodes),
        "primary_model": config.get("LLM_MODEL"),
        "fallback_model": config.get("LLM_FALLBACK_MODEL"),
        "pipeline": call_totals([r for r in calls if r.get("prompt_hash") in hashes]),
        "all": call_totals(calls),
        "stage_seconds": stage_seconds,
        "audit": audit.get("items", []),
    }


def _md(text) -> str:
    return " ".join(str(text).split()).replace("|", "/")


def render(s: dict) -> str:
    L: list[str] = ["# Run stats", "",
                    f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
                    f"- pricing: ${PRICE_IN_PER_M:.2f}/M prompt tokens, "
                    f"${PRICE_OUT_PER_M:.2f}/M completion tokens", ""]

    L += ["## Episodes", ""]
    parsed = "-" if s["n_parsed"] is None else s["n_parsed"]
    L += [f"- screenplays parsed: {parsed}", f"- episodes processed (with output): {s['n_episodes']}"]
    m = s["models"]
    L += [f"- answered by primary ({s['primary_model']}): {m.get('primary', 0)}",
          f"- answered by fallback ({s['fallback_model']}): {m.get('fallback', 0)}"]
    if m.get("other") or m.get("none"):
        L.append(f"- other / unknown model: {m.get('other', 0) + m.get('none', 0)} "
                 f"({dict(s['model_names'])})")
    L.append("")

    L += ["## Candidates", "",
          f"- proposed: {s['proposed']}", f"- accepted: {s['accepted']}",
          f"- rejected: {sum(s['rejected'].values())}"]
    L += [f"  - {r}: {n}" for r, n in s["rejected"].most_common()]
    L.append("")

    L += ["## Score distribution (all accepted candidates)", "",
          f"n = {s['accepted']}. Share of 5s shows how saturated the extraction scores are.", "",
          "| score | 1 | 2 | 3 | 4 | 5 | mean | share of 5s |", "|---|---|---|---|---|---|---|---|"]
    for f, cnt in s["dist"].items():
        n = sum(cnt.values())
        mean = sum(k * v for k, v in cnt.items()) / n if n else 0.0
        share = cnt.get(5, 0) / n if n else 0.0
        L.append(f"| {f} | " + " | ".join(str(cnt.get(k, 0)) for k in range(1, 6))
                 + f" | {mean:.2f} | {share:.0%} |")
    L.append("")

    L += ["## Episodes with zero accepted candidates", ""]
    if not s["zero"]:
        L.append("none")
    else:
        L += ["| episode | rejected (by reason) | episode_note (LLM) |", "|---|---|---|"]
        for z in s["zero"]:
            rej = ", ".join(f"{r} {n}" for r, n in z["rejected"].most_common()) or "none proposed"
            L.append(f"| {z['episode']} | {rej} | {_md(z['note'])} |")
    L.append("")

    if s["audit"]:
        ok = [it for it in s["audit"] if not it.get("error")]
        d = [it["delta"] for it in ok if it.get("delta") is not None]
        L += ["## Audit (top promos, cold viewer)", "",
              f"- audited: {len(s['audit'])} ({len(s['audit']) - len(ok)} errors)",
              f"- mean followable: {sum(it['followable_score'] for it in ok) / len(ok):.2f}"
              if ok else "- mean followable: -",
              f"- mean delta (followable − standalone): {sum(d) / len(d):+.2f}" if d else "",
              f"- |delta| ≥ 2: {sum(1 for x in d if abs(x) >= 2)}",
              f"- would tap next: {sum(1 for it in ok if it.get('would_tap_next'))}/{len(ok)}", ""]

    L += ["## LLM calls and cost", "",
          "Pipeline = calls whose prompt produced the current outputs (episodes, ranking, "
          "audit), including retries and failed attempts for those prompts. All = every "
          "call in outputs/logs/calls.jsonl, including development runs.", "",
          "| | calls | ok | errors | prompt tokens | completion tokens | cost (USD) | LLM time |",
          "|---|---|---|---|---|---|---|---|"]
    for name in ("pipeline", "all"):
        t = s[name]
        L.append(f"| {name} | {t['calls']} | {t['ok']} | {t['errors']} | {t['prompt_tokens']:,} | "
                 f"{t['completion_tokens']:,} | ${t['cost_usd']:.4f} | {t['llm_seconds']:.0f}s |")
    L += ["", "## Wall time", "",
          f"- pipeline LLM time (sum of call latencies): {s['pipeline']['llm_seconds']:.0f}s",
          f"- pipeline span, first → last call: {_span(s['pipeline']['first_ts'], s['pipeline']['last_ts'])}"
          " (spans several separate invocations, including sleeps and reruns)"]
    if s["stage_seconds"] is not None:
        L.append(f"- this stage-4 invocation: {s['stage_seconds']:.0f}s")
    return "\n".join(x for x in L if x is not None) + "\n"


def write_stats(out: Path, **kw) -> tuple[Path, dict]:
    s = build_stats(out, **kw)
    path = out / "stats.md"
    path.write_text(render(s), encoding="utf-8")
    return path, s
