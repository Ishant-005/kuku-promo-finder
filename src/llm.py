"""Single gateway for all LLM access (OpenAI-compatible endpoints from env).

- Primary: LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
- Fallback (optional): LLM_FALLBACK_BASE_URL / LLM_FALLBACK_API_KEY / LLM_FALLBACK_MODEL.
- timeout=60s, SDK retries disabled; our own loop retries 429/5xx/timeout
  (5 attempts, backoff 2/4/8/16 s between them).
- Fallback is tried immediately on 400/404 from the primary, or after the
  primary's retries are exhausted on 429/5xx/timeout. The fallback gets the
  same retry policy; there is no second fallback.
- Every real call attempt is appended to outputs/logs/calls.jsonl.
- DRY_RUN=1: no network; prompt is written to outputs/logs/dry/ instead.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import openai

from . import config

TIMEOUT_S = 60
MAX_ATTEMPTS = 5
BACKOFF_S = (2, 4, 8, 16, 32)  # BACKOFF_S[i] = sleep after failed attempt i+1

LOG_DIR: Path = config.ROOT / "outputs" / "logs"


@dataclass(frozen=True)
class _Endpoint:
    base_url: str
    api_key: str
    model: str


_clients: dict[tuple[str, str], openai.OpenAI] = {}

# Model that produced the most recent successful response (primary or fallback).
# None after a DRY_RUN call. Single-threaded use only.
last_model: str | None = None


def _get_client(ep: _Endpoint) -> openai.OpenAI:
    key = (ep.base_url, ep.api_key)
    if key not in _clients:
        _clients[key] = openai.OpenAI(
            base_url=ep.base_url,
            api_key=ep.api_key,
            timeout=TIMEOUT_S,
            max_retries=0,
        )
    return _clients[key]


def _primary() -> _Endpoint:
    return _Endpoint(
        config.require("LLM_BASE_URL"),
        config.require("LLM_API_KEY"),
        config.require("LLM_MODEL"),
    )


def _fallback() -> _Endpoint | None:
    """Fallback endpoint, or None if not configured. URL/key default to primary's."""
    model = config.get("LLM_FALLBACK_MODEL")
    if model is None:
        return None
    return _Endpoint(
        config.get("LLM_FALLBACK_BASE_URL") or config.require("LLM_BASE_URL"),
        config.get("LLM_FALLBACK_API_KEY") or config.require("LLM_API_KEY"),
        model,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n\x00\n{user}".encode("utf-8")).hexdigest()[:12]


def _append_log(record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "calls.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _status(exc: Exception) -> int | None:
    return exc.status_code if isinstance(exc, openai.APIStatusError) else None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, openai.APITimeoutError):
        return True
    s = _status(exc)
    return s is not None and (s == 429 or 500 <= s < 600)


def _is_model_error(exc: Exception) -> bool:
    return _status(exc) in (400, 404)


def _usage_dict(usage) -> dict | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return {
        k: getattr(usage, k, None)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _dry_run(system: str, user: str, json_mode: bool, max_tokens: int, phash: str) -> str:
    dry_dir = LOG_DIR / "dry"
    dry_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    payload = {
        "ts": _now(),
        "model": config.get("LLM_MODEL"),
        "prompt_hash": phash,
        "json_mode": json_mode,
        "max_tokens": max_tokens,
        "system": system,
        "user": user,
    }
    (dry_dir / f"{stamp}_{phash}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return '{"dry_run": true}'


def _call_with_retries(ep: _Endpoint, base_kwargs: dict, phash: str, fallback_used: bool) -> str:
    """Try one endpoint up to MAX_ATTEMPTS times. Raises the last exception."""
    client = _get_client(ep)
    kwargs = {**base_kwargs, "model": ep.model}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - logged, then retried or re-raised
            _append_log({
                "ts": _now(),
                "model": ep.model,
                "prompt_hash": phash,
                "fallback_used": fallback_used,
                "attempt": attempt,
                "latency_ms": round((time.monotonic() - t0) * 1000),
                "usage": None,
                "error": f"{type(exc).__name__}: {_status(exc) or ''} {exc}"[:2000],
            })
            if attempt < MAX_ATTEMPTS and _is_retryable(exc):
                time.sleep(BACKOFF_S[attempt - 1])
                continue
            raise

        text = (resp.choices[0].message.content or "") if resp.choices else ""
        _append_log({
            "ts": _now(),
            "model": ep.model,
            "prompt_hash": phash,
            "fallback_used": fallback_used,
            "attempt": attempt,
            "latency_ms": round((time.monotonic() - t0) * 1000),
            "usage": _usage_dict(resp.usage),
            "response": text,
        })
        global last_model
        last_model = ep.model
        return text

    raise RuntimeError("unreachable")  # pragma: no cover


def complete(system: str, user: str, *, json_mode: bool = True, max_tokens: int = 4000) -> str:
    """Send one chat completion; return the raw response text."""
    global last_model
    last_model = None
    phash = _prompt_hash(system, user)
    if config.dry_run():
        return _dry_run(system, user, json_mode, max_tokens, phash)

    base_kwargs: dict = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    }
    if json_mode:
        base_kwargs["response_format"] = {"type": "json_object"}

    try:
        return _call_with_retries(_primary(), base_kwargs, phash, fallback_used=False)
    except Exception as exc:  # noqa: BLE001
        fb = _fallback()
        if fb is None or not (_is_model_error(exc) or _is_retryable(exc)):
            raise
        return _call_with_retries(fb, base_kwargs, phash, fallback_used=True)
