"""Environment config. Loads .env (python-dotenv) exactly once per process."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

_loaded = False


def load() -> None:
    """Load ROOT/.env once. Real env vars win over .env values."""
    global _loaded
    if not _loaded:
        load_dotenv(ROOT / ".env", override=False)
        _loaded = True


def get(name: str, default: str | None = None) -> str | None:
    load()
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def require(name: str) -> str:
    val = get(name)
    if val is None:
        raise RuntimeError(f"Missing required env var {name} (set it in .env)")
    return val


def dry_run() -> bool:
    return get("DRY_RUN", "0") == "1"
