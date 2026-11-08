"""DRY_RUN coverage only. No network: the client constructor is booby-trapped."""
import json

import pytest

from src import llm


@pytest.fixture
def dry(tmp_path, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "1")
    monkeypatch.setattr(llm, "LOG_DIR", tmp_path)
    monkeypatch.setattr(llm, "_clients", {})

    def _no_network(*a, **k):
        raise AssertionError("network client created during DRY_RUN")

    monkeypatch.setattr(llm.openai, "OpenAI", _no_network)
    return tmp_path


def test_dry_run_returns_sentinel(dry):
    out = llm.complete("sys", "hello")
    assert json.loads(out) == {"dry_run": True}


def test_dry_run_writes_prompt_file(dry):
    llm.complete("SYSTEM TEXT", "USER TEXT", json_mode=False, max_tokens=123)
    files = list((dry / "dry").glob("*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    assert rec["system"] == "SYSTEM TEXT"
    assert rec["user"] == "USER TEXT"
    assert rec["json_mode"] is False
    assert rec["max_tokens"] == 123
    assert len(rec["prompt_hash"]) == 12
    assert files[0].name.endswith(f"_{rec['prompt_hash']}.json")


def test_dry_run_does_not_log_calls(dry):
    llm.complete("sys", "hello")
    assert not (dry / "calls.jsonl").exists()


def test_dry_run_one_file_per_call(dry):
    llm.complete("sys", "a")
    llm.complete("sys", "b")
    assert len(list((dry / "dry").glob("*.json"))) == 2


def test_prompt_hash_is_stable_and_input_sensitive():
    assert llm._prompt_hash("s", "u") == llm._prompt_hash("s", "u")
    assert llm._prompt_hash("s", "u") != llm._prompt_hash("s", "u2")
    assert llm._prompt_hash("ab", "c") != llm._prompt_hash("a", "bc")


# ---------------------------------------------------------------------------
# Retry + fallback path. Fake OpenAI client; no network, no real sleeps.
# ---------------------------------------------------------------------------
from types import SimpleNamespace

from src import config

# Duck-typed stand-ins for the SDK's transport Request/Response: the openai
# exception constructors only read .status_code, .headers and .request.
_REQ = SimpleNamespace(method="POST", url="http://fake/chat/completions")


def _fake_response(code: int):
    return SimpleNamespace(status_code=code, headers={}, request=_REQ)


def _status_err(code: int) -> Exception:
    cls = {
        400: llm.openai.BadRequestError,
        404: llm.openai.NotFoundError,
        429: llm.openai.RateLimitError,
        500: llm.openai.InternalServerError,
        503: llm.openai.InternalServerError,
        401: llm.openai.AuthenticationError,
    }[code]
    return cls(f"err {code}", response=_fake_response(code), body=None)


def _timeout_err() -> Exception:
    return llm.openai.APITimeoutError(request=_REQ)


class _Usage:
    def __init__(self):
        self.prompt_tokens, self.completion_tokens, self.total_tokens = 7, 3, 10


class _Resp:
    def __init__(self, text):
        msg = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = _Usage()


@pytest.fixture
def live(tmp_path, monkeypatch):
    """Configure primary+fallback env, fake client, capture sleeps.

    Returns (script, calls, sleeps, log_path). `script[model]` is a list of
    outcomes consumed in order: an Exception is raised, a str is returned.
    """
    monkeypatch.setattr(config, "_loaded", True)  # never read the real .env
    for k, v in {
        "DRY_RUN": "0",
        "LLM_BASE_URL": "http://primary", "LLM_API_KEY": "pk", "LLM_MODEL": "prim",
        "LLM_FALLBACK_BASE_URL": "http://fb", "LLM_FALLBACK_API_KEY": "fk",
        "LLM_FALLBACK_MODEL": "fb-model",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(llm, "LOG_DIR", tmp_path)
    monkeypatch.setattr(llm, "_clients", {})

    script: dict[str, list] = {}
    calls: list[tuple[str, str]] = []  # (base_url, model)
    sleeps: list[float] = []

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key, timeout, max_retries):
            assert timeout == 60 and max_retries == 0
            self.base_url = base_url
            outer = self

            class _Completions:
                def create(self, **kw):
                    calls.append((outer.base_url, kw["model"]))
                    outcome = script[kw["model"]].pop(0)
                    if isinstance(outcome, Exception):
                        raise outcome
                    return _Resp(outcome)

            self.chat = type("Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr(llm.openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    return script, calls, sleeps, tmp_path / "calls.jsonl"


def _log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_primary_success_no_fallback(live):
    script, calls, sleeps, log = live
    script["prim"] = ["hi"]
    assert llm.complete("s", "u") == "hi"
    assert calls == [("http://primary", "prim")]
    assert sleeps == []
    (rec,) = _log(log)
    assert rec["model"] == "prim" and rec["fallback_used"] is False
    assert rec["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert rec["prompt_hash"] == llm._prompt_hash("s", "u")
    assert rec["response"] == "hi"
    assert {"ts", "model", "prompt_hash", "usage", "response", "fallback_used"} <= rec.keys()


@pytest.mark.parametrize("code", [400, 404])
def test_model_error_falls_back_immediately(live, code):
    script, calls, sleeps, log = live
    script["prim"] = [_status_err(code)]
    script["fb-model"] = ["from fallback"]
    assert llm.complete("s", "u") == "from fallback"
    assert calls == [("http://primary", "prim"), ("http://fb", "fb-model")]
    assert sleeps == []  # no retry before falling back
    recs = _log(log)
    assert [(r["model"], r["fallback_used"]) for r in recs] == [
        ("prim", False), ("fb-model", True)]
    assert "error" in recs[0] and recs[0]["usage"] is None
    assert recs[1]["response"] == "from fallback"


@pytest.mark.parametrize("make_err", [
    lambda: _status_err(429), lambda: _status_err(500),
    lambda: _status_err(503), _timeout_err,
])
def test_retryable_exhausts_then_falls_back(live, make_err):
    script, calls, sleeps, log = live
    script["prim"] = [make_err() for _ in range(5)]
    script["fb-model"] = ["ok"]
    assert llm.complete("s", "u") == "ok"
    assert calls.count(("http://primary", "prim")) == 5
    assert calls[-1] == ("http://fb", "fb-model")
    assert sleeps == [2, 4, 8, 16]
    recs = _log(log)
    assert [r["attempt"] for r in recs] == [1, 2, 3, 4, 5, 1]
    assert [r["fallback_used"] for r in recs] == [False] * 5 + [True]


def test_retry_recovers_on_primary_without_fallback(live):
    script, calls, sleeps, log = live
    script["prim"] = [_status_err(429), _status_err(503), "third time"]
    assert llm.complete("s", "u") == "third time"
    assert all(c == ("http://primary", "prim") for c in calls)
    assert sleeps == [2, 4]
    assert all(r["fallback_used"] is False for r in _log(log))


def test_non_retryable_non_model_error_does_not_fall_back(live):
    script, calls, sleeps, log = live
    script["prim"] = [_status_err(401)]
    with pytest.raises(llm.openai.AuthenticationError):
        llm.complete("s", "u")
    assert calls == [("http://primary", "prim")]
    assert sleeps == []


def test_fallback_also_retries_then_raises(live):
    script, calls, sleeps, log = live
    script["prim"] = [_status_err(404)]
    script["fb-model"] = [_status_err(503) for _ in range(5)]
    with pytest.raises(llm.openai.InternalServerError):
        llm.complete("s", "u")
    assert calls.count(("http://fb", "fb-model")) == 5
    assert sleeps == [2, 4, 8, 16]
    assert len(_log(log)) == 6


def test_no_fallback_configured_reraises(live, monkeypatch):
    script, calls, sleeps, log = live
    monkeypatch.delenv("LLM_FALLBACK_MODEL")
    script["prim"] = [_status_err(404)]
    with pytest.raises(llm.openai.NotFoundError):
        llm.complete("s", "u")
    assert calls == [("http://primary", "prim")]


def test_fallback_url_and_key_default_to_primary(live, monkeypatch):
    script, calls, sleeps, log = live
    monkeypatch.delenv("LLM_FALLBACK_BASE_URL")
    monkeypatch.delenv("LLM_FALLBACK_API_KEY")
    script["prim"] = [_status_err(404)]
    script["fb-model"] = ["ok"]
    assert llm.complete("s", "u") == "ok"
    assert calls[-1] == ("http://primary", "fb-model")
