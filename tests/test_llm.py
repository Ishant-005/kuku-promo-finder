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


def test_dry_run_clears_last_model(dry, monkeypatch):
    monkeypatch.setattr(llm, "last_model", "stale")
    llm.complete("sys", "hello")
    assert llm.last_model is None


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


def _status_err(code: int, body=None, msg: str | None = None) -> Exception:
    cls = {
        400: llm.openai.BadRequestError,
        404: llm.openai.NotFoundError,
        429: llm.openai.RateLimitError,
        500: llm.openai.InternalServerError,
        503: llm.openai.InternalServerError,
        401: llm.openai.AuthenticationError,
        413: llm.openai.APIStatusError,
    }[code]
    return cls(msg or f"err {code}", response=_fake_response(code), body=body)


def _timeout_err() -> Exception:
    return llm.openai.APITimeoutError(request=_REQ)


class _Usage:
    def __init__(self):
        self.prompt_tokens, self.completion_tokens, self.total_tokens = 7, 3, 10


class _Resp:
    def __init__(self, text, model=None):
        msg = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = _Usage()
        self.model = model


KWARGS: list[dict] = []  # every create(**kw) seen by the fake client, per test


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
    KWARGS.clear()
    kwargs_seen = KWARGS
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key, timeout, max_retries):
            assert timeout == 60 and max_retries == 0
            self.base_url = base_url
            outer = self

            class _Completions:
                def create(self, **kw):
                    calls.append((outer.base_url, kw["model"]))
                    kwargs_seen.append(kw)
                    outcome = script[kw["model"]].pop(0)
                    if isinstance(outcome, Exception):
                        raise outcome
                    return _Resp(outcome, model=f"{kw['model']}-served")

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
    assert llm.last_model == "prim"
    assert calls == [("http://primary", "prim")]
    assert sleeps == []
    (rec,) = _log(log)
    assert rec["model"] == "prim" and rec["fallback_used"] is False
    assert rec["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert rec["prompt_hash"] == llm._prompt_hash("s", "u")
    assert rec["response"] == "hi"
    assert rec["response_model"] == "prim-served" and llm.last_response_model == "prim-served"
    assert {"ts", "model", "prompt_hash", "usage", "response", "fallback_used"} <= rec.keys()


@pytest.mark.parametrize("code", [400, 404])
def test_model_error_falls_back_immediately(live, code):
    script, calls, sleeps, log = live
    script["prim"] = [_status_err(code)]
    script["fb-model"] = ["from fallback"]
    assert llm.complete("s", "u") == "from fallback"
    assert calls == [("http://primary", "prim"), ("http://fb", "fb-model")]
    assert llm.last_model == "fb-model"
    assert sleeps == []  # no retry before falling back
    recs = _log(log)
    assert [(r["model"], r["fallback_used"]) for r in recs] == [
        ("prim", False), ("fb-model", True)]
    assert "error" in recs[0] and recs[0]["usage"] is None
    assert recs[1]["response"] == "from fallback"
    assert recs[1]["model"] == "fb-model" and recs[1]["response_model"] == "fb-model-served"
    assert "response_model" not in recs[0]  # error records have no response


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


# ---------------------------------------------------------------------------
# reasoning_effort, max_tokens default, error-body logging
# ---------------------------------------------------------------------------

def _reasoning_400():
    body = {"error": {"message": "Unrecognized request argument supplied: reasoning_effort",
                      "type": "invalid_request_error"}}
    return _status_err(400, body=body, msg="Error code: 400")


def test_reasoning_effort_default_low_and_max_tokens_3000(live):
    script, calls, sleeps, log = live
    script["prim"] = ["ok"]
    llm.complete("s", "u")
    (kw,) = KWARGS
    assert kw["extra_body"] == {"reasoning_effort": "low"}
    assert kw["max_tokens"] == 3000
    assert _log(log)[0]["reasoning_effort"] == "low"


def test_reasoning_effort_from_env_and_off(live, monkeypatch):
    script, calls, sleeps, log = live
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    script["prim"] = ["a", "b"]
    llm.complete("s", "u")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "off")
    llm.complete("s", "u")
    assert KWARGS[0]["extra_body"] == {"reasoning_effort": "high"}
    assert "extra_body" not in KWARGS[1]


def test_reasoning_400_retries_once_without_param_same_endpoint(live):
    script, calls, sleeps, log = live
    script["prim"] = [_reasoning_400(), "ok without"]
    script["fb-model"] = ["should not be used"]
    assert llm.complete("s", "u") == "ok without"
    assert calls == [("http://primary", "prim")] * 2  # no fallback
    assert sleeps == []
    assert "extra_body" in KWARGS[0]
    assert "extra_body" not in KWARGS[1]
    recs = _log(log)
    assert recs[0]["reasoning_effort_dropped"] is True
    assert "reasoning_effort" in recs[0]["error"]  # body text is logged
    assert recs[1]["reasoning_effort"] is None and recs[1]["response"] == "ok without"


def test_reasoning_400_only_dropped_once_then_falls_back(live):
    script, calls, sleeps, log = live
    script["prim"] = [_reasoning_400(), _reasoning_400()]
    script["fb-model"] = ["fb ok"]
    assert llm.complete("s", "u") == "fb ok"
    assert calls == [("http://primary", "prim")] * 2 + [("http://fb", "fb-model")]
    recs = _log(log)
    assert [r.get("reasoning_effort_dropped", False) for r in recs] == [True, False, False]
    # fallback endpoint starts fresh, with the param again
    assert KWARGS[2]["extra_body"] == {"reasoning_effort": "low"}


def test_unrelated_400_does_not_drop_param(live):
    script, calls, sleeps, log = live
    script["prim"] = [_status_err(400, body={"error": {"message": "bad model"}})]
    script["fb-model"] = ["fb ok"]
    assert llm.complete("s", "u") == "fb ok"
    assert calls == [("http://primary", "prim"), ("http://fb", "fb-model")]
    assert "reasoning_effort_dropped" not in _log(log)[0]


def test_error_log_includes_truncated_body(live):
    script, calls, sleeps, log = live
    script["prim"] = [_status_err(401, body={"error": {"message": "x" * 1000}})]
    with pytest.raises(llm.openai.AuthenticationError):
        llm.complete("s", "u")
    err = _log(log)[0]["error"]
    assert err.startswith("AuthenticationError: 401 {")
    assert '"message": "xxx' in err
    body_part = err[len("AuthenticationError: 401 "):]
    assert len(body_part) == llm.ERROR_BODY_MAX


def test_error_log_falls_back_to_str_when_no_body(live):
    script, calls, sleeps, log = live
    script["prim"] = [_status_err(401, msg="plain message")]
    with pytest.raises(llm.openai.AuthenticationError):
        llm.complete("s", "u")
    assert "plain message" in _log(log)[0]["error"]


# --- LLM_MAX_TOKENS and 413 self-healing --------------------------------------

def _tpm_413():
    return _status_err(413, body={"error": {"message": "Request too large for model on tokens "
                                            "per minute (TPM): Limit 8000, Requested 8282"}})


def test_max_tokens_from_env(live, monkeypatch):
    script, calls, sleeps, log = live
    script["prim"] = ["a", "b"]
    monkeypatch.setenv("LLM_MAX_TOKENS", "2000")
    llm.complete("s", "u")
    llm.complete("s", "u", max_tokens=500)       # explicit argument wins
    assert [k["max_tokens"] for k in KWARGS] == [2000, 500]
    assert _log(log)[0]["max_tokens"] == 2000


@pytest.mark.parametrize("bad", ["abc", "0", "-5"])
def test_max_tokens_env_invalid(live, monkeypatch, bad):
    monkeypatch.setenv("LLM_MAX_TOKENS", bad)
    with pytest.raises(RuntimeError, match="LLM_MAX_TOKENS"):
        llm.complete("s", "u")


def test_413_tokens_retries_once_with_halved_max_tokens(live):
    script, calls, sleeps, log = live
    script["prim"] = [_tpm_413(), "ok smaller"]
    script["fb-model"] = ["should not be used"]
    assert llm.complete("s", "u") == "ok smaller"
    assert calls == [("http://primary", "prim")] * 2 and sleeps == []
    assert [k["max_tokens"] for k in KWARGS] == [3000, 1500]
    recs = _log(log)
    assert recs[0]["max_tokens_reduced"] is True and recs[0]["max_tokens_next"] == 1500
    assert recs[0]["attempt"] == 1 and recs[1]["attempt"] == 1   # not counted
    assert "max_tokens_reduced" not in recs[1] and recs[1]["max_tokens"] == 1500


def test_413_halving_has_floor_1200(live):
    script, calls, sleeps, log = live
    script["prim"] = [_tpm_413(), "ok"]
    llm.complete("s", "u", max_tokens=2000)
    assert [k["max_tokens"] for k in KWARGS] == [2000, 1200]


def test_413_only_reduced_once_then_raises(live):
    script, calls, sleeps, log = live
    script["prim"] = [_tpm_413(), _tpm_413()]
    script["fb-model"] = ["should not be used"]
    with pytest.raises(llm.openai.APIStatusError):
        llm.complete("s", "u")
    assert calls == [("http://primary", "prim")] * 2
    assert [r.get("max_tokens_reduced", False) for r in _log(log)] == [True, False]


def test_413_not_reduced_at_or_below_floor(live):
    script, calls, sleeps, log = live
    script["prim"] = [_tpm_413()]
    with pytest.raises(llm.openai.APIStatusError):
        llm.complete("s", "u", max_tokens=1200)
    assert len(calls) == 1 and "max_tokens_reduced" not in _log(log)[0]


def test_413_without_tokens_in_body_is_not_reduced(live):
    script, calls, sleeps, log = live
    script["prim"] = [_status_err(413, body={"error": {"message": "payload too big"}})]
    with pytest.raises(llm.openai.APIStatusError):
        llm.complete("s", "u")
    assert len(calls) == 1


def test_413_reduction_does_not_eat_retry_budget(live):
    script, calls, sleeps, log = live
    script["prim"] = [_tpm_413()] + [_status_err(429) for _ in range(4)] + ["ok"]
    assert llm.complete("s", "u") == "ok"
    assert len(calls) == 6 and [k["max_tokens"] for k in KWARGS][-1] == 1500
