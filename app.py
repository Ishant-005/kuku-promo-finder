"""Streamlit front-end for the promo finder.

Runs the same pipeline as run.py (parse -> extract -> validate -> rank [-> audit])
on screenplays the user uploads, against any OpenAI-compatible endpoint entered
in the sidebar. Nothing from the input pack ships with the app; outputs shown
on screen quote only the hook and cliffhanger lines.

    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from src import audit, extract, llm, rank, stats
from src.parse import parse_episode

st.set_page_config(page_title="Promo finder", page_icon="🎬", layout="wide")

# --- helpers ------------------------------------------------------------------

SECRET_KEYS = ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_FALLBACK_MODEL",
               "LLM_REASONING_EFFORT", "LLM_MAX_TOKENS"]


def _secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # no secrets.toml
        return os.environ.get(name, default)


def _apply_env(values: dict[str, str]) -> None:
    """Sidebar values win over .env (config.get reads os.environ first)."""
    for k, v in values.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    llm._clients.clear()  # new endpoint/key -> new client


def _session_out() -> Path:
    if "out_dir" not in st.session_state:
        st.session_state.out_dir = Path(tempfile.mkdtemp(prefix="promo_"))
    out = st.session_state.out_dir
    llm.LOG_DIR = out / "logs"
    return out


def _read_uploads(files) -> list[tuple[str, bytes]]:
    """Accept .txt files and/or .zip archives; return (name, bytes) for each .txt."""
    found: list[tuple[str, bytes]] = []
    for f in files:
        data = f.read()
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    n = info.filename
                    if n.lower().endswith(".txt") and "__MACOSX" not in n and not Path(n).name.startswith("."):
                        found.append((Path(n).name, z.read(info)))
        elif f.name.lower().endswith(".txt"):
            found.append((f.name, data))
    return found


def _parse_all(found: list[tuple[str, bytes]], out: Path):
    src_dir = out / "input"
    src_dir.mkdir(parents=True, exist_ok=True)
    episodes, errors = [], []
    for name, data in found:
        p = src_dir / name
        p.write_bytes(data)
        try:
            episodes.append(parse_episode(p))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    episodes.sort(key=lambda e: e.number)
    return episodes, errors


# --- sidebar: model endpoint ----------------------------------------------------

with st.sidebar:
    st.header("Model endpoint")
    st.caption("Any OpenAI-compatible API. Swap the model here; the pipeline does not change.")
    base_url = st.text_input(
        "Base URL", _secret("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        help=(
            "The address of the provider's OpenAI-compatible API. The app sends every model call to "
            "`<Base URL>/chat/completions`, so any provider that speaks the OpenAI format works — only the "
            "URL, key and model name change.\n\n"
            "| provider | base URL |\n|---|---|\n"
            "| Groq | `https://api.groq.com/openai/v1` |\n"
            "| Cerebras | `https://api.cerebras.ai/v1` |\n"
            "| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` |\n"
            "| Mistral | `https://api.mistral.ai/v1` |\n"
            "| OpenRouter | `https://openrouter.ai/api/v1` |\n"
            "| OpenAI | `https://api.openai.com/v1` |\n"
            "| Together | `https://api.together.xyz/v1` |\n"
            "| Ollama (local) | `http://localhost:11434/v1` |\n\n"
            "Copy it exactly, including any trailing slash. A wrong URL shows as a connection error or 404 in the run log."
        ),
    )
    api_key = st.text_input("API key", _secret("LLM_API_KEY"), type="password",
                            help="Your key for the provider above. Used only for this session's calls; never stored by the app.")
    model = st.text_input("Model", _secret("LLM_MODEL", "openai/gpt-oss-120b"),
                          help="The exact model id as the provider lists it, e.g. `openai/gpt-oss-120b` (Groq), "
                               "`gpt-oss-120b` (Cerebras), `mistral-small-latest` (Mistral). A wrong id shows as 404.")
    fallback = st.text_input("Fallback model (optional, same endpoint)", "",
                             help="Leave empty unless the endpoint above serves a second model you want tried when the first fails.")
    with st.expander("Advanced"):
        effort_choice = st.selectbox(
            "Reasoning effort", ["auto", "low", "medium", "high", "off"], index=0,
            help="auto: send 'low' only when the model name looks like a reasoning model "
                 "(gpt-oss, o1/o3/o4, deepseek-r1, qwen3, gemini 2.5+/3.x); otherwise send nothing. "
                 "'off' never sends it. If a provider rejects the parameter, the wrapper retries once without it.")
        max_tokens = st.text_input("Max completion tokens", _secret("LLM_MAX_TOKENS", "3000"))
        sleep_s = st.number_input("Seconds between episodes (free-tier rate limits)", 0.0, 120.0, 20.0, 1.0)
        audit_sleep = st.number_input("Seconds between audit calls", 0.0, 60.0, 5.0, 1.0)
    st.divider()
    st.caption("Presets: Groq `https://api.groq.com/openai/v1` · Gemini "
               "`https://generativelanguage.googleapis.com/v1beta/openai/` · OpenAI `https://api.openai.com/v1`")


REASONING_PATTERNS = ("gpt-oss", "o1", "o3", "o4", "deepseek-r1", "r1-", "qwen3", "qwen-3", "qwq", "thinking",
                      "gemini-2.5", "gemini-3", "grok-3-mini", "grok-4", "sonar-reasoning")


def _is_reasoning_model(name: str) -> bool:
    n = (name or "").lower()
    return any(tok in n for tok in REASONING_PATTERNS)


def _resolve_effort(choice: str, model_name: str) -> str:
    if choice != "auto":
        return choice
    return "low" if _is_reasoning_model(model_name) else "off"


effort = _resolve_effort(effort_choice, model)
with st.sidebar:
    if effort_choice == "auto":
        st.caption(f"Reasoning effort auto → **{effort}** for `{model or '…'}`"
                   + ("" if effort == "low" else " (treated as a non-reasoning model; nothing sent)"))
_apply_env({
    "LLM_BASE_URL": base_url, "LLM_API_KEY": api_key, "LLM_MODEL": model,
    "LLM_FALLBACK_MODEL": fallback, "LLM_FALLBACK_BASE_URL": "", "LLM_FALLBACK_API_KEY": "",
    "LLM_REASONING_EFFORT": effort, "LLM_MAX_TOKENS": max_tokens,
    "LLM_SLEEP_S": str(sleep_s), "AUDIT_SLEEP_S": str(audit_sleep),
})

# --- main ---------------------------------------------------------------------

st.title("🎬 Promo finder")
st.write(
    "Upload microdrama screenplays (the `.txt` files, or a `.zip` of them), pick episodes, "
    "and get ranked 20–60 s promo clips: episode, start/end timestamps, hook, cliffhanger, "
    "and why it works. The LLM proposes clips by **beat ID**; code resolves timestamps, "
    "validates (20–60 s, no mid-line cuts, no duplicates) and ranks."
)

tab_run, tab_view, tab_manual, tab_about = st.tabs(["Run on screenplays", "View a previous run", "User manual", "How it works"])

with tab_run:
    files = st.file_uploader("Screenplays (.txt) or a .zip of them", type=["txt", "zip"],
                             accept_multiple_files=True)
    found = _read_uploads(files) if files else []
    out = _session_out()

    if found:
        episodes, perrs = _parse_all(found, out)
        st.success(f"Parsed {len(episodes)} episode(s), "
                   f"{sum(len(e.beats) for e in episodes)} beats, "
                   f"{sum(e.duration_s for e in episodes)/60:.1f} min of screen time.")
        for e in perrs:
            st.warning(e)

        nums = [e.number for e in episodes]
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            chosen = st.multiselect("Episodes to run (one LLM call each)", nums,
                                    default=nums[:min(3, len(nums))])
        with c2:
            do_audit = st.checkbox("Cold-viewer audit on final promos", value=False,
                                   help="One extra small LLM call per final promo (≤15).")
        with c3:
            est = len(chosen) + 1 + (15 if do_audit else 0)
            st.metric("LLM calls", est)
            st.caption(f"≈ {len(chosen) * sleep_s / 60:.0f} min at current sleep")

        blockers = []
        if not api_key:
            blockers.append("enter an **API key** in the sidebar")
        if not base_url:
            blockers.append("enter a **Base URL** in the sidebar")
        if not model:
            blockers.append("enter a **Model** name in the sidebar")
        if not chosen:
            blockers.append("select at least one **episode** above")
        if blockers:
            st.info("**Find promos** is disabled until you " + "; ".join(blockers) + ".")
        if st.button("Find promos", type="primary", disabled=bool(blockers)):
            run_eps = [e for e in episodes if e.number in chosen]
            # fresh run: drop results from any earlier run in this session
            import shutil
            shutil.rmtree(out / "episodes", ignore_errors=True)
            for stale in ("promos.json", "promos.csv", "ranked.md", "audit.json", "stats.md"):
                (out / stale).unlink(missing_ok=True)
            st.session_state.ran = False
            prog = st.progress(0.0, text="Extracting…")
            log = st.empty()
            lines: list[str] = []
            rejected: Counter = Counter()
            accepted = failed = 0
            t_start = time.monotonic()
            for i, ep in enumerate(run_eps):
                if i and sleep_s:
                    for s in range(int(sleep_s), 0, -1):
                        prog.progress(i / len(run_eps), text=f"Rate-limit pause {s}s before ep{ep.number:03d}…")
                        time.sleep(1)
                prog.progress(i / len(run_eps), text=f"ep{ep.number:03d}: asking the model…")
                t0 = time.monotonic()
                try:
                    res = extract.extract_episode(ep)
                    extract.write_result(res, out)
                    accepted += len(res.accepted)
                    rejected.update((c.reject_reason or "").split(":")[0] for c in res.rejected)
                    lines.append(f"ep{ep.number:03d} ok · accepted {len(res.accepted)} · rejected {len(res.rejected)} "
                                 f"· model {res.response_model or res.model} · {time.monotonic()-t0:.1f}s")
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    lines.append(f"ep{ep.number:03d} FAILED {type(exc).__name__}: {str(exc)[:160]}")
                log.code("\n".join(lines))
            rres = None
            if accepted == 0:
                prog.progress(1.0, text="Stopped: nothing to rank")
                if failed == len(run_eps):
                    st.error("Every episode failed before validation (see the log above). Usually a wrong model id (404), "
                             "a key problem (401), a rate limit (429), or a model that returned no JSON — "
                             "try a different model or set Reasoning effort to `low`.")
                else:
                    st.warning("The model answered, but every proposed clip was rejected by the validator "
                               "(unknown beat IDs, bad length, mid-line cuts). Open the rejected-candidates list below, "
                               "or re-run — a retry often succeeds.")
            else:
                prog.progress(1.0, text="Ranking (one LLM call)…")
                try:
                    rres = rank.rank_all(out)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Ranking failed: {type(exc).__name__}: {exc}")
            if do_audit and rres:
                prog.progress(1.0, text="Cold-viewer audit…")
                try:
                    audit.audit_all(out, run_eps)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Audit failed: {type(exc).__name__}: {exc}")
            try:
                stats.write_stats(out, n_parsed=len(episodes))
            except Exception:  # noqa: BLE001
                pass
            if rres:
                prog.progress(1.0, text=f"Done in {time.monotonic()-t_start:.0f}s")
                st.session_state.ran = True

    if (out / "episodes").exists() and not st.session_state.get("ran") and any((out / "episodes").glob("ep*.json")):
        with st.expander("Rejected candidates from this run", expanded=True):
            rows = []
            for ej in sorted((out / "episodes").glob("ep*.json")):
                d = json.loads(ej.read_text(encoding="utf-8"))
                for c in d.get("rejected", []):
                    segs = c.get("segments") or []
                    rows.append({"episode": d.get("episode"), "candidate": c.get("candidate_id"),
                                 "reason": c.get("reject_reason"),
                                 "proposed_ids": "; ".join(f"{x.get('start_id')}→{x.get('end_id')}" for x in segs),
                                 "model": d.get("response_model") or d.get("model")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True) if rows else st.write("No rejections recorded.")

    if st.session_state.get("ran") and (out / "promos.csv").exists():
        st.subheader("Ranked promos")
        df = pd.read_csv(out / "promos.csv")
        df = df.rename(columns={"duration": "duration_s", "why": "why_it_works",
                                "h": "hook_score", "c": "cliff_score", "s": "standalone_score"})
        show_cols = [c for c in ["rank", "episode", "segments", "duration_s", "hook_type", "hook",
                                 "cliffhanger", "why_it_works", "hook_score", "cliff_score",
                                 "standalone_score", "score"] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

        with st.expander("How the score is calculated"):
            st.markdown(
                """
`score = 0.40 × hook + 0.35 × cliffhanger + 0.25 × standalone + duration bonus − same-episode penalty`

- **hook / cliffhanger / standalone** — 1–5 ratings from the extraction model (one LLM call per episode).
- **duration bonus** — +0.3 if the clip is 25–45 s (the sweet spot for Reels/Shorts), else 0.
- **same-episode penalty** — 0 for the best clip from an episode, −0.5 for the second, −1.0 for the third, −2.0 beyond. Keeps the list from being ten clips of episode 1.

Example: 5/5/5, 30 s, best in its episode → 5.0 + 0.3 = **5.3**. 4/5/5, 22 s, second in its episode → 4.6 − 0.5 = **4.1**.

The score picks the top 30 candidates. The **final posting order** is a separate LLM re-rank of those 30 (`rank_reason` column); the score is kept alongside for transparency. Note the model's ratings cluster at 4–5, so the score separates clips mostly by duration and episode spread, not fine quality.
                """
            )
        m1, m2, m3, m4 = st.columns(4)
        pj = json.loads((out / "promos.json").read_text(encoding="utf-8"))
        m1.metric("Candidates accepted", pj.get("loaded", len(df)))
        m2.metric("Final promos", len(df))
        m3.metric("Ranking model", str(pj.get("response_model") or pj.get("model") or "—"))
        m4.metric("Mean duration", f"{df['duration_s'].mean():.1f}s" if "duration_s" in df else "—")

        if (out / "audit.json").exists():
            st.subheader("Cold-viewer audit")
            aj = json.loads((out / "audit.json").read_text(encoding="utf-8"))
            items = aj.get("items", aj if isinstance(aj, list) else [])
            if items:
                adf = pd.DataFrame(items)
                keep = [c for c in ["rank", "standalone_score", "followable_score", "delta",
                                    "would_tap_next", "what_is_happening", "open_question"] if c in adf.columns]
                st.dataframe(adf[keep] if keep else adf, use_container_width=True, hide_index=True)

        with st.expander("Rejected candidates (what the model proposed and why it was rejected)"):
            rows = []
            for ej in sorted((out / "episodes").glob("ep*.json")):
                d = json.loads(ej.read_text(encoding="utf-8"))
                n_beats = None
                for c in d.get("rejected", []):
                    segs = c.get("segments") or []
                    rows.append({"episode": d.get("episode"), "candidate": c.get("candidate_id"),
                                 "reason": c.get("reject_reason"),
                                 "proposed_ids": "; ".join(f"{s.get('start_id')}→{s.get('end_id')}" for s in segs),
                                 "declared_s": c.get("declared_duration_s"), "model": d.get("response_model") or d.get("model")})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.write("No rejections.")
            st.caption("Valid IDs are e<episode>-b<beat>, numbered from b001 up to the episode's beat count.")
        with st.expander("Run stats"):
            if (out / "stats.md").exists():
                st.markdown((out / "stats.md").read_text(encoding="utf-8"))

        d1, d2, d3 = st.columns(3)
        d1.download_button("promos.csv", (out / "promos.csv").read_bytes(), "promos.csv", "text/csv")
        d2.download_button("promos.json", (out / "promos.json").read_bytes(), "promos.json", "application/json")
        d3.download_button("ranked.md", (out / "ranked.md").read_bytes(), "ranked.md", "text/markdown")

with tab_view:
    st.write("Upload a `promos.csv` or `ranked.md` from a previous run (e.g. the full 50-episode run in the raw-outputs folder) to browse it here.")
    prev = st.file_uploader("promos.csv / ranked.md", type=["csv", "md"], key="prev")
    if prev is not None:
        if prev.name.endswith(".csv"):
            st.dataframe(pd.read_csv(prev), use_container_width=True, hide_index=True)
        else:
            st.markdown(prev.read().decode("utf-8", errors="replace"))

with tab_manual:
    st.markdown(
        """
### User manual

**What you need**
- The screenplays: one `.txt` per episode in the Kuku format (scene headings, timestamped CAMERA / ACTION / dialogue blocks). A `.zip` of them also works.
- An API key for any OpenAI-compatible LLM endpoint. If the sidebar is already filled in, a key has been provided for you. Free options:

| provider | base URL | reasoning model | non-reasoning model |
|---|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` | `llama-3.3-70b-versatile` (if listed for your key) |
| Cerebras | `https://api.cerebras.ai/v1` | `gpt-oss-120b` | `llama-3.3-70b` |
| Mistral | `https://api.mistral.ai/v1` | — | `mistral-small-latest` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-*-flash` (20 req/day free) | — |
| OpenRouter | `https://openrouter.ai/api/v1` | varies | models ending in `:free` |

  Model ids change; copy the exact id from the provider's model list. A wrong id shows as `404` in the run log.

**Step by step**
1. **Sidebar → Model endpoint.** Paste the Base URL, API key and the exact model id the provider lists (e.g. `openai/gpt-oss-120b` on Groq). A wrong model id gives a `404` in the run log. Leave *Fallback model* empty unless you know a second model on the *same* endpoint.
2. **Upload screenplays.** Drag the `.txt` files (or a `.zip`) into the uploader. The green bar confirms how many episodes parsed and how many beats were found. A parse warning means that file's format differs; it is skipped.
3. **Pick episodes.** The first 3 are pre-selected. Each episode is one LLM call. On a free tier, 3–5 episodes is a sensible demo; the full 50-episode run takes ~30 minutes because of rate limits.
4. **Cold-viewer audit (optional).** Adds one small call per final promo (up to 15). It re-reads each clip with no show context and scores how followable it is to a stranger. Turn it off when you are short on quota.
5. **Click Find promos.** The log shows one line per episode: accepted / rejected counts, which model answered, time taken. Then one ranking call, then (optionally) the audit.
6. **Read the results.** The table lists the final promos in posting order with episode, start–end timestamps, duration, hook, cliffhanger, why it works and the 1–5 scores. Download `promos.csv`, `promos.json` or `ranked.md` for the editor.
7. **Check the rejections.** The *Rejected candidates* expander shows every proposal the validator threw out and why (unknown beat id, too short/long, cut inside a line, duplicate). This is where to look if an episode returned nothing.

**Advanced (sidebar → Advanced)**
- *Reasoning effort* — `auto` (default) sends `low` only when the model name looks like a reasoning model (gpt-oss, o-series, DeepSeek-R1, Qwen3, Gemini 2.5+/3.x) and sends nothing otherwise. Force `low`/`off` if auto guesses wrong; a rejected parameter is retried without it automatically.
- *Max completion tokens* — 3000 is enough; lower it to 2000 if the provider returns `413` (request too large for a free-tier minute).
- *Seconds between episodes* — raise it if you see `429` rate-limit errors; 20–30 s suits Groq's free tier.

**Reading errors in the run log**
- `404` → model id does not exist on that endpoint. Check the provider's model list.
- `401` → wrong API key.
- `429` → rate limit. Increase the pause, or wait; daily caps (Gemini free: 20 requests/day per model) reset at midnight Pacific.
- `413` → request too large for the per-minute token budget. Lower max tokens.
- `json_validate_failed` / empty output → the model spent its budget reasoning. Set reasoning effort to `low`, or pick a non-reasoning model.

**Same episode, different results?** LLM output is sampled; temperature is pinned to 0 to reduce this, but providers are not perfectly deterministic. If every candidate in an episode is rejected for `unknown_id`, the pipeline re-asks once automatically.

**Privacy.** Uploaded scripts stay in a temporary folder for this browser session only and are never stored or displayed; results quote only the hook and cliffhanger lines.
        """
    )

with tab_about:
    st.markdown(
        """
**Pipeline** (same code as `python run.py`):

```
parse (code)      .txt -> numbered beats (e001-b042): kind, speaker, start/end, cumulative s
extract (LLM)     one call per episode -> up to 5 promos as beat-ID ranges + scores
validate (code)   IDs -> timestamps; reject unknown ID, <20 s, >60 s, SFX opener,
                  mid-dialogue cut, duplicate
rank (code+LLM)   weighted score + 25-45 s bonus + per-episode cap -> top 30;
                  one LLM call orders the final 15
audit (LLM)       clip beats only, no show context -> cold-viewer "followable" 1-5
```

**Why beat IDs.** The model never writes a timestamp. It picks beat IDs; code resolves
them, so a hallucinated timestamp is impossible and every rejection is checkable.

**Cost.** ~66 LLM calls per 50-episode show; ≈ $0.27 at $0.30/$2.50 per M tokens.
Free tiers are rate-limit bound (hence the pause between episodes).

**Known failure modes.** Score inflation (half of all scores are 5); the same ending
proposed with several starts (deduped); hooks that are reactions rather than events;
cutting on the answer instead of the question; occasional hallucinated beat IDs
(rejected). Details in the solution doc and `notes/ai-mistakes.md`.
        """
    )
