# Promo finder

Finds the best promo clips in a microdrama screenplay.

Give it a folder of screenplays (one `.txt` per episode, timestamped) and it returns a ranked list of 20–60 s clips that **open on a hook**, **end on a cliffhanger**, and **make sense to someone who has never seen the show** — each with episode, start/end timestamps, the hook line, the cliffhanger, scores and a one-line reason. A promo editor reviews the top 15 instead of watching every episode.

Built for the Kuku AI Generalist assignment (Problem 1). Nothing is specific to one show.

| | |
|---|---|
| **Web app** | `<streamlit link>` — upload screenplays, plug in any model, click *Find promos* |
| **Solution doc** | `docs/SOLUTION.pdf` (2 pages: approach, results, failure modes) |
| **Raw outputs** | `<drive link>` — unedited results for all 50 episodes of the input pack |
| **Runtime** | Python 3.10+, one dependency that matters: the `openai` client (used against any OpenAI-compatible endpoint) |

---

## Contents

1. [Use it with your own model](#1-use-it-with-your-own-model)
2. [Quick start](#2-quick-start)
3. [How it works](#3-how-it-works)
4. [Configuration reference](#4-configuration-reference)
5. [Outputs](#5-outputs)
6. [Results on the input pack](#6-results-on-the-input-pack)
7. [Known limitations and failure modes](#7-known-limitations-and-failure-modes)
8. [Troubleshooting](#8-troubleshooting)
9. [Project layout](#9-project-layout)
10. [Confidentiality, tests, AI use](#10-confidentiality-tests-ai-use)

---

## 1. Use it with your own model

The pipeline is model-agnostic by construction. Every LLM call goes through one function in `src/llm.py`, which speaks the OpenAI chat-completions format against whatever endpoint you give it. Prompts live in `prompts/*.md`, not in code. **Nothing else changes when you change the model.**

### Three ways in

**A. Web app — no setup.** Open the Streamlit link. In the sidebar enter three things: the provider's **Base URL**, your **API key**, and the **model id** exactly as the provider lists it. Upload `.txt` screenplays (or a `.zip`), pick episodes, click **Find promos**. The sidebar ⓘ icons list base URLs and example model ids.

**B. Command line — your machine.**

```bash
cp .env.example .env     # fill in LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
python run.py --input "data/<show folder>" --out outputs
```


### Providers that have been run

| provider | base URL | models used | notes |
|---|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` (primary), `openai/gpt-oss-20b` (fallback) | Free tier ≈ 8k tokens/min → `LLM_SLEEP_S=30` |
| Cerebras | `https://api.cerebras.ai/v1` | `gpt-oss-120b` | Same model, higher free limits |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-3.8-flash` | Free tier = **20 requests/day per model**; not enough for one show |

Also OpenAI-compatible and expected to work unchanged: Mistral (`https://api.mistral.ai/v1`), OpenRouter (`https://openrouter.ai/api/v1`), OpenAI, Together, Anthropic's compatibility endpoint, local Ollama (`http://localhost:11434/v1`). Model ids change often — copy them from the provider's model list; a wrong id shows as `404`.

### The one limit

The wrapper needs an **OpenAI-compatible** chat endpoint. A provider with only a native API would need a small adapter in `src/llm.py` (one function: `complete(system, user) -> str`) or a proxy such as LiteLLM in front. That change is confined to one file.

### Reasoning vs non-reasoning models

| | Reasoning (`gpt-oss`, Qwen 3, Gemini thinking, o-series, DeepSeek-R1) | Non-reasoning (Llama, Mistral, …) |
|---|---|---|
| What happens | Hidden thinking consumes the output budget. With a small `max_tokens` the model returns an **empty** answer with no error. One episode burned 7,900 tokens until `reasoning_effort=low` was sent. | Fast, cheap, consistent JSON. Reject the `reasoning_effort` parameter. |
| Fit for this task | Better at the many simultaneous constraints (20–60 s window, beat order, cut before the payoff). | Weaker at the taste judgment; fewer hallucinated IDs in our runs. |
| What the code does | `LLM_REASONING_EFFORT=auto` (app default) detects the family from the model name and sends `low`. | Sends nothing; retries without the parameter if a provider rejects it anyway. |

---

## 2. Quick start

```bash
# 1. Python 3.10+ (3.12 recommended)
uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt
#    (or: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt)

# 2. Model
cp .env.example .env            # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

# 3. Screenplays
#    put them under data/<show>/episode-001-….txt … (one .txt per episode)

# 4. Run
python run.py --input "data/<show>" --out outputs
#    → outputs/ranked.md, promos.csv, promos.json, audit.json, stats.md

# Web app instead of CLI
streamlit run app.py

# No API key? Build every prompt without calling anything:
DRY_RUN=1 python run.py --episodes 1      # writes prompts to outputs/dry/logs/dry/

# Tests (no network)
python -m pytest -q
```

CLI flags: `--episodes 1,10,25` (subset) · `--force` (re-extract episodes that already have output) · `--stages 3` (re-rank only) · `--stages 4` (re-audit only) · `--input` may be omitted when `data/` holds exactly one show folder.

Deploying the web app to Streamlit Community Cloud: see `DEPLOY.md`.

---

## 3. How it works

```
parse (code)      .txt -> numbered beats (e001-b042): kind, speaker, start/end, cumulative seconds
extract (LLM)     ONE call per episode -> up to 5 promos as beat-ID ranges + hook/cliff/standalone 1-5
validate (code)   IDs -> timestamps; reject unknown ID, <20 s, >60 s, SFX opener, mid-dialogue cut, duplicate
rank (code+LLM)   weighted score + duration bonus + per-episode penalty -> top 30; ONE LLM call orders final 15
audit (LLM)       each final clip's beats only, no show context -> "cold viewer" followable 1-5
stats (code)      counts, reject reasons, score distribution, tokens, cost, wall time
```

**Design principle: AI for taste, code for everything checkable.** The model decides which moments are strong and why. It never writes a timestamp. It names **beat IDs**; code resolves them. A hallucinated timestamp is therefore impossible, and every rejection is explainable from the log.

### 3.1 Parse (`src/parse.py`)

Each screenplay is split into **beats**: one per `**CAMERA:**`, `**ACTION:**`, `**SOUND FX:**` or speaker block (`**JULIANA (Defiant):**` — the parenthetical is a delivery note, not part of the name). Every beat gets an id `e<episode>-b<index>`, a kind (`CAMERA`, `ACTION`, `DIALOGUE`, `OTHER` for sound FX), speaker, start/end seconds and cumulative seconds from the first beat. Header matching is case-insensitive (one episode in the pack uses uppercase headers). `scripts/inspect.py` prints a one-line summary per episode for sanity checks.

### 3.2 Extract (`src/extract.py`, `prompts/extract.md`)

One LLM call per episode. Input: the logline plus a compact beat table (`id | kind | speaker | start | end | cum | text`, text trimmed to 160 chars). Output: strict JSON with up to 5 candidates, each a list of `{start_id, end_id}` segments plus hook, cliffhanger, three 1–5 scores, a hook type, a one-line reason and a `declared_duration_s` (so the model's arithmetic can be checked against the real length).

If a response yields zero accepted candidates and at least one `unknown_id` rejection, the model is asked once more with `prompts/extract_id_retry.md` appended ("copy IDs exactly as printed"). This rescued two of three all-invalid responses in testing.

### 3.3 Validate (code, same file)

| reject reason | rule |
|---|---|
| `unknown_id` | a beat id not in this episode |
| `out_of_order_or_overlap` | segments not in episode order, or overlapping |
| `too_short` / `too_long` | total < 20 s or > 60 s (resolved from timestamps, not the model's claim) |
| `bad_first_beat` | first beat is a sound-FX beat (camera beats are allowed: in these scripts they carry the physical events) |
| `cuts_mid_beat` | a segment boundary falls inside a **dialogue** beat (0.3 s tolerance); camera/action beats may overlap a boundary |
| `duplicate_of:<id>` | shares an end beat with, or overlaps > 50 % of, a higher-scoring candidate in the same episode |

Both accepted and rejected candidates are written to `outputs/episodes/epNNN.json`, with the reason and the model that answered.

### 3.4 Rank (`src/rank.py`, `prompts/rank.md`)

```
score = 0.40·hook + 0.35·cliffhanger + 0.25·standalone
      + 0.3  if 25 s ≤ duration ≤ 45 s
      − 0.5 / 1.0 / 2.0  for the 2nd / 3rd / 4th+ best clip from the same episode
```

Top 30 by score (max 3 per episode) are sent as compact cards to **one** LLM call, which returns the final 15 in posting order with a one-line reason, optimising for variety of hook type in the first five. The numeric score is kept alongside for transparency. Note from the run: the extraction model's scores cluster at 4–5, so the numeric score mostly separates clips by duration and episode spread; the final order is effectively the LLM re-rank.

### 3.5 Audit (`src/audit.py`, `prompts/audit.md`)

Each final clip is sent back to the model with **only the beats inside the clip** — no logline, no episode, speakers anonymised to letters — and asked, as "a stranger scrolling TikTok": what is happening, who are these people, what is unresolved, and 1–5 how well could you follow this. The table in `ranked.md` shows the audit's `followable` score next to the extraction model's own `standalone` score. It is the closest free proxy for "makes sense on its own".

### 3.6 The LLM wrapper (`src/llm.py`)

- `complete(system, user, *, json_mode=True, max_tokens=None) -> str`, OpenAI client, 60 s timeout.
- Retries: 5 attempts, backoff 2/4/8/16/32 s on 429 / 5xx / timeout.
- Fallback model (optional): tried on a 400/404 from the primary, or after its retries are exhausted. The log records which model answered.
- Self-healing: a 400 naming `reasoning_effort`, `temperature` or `seed` → retried once without that parameter; a 413 "too large" mentioning tokens → retried once with `max_tokens` halved (floor 1200).
- `temperature=0` and `seed=42` on every call (providers do not all honour the seed).
- Every attempt — success or failure — is appended to `outputs/logs/calls.jsonl` with timestamp, requested model, **server-reported model**, prompt hash, token usage, latency, raw response or error body.
- `DRY_RUN=1` writes the prompt to disk and returns a stub, no network.

---

## 4. Configuration reference

All settings are environment variables, read from `.env` (CLI) or the sidebar (web app). Real environment variables win over `.env`.

| variable | default | meaning |
|---|---|---|
| `LLM_BASE_URL` | — | OpenAI-compatible endpoint, e.g. `https://api.groq.com/openai/v1` |
| `LLM_API_KEY` | — | key for that endpoint |
| `LLM_MODEL` | — | model id exactly as the provider lists it |
| `LLM_FALLBACK_MODEL` | empty = off | second model tried when the primary fails |
| `LLM_FALLBACK_BASE_URL` / `_API_KEY` | primary's | only if the fallback lives on another endpoint |
| `LLM_REASONING_EFFORT` | `low` (CLI) / `auto` (app) | `low`/`medium`/`high` sent to reasoning models; `off` never sends; `auto` decides from the model name |
| `LLM_TEMPERATURE` | `0` | sampling temperature; `seed=42` is always sent |
| `LLM_MAX_TOKENS` | `3000` | completion budget per call |
| `LLM_SLEEP_S` | `4` | seconds between episode calls (use 20–30 on Groq free tier) |
| `AUDIT_SLEEP_S` | `5` | seconds between audit calls |
| `DRY_RUN` | `0` | `1` = no network, prompts written to `outputs/dry/` |

---

## 5. Outputs

| file | contents |
|---|---|
| `outputs/ranked.md` | the final 15 promos, human-readable, plus the cold-viewer audit table |
| `outputs/promos.csv` | one row per final promo: rank, episode, segments, duration, hook type, hook, cliffhanger, why, h/c/s scores, score |
| `outputs/promos.json` | same plus the top 30, the dropped candidates, weights and model metadata |
| `outputs/audit.json` | per-promo cold-viewer answers and followable score |
| `outputs/stats.md` | episodes, proposed/accepted/rejected by reason, score distribution, primary vs fallback split, calls, tokens, cost, wall time |
| `outputs/episodes/epNNN.json` | per episode: accepted and rejected candidates with reasons, declared vs actual duration, model that answered |
| `outputs/logs/calls.jsonl` | every LLM attempt |
| `outputs/logs/errors.log` | episodes that failed outright (the run continues past them) |
| `outputs/logs/durations.jsonl` | model-declared vs code-computed clip lengths |

Outputs quote at most the hook and cliffhanger line of each clip; console output never prints scene text.

---

## 6. Results on the input pack

Full run on the 50-episode pack, `gpt-oss-120b` on Groq (primary) with `gpt-oss-20b` fallback:

| | |
|---|---|
| Episodes processed | **50 / 50** — 36 answered by the primary, 14 by the fallback |
| Candidates | 136 proposed → **95 accepted** → top 30 → **15 final**, from 14 different episodes, mean 29 s |
| Rejections (41) | 21 duplicate · 12 mid-dialogue cut · 4 unknown beat id · 2 too short · 1 too long · 1 SFX opener |
| Episodes with nothing usable | 2 (ep 8: every id invalid; ep 45: every clip cut mid-line) — reported, not patched |
| Cold-viewer audit, top 15 | mean followable **3.9 / 5**, 0.9 below the extraction model's self-rated standalone score |
| Calls · tokens · cost | ~66 per show (214 with free-tier retries) · 272k in / 76k out · **≈ $0.27 (~₹23) per show** at $0.30 / $2.50 per M tokens |
| Wall time | ~30 min per show on the free tier, rate-limit bound; ~4 min of actual model time |

Full numbers: `outputs/stats.md` in the raw-outputs folder; narrative and failure analysis: `docs/SOLUTION.pdf`.

---

## 7. Known limitations and failure modes

Found by reading the top 15 and bottom 5 against the scripts, the rejection log, the audit, and a 3×3 repeat run. Most important first.

1. **Score inflation.** About half of all hook / cliffhanger / standalone scores are 5 and none is below 3. The numeric score therefore ranks weakly; the LLM re-rank decides — and it once placed a context-free reply ("We will not.") at #2. *Caught by:* human review only.
2. **Non-determinism.** The same episode and prompt produced 1, 4 and 4 accepted clips across three runs, even at `temperature=0` with a fixed seed (Groq does not honour the seed; reasoning models re-think each time). The id re-ask rescued two all-invalid responses; one run still failed on malformed JSON. *Conclusion:* the editor's review step stays.
3. **Duplicates.** The model's most common mistake: one ending proposed with several start points (21 of 41 rejections). A prompt rule did not stop it; the code dedupe does.
4. **Hook is a reaction, not the event.** e.g. opening on "Dear God." with the scar reveal three seconds before the clip. A prompt rule reduced it; it still recurs.
5. **Cut on the answer, not the question.** e.g. ending on "Lady Catherine." instead of the "Who did this?" before it. *Caught by:* human review only.
6. **Smaller models hallucinate more.** The 20b fallback produced most invalid ids and the large duration errors (declared 22 s for a 72 s clip). The validator catches all of these; it cannot catch a dull hook.
7. **Audit is optimistic.** Action lines still contain character names, "would tap next" was true 15/15, and scores shift ±1 between identical calls.
8. **Labels need review.** `hook_type` is sometimes wrong (visual moments tagged "reveal"); some hooks are paraphrases rather than the exact first line.
9. **Free-tier throughput.** Tokens-per-minute caps force ~30 s between episodes; Gemini's 20-requests/day cap cannot process one show. A paid tier removes both.
10. **Timing assumption.** Timestamps come from the screenplay and assume it matches the final cut.

The build log of 31 AI mistakes (model ids that no longer existed, a fallback that returned empty JSON, an over-strict validator rule, a confounded experiment) is in `notes/ai-mistakes.md` (not in the repo; summarised in the solution doc).

---

## 8. Troubleshooting

| symptom (run log / app) | cause | fix |
|---|---|---|
| `404` model not found | wrong model id for that endpoint | copy the id from the provider's model list (`client.models.list()`) |
| `401` | wrong key or key for a different provider | re-paste |
| `429` with "TPM" | per-minute token cap | raise `LLM_SLEEP_S` (20–30 on Groq) |
| `429` with "per day" / "RPD" | daily cap exhausted (Gemini free: 20/day) | wait for reset or switch provider |
| `413` | prompt + `max_tokens` over the per-minute budget | handled automatically once; else lower `LLM_MAX_TOKENS` to 2000 |
| `ExtractionError: no JSON object in response` / `json_validate_failed` with empty output | reasoning model spent the budget thinking | `LLM_REASONING_EFFORT=low`, or pick a non-reasoning model |
| `400` mentioning `reasoning_effort` / `temperature` / `seed` | provider rejects the parameter | handled automatically (dropped and retried, logged as `*_dropped`) |
| all candidates `unknown_id` | model used its own numbering | handled automatically (one re-ask); re-run if it persists |
| episode listed in `errors.log` | failed after all retries | re-run without `--force` — only missing episodes are retried |
| app button disabled | key, URL, model or episode missing | the app says which, above the button |

---

## 9. Project layout

```
app.py                  Streamlit web app (same pipeline as the CLI)
run.py                  CLI: stages 1-4, resumable, per-episode error isolation
src/
  parse.py              screenplay -> Episode/Beat models
  extract.py            per-episode LLM call, JSON parsing, validation, dedupe, id re-ask
  rank.py               scoring, per-episode cap, LLM re-rank, promos.{json,csv}, ranked.md
  audit.py              cold-viewer audit
  stats.py              stats.md from outputs and call log
  llm.py                the only place that talks to a model
  config.py             .env loading
  models.py             pydantic models (Episode, Beat, Candidate, ...)
prompts/
  extract.md            extraction rules + JSON schema
  extract_id_retry.md   note appended when every id was invalid
  rank.md               re-rank instructions
  audit.md              cold-viewer prompt
scripts/inspect.py      one-line-per-episode parse summary
tests/                  pytest, no network (parse, extract/validate, llm retries, rank, audit, run)
docs/                   solution doc (gitignored — see Confidentiality)
DEPLOY.md               Streamlit Community Cloud steps
.env.example            every setting, documented
```

---

## 10. Confidentiality, tests, AI use

**Confidentiality.** The input pack is Kuku's content. `data/` (screenplays), `docs/` (brief + solution doc) and `outputs/` (quotes hook/cliffhanger lines) are excluded from the repository via `.gitignore`. The web app keeps uploads in a per-session temp folder and never displays scene text.

**Tests.** `python -m pytest -q` — ~100 tests covering the parser against the real format, every validation rule, the retry / fallback / parameter-dropping logic in the wrapper (with a fake client), scoring and ranking, audit parsing, and the CLI stages. No test touches the network.

**AI use.** Planned and reviewed with Claude (chat), built with Claude Code, runtime model `gpt-oss-120b` on Groq. Every design decision in this README — beat IDs instead of timestamps, one call per episode, code-side validation and dedupe, the fallback, temperature 0, the audit — came out of reading real outputs against the scripts and fixing what was wrong, and each is defended in `docs/SOLUTION.pdf`.
