# Promo finder

A CLI that reads a folder of microdrama screenplays (one `.txt` per episode) and
outputs a ranked list of promo clips: 20–60 s cuts that open on a hook in the
first 3 seconds, end on a cliffhanger, and make sense to a cold viewer who has
never seen the show. Each promo is given as episode + timestamp range(s), with
the hook line, cliffhanger line, scores and a one-line rationale, so an editor
can review the top 15 instead of watching every episode. Nothing is specific to
one show: point it at any folder of screenplays in the same format.

## Architecture

```
parse (code)    screenplay .txt -> numbered beats (e001-b042) with kind, start/end seconds
extract (LLM)   one call per episode: propose up to N promos as beat-ID ranges + 1–5 scores
validate (code) resolve IDs -> timestamps; reject unknown IDs, <20 s / >60 s, SFX openers, mid-line cuts, duplicates
rank (code+LLM) code scores + 25–45 s bonus + per-episode diversity cap -> top 30; one LLM call orders the final 15
audit (LLM)     each final clip's beats only, no show context -> "cold viewer" followable score
stats (code)    counts, reject reasons, tokens, cost, wall time from logs/calls.jsonl
```

The LLM only makes judgments (which moments, how strong). It never emits
timestamps; code owns parsing, IDs, timing, validation and ranking math. All
LLM traffic goes through `src/llm.py` (any OpenAI-compatible endpoint, 60 s
timeout, retry with backoff on 429/5xx, optional fallback model, every call
logged). Prompts live in `prompts/*.md`.

## Setup

Python 3.10+ (3.12 recommended).

```bash
# recommended
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# fallback without uv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # then fill in LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
```

## Run

```bash
python run.py --input "data/<show folder>" --out outputs
```

- `--episodes 1,10,25` runs a subset; `--force` re-extracts episodes that already have output.
- `--stages 3` re-ranks existing extractions without re-calling extraction; `--stages 4` re-runs the audit.
- `--input` can be omitted when `data/` contains exactly one show folder.
- Free-tier rate limits: set `LLM_SLEEP_S` (seconds between episodes, default 4).

**No API key? Dry run works:**

```bash
DRY_RUN=1 python run.py --episodes 1
```

This parses, builds every prompt and writes it to `outputs/logs/dry/` without
any network call, then runs the remaining stages on the empty results. Dry-run
outputs go to `outputs/dry/` so they never overwrite a real run.

Tests: `python -m pytest -q` (no network).

## Swapping the model

Any OpenAI-compatible endpoint works; change three env vars, no code changes.
Model IDs change often, so check your provider's current list.

```bash
# Gemini
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=<gemini key>
LLM_MODEL=gemini-3.8-flash

# Groq
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=<groq key>
LLM_MODEL=openai/gpt-oss-120b

# Anthropic (OpenAI-compatible endpoint)
LLM_BASE_URL=https://api.anthropic.com/v1/
LLM_API_KEY=<anthropic key>
LLM_MODEL=claude-sonnet-5-5

# OpenAI
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=<openai key>
LLM_MODEL=gpt-4o-mini
```

Optional: `LLM_FALLBACK_MODEL` (+ `_BASE_URL`, `_API_KEY`) is tried when the
primary 400/404s or exhausts its retries. `LLM_REASONING_EFFORT` (default `low`,
`off` to disable) and `LLM_MAX_TOKENS` (default 3000) are documented in
`.env.example`.

## Output files

| file | contents |
|---|---|
| `outputs/ranked.md` | human-readable final 15 promos, plus the cold-viewer audit table |
| `outputs/promos.json` | final promos with segments, timestamps, scores, rationale |
| `outputs/promos.csv` | same, one row per promo, for spreadsheets |
| `outputs/audit.json` | per-promo cold-viewer answers and followable score |
| `outputs/stats.md` | episodes, proposed/accepted/rejected by reason, score distribution, calls, tokens, cost, time |
| `outputs/episodes/epNNN.json` | per-episode accepted and rejected candidates (with reject reason) |
| `outputs/logs/calls.jsonl` | every LLM call: ts, model, served model, prompt hash, usage, raw response or error |
| `outputs/logs/errors.log` | episodes that failed; the run continues past them |
| `outputs/logs/durations.jsonl` | LLM-declared vs code-computed clip durations |
| `outputs/logs/dry/` | prompts written during `DRY_RUN=1` |

## Known limitations

From the full 50-episode run (gpt-oss-120b primary, gpt-oss-20b fallback):

- **Duplicate proposals.** 20 of 136 proposals were duplicates of the same
  ending (several start points, one cut); the validator drops them, but they
  waste the per-episode budget.
- **Mid-dialogue cuts.** 11 proposals started or ended in the middle of a
  spoken line despite an explicit prompt rule; rejected by code.
- **Hallucinated beat IDs.** 4 proposals cited beat IDs that don't exist in the
  episode; rejected by code.
- **Weak numeric ranking.** Self-scores cluster at 4–5 (about half of all
  accepted candidates score 5 on hook / cliffhanger), so score-based ordering
  separates little; the final order leans on the ranking call.
- **Silent episodes.** Episodes 8 and 45 produced no accepted promo (ep 8: all
  candidates had hallucinated IDs; ep 45: all cut mid-line), although the model's
  own notes called both strong promo material. There is no retry for them.
- **Throughput on free tiers.** Free-tier tokens-per-minute limits forced ~30 s
  between episodes (`LLM_SLEEP_S=30`), about 30 minutes per 50-episode show.
  Groq also returns intermittent 413 "request too large" on long episodes;
  `llm.py` retries once with a smaller `max_tokens`.
- **Labels and hooks need review.** `hook_type` is sometimes wrong (visual
  moments labelled "reveal" or "threat"), and some hooks are paraphrases rather
  than the exact first line. Not code-checkable.
- **Audit is optimistic.** Action lines still contain character names, so the
  "cold viewer" learns names a real viewer wouldn't; `would_tap_next` was true
  for 15/15, and followable scores shift by ±1 between identical calls.
- **Timing assumption.** Timestamps come from the screenplay and are
  episode-relative; they assume the final cut matches the script.

## Data

`data/` (screenplays) and `docs/` are confidential and excluded from the
repository via `.gitignore`, as is `outputs/` (it quotes hook and cliffhanger
lines). Put your own show under `data/<show>/` to run. Console output never
prints scene text; outputs quote at most the hook and cliffhanger line per promo.
