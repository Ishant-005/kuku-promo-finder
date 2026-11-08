# Kuku promo finder

## Goal
CLI that reads microdrama screenplays (one .txt per episode under data/<show>/) and
outputs ranked promo candidates: 20–60s clips with a hook in the first 3s, a
cliffhanger ending, and standalone comprehensibility for a cold viewer who has
never seen the show.

## Hard rules
- data/ and docs/ are confidential and gitignored. Never print full scenes to
  stdout. Outputs may quote at most the hook line and cliffhanger line per promo.
- Nothing hardcoded to this show. Must run on any folder of screenplays in the
  same format.
- All LLM access goes through src/llm.py: openai package, base_url / api_key /
  model from env (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL). No provider SDKs anywhere.
- Code does parsing, candidate structure, validation, ranking math. LLM does
  judgment only.
- LLM proposes promos by beat ID, never by raw timestamp. Code resolves IDs to
  timestamps and rejects anything invalid.
- Python 3.10+. Deps: openai, python-dotenv, pydantic, pytest. Nothing else
  without asking.
- Prompts live in prompts/*.md, loaded at runtime. No prompt text in .py files.
- Every LLM call: explicit timeout (60s), retry with backoff on 429/5xx, logged to
  outputs/logs/calls.jsonl (ts, model, prompt_hash, usage, raw response).
- Tests in tests/ via pytest. Run them before telling me a step is done.

## Workflow
- Plan before writing files. Wait for my go-ahead.
- One step per task. Show me actual command output, not a summary of it.
- Whenever the model (you or the runtime LLM) gets something wrong, append a
  dated line to notes/ai-mistakes.md.
- Commit messages: short imperative. Never run git push.
- The step-by-step build plan with exact prompts is in docs/BUILD_PLAYBOOK.md.
