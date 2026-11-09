# Deploying the Streamlit app (free, ~10 minutes)

The app (`app.py`) runs the same pipeline as `run.py` on screenplays the reviewer
uploads, against any OpenAI-compatible endpoint entered in the sidebar. The input
pack is never bundled.

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## Streamlit Community Cloud

1. Push this repo to GitHub (public, or private with Streamlit granted access).
2. Go to https://share.streamlit.io -> "Create app" -> "Deploy a public app from GitHub".
3. Repository: <your repo>, Branch: main, Main file path: `app.py`. Python 3.12.
4. "Advanced settings" -> Secrets, paste (so reviewers don't need their own key):

       LLM_BASE_URL = "https://api.groq.com/openai/v1"
       LLM_API_KEY = "gsk_..."
       LLM_MODEL = "openai/gpt-oss-120b"
       LLM_FALLBACK_MODEL = "openai/gpt-oss-20b"
       LLM_REASONING_EFFORT = "low"
       LLM_MAX_TOKENS = "3000"

5. Deploy. First build takes 2-4 minutes. The URL is `https://<app-name>.streamlit.app`.
6. Test in an incognito window: upload 2-3 episode .txt files, click "Find promos".

Notes
- Secrets pre-fill the sidebar; a reviewer can overwrite them with their own
  endpoint/key/model to test a stronger model.
- Free-tier rate limits apply to whoever's key is used; the default 20 s pause
  between episodes keeps Groq's 8k tokens/min limit happy.
- Each browser session gets its own temp output folder; nothing is persisted on
  the server.
