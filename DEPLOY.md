# Deploying to Streamlit Community Cloud

The app reads credentials from `st.secrets` (bridged to env at startup). State lives in
Supabase; the filesystem is ephemeral. This guide covers the dedicated IAM user, the
secrets, and the deploy.

## 1. Create a dedicated, least-privilege IAM user (do NOT reuse `vscode-user`)

1. AWS Console → **IAM → Users → Create user** → name e.g. `gas-energy-copilot-streamlit`.
2. **Do not** give it console access. On permissions, choose **Attach policies directly →
   Create policy**, paste `iam/streamlit-bedrock-policy.json`, and **replace `ACCOUNT_ID`**
   with your account number (`414994224379`). This grants only `bedrock:InvokeModel`/
   `InvokeModelWithResponseStream` on the five models we use + `bedrock:Rerank` for Cohere.
3. Create the user, then **Security credentials → Create access key → Application running
   outside AWS**. Copy the **Access key ID** and **Secret access key** (shown once).

> If a query later returns AccessDenied, broaden the InvokeModel `Resource` to
> `arn:aws:bedrock:*::foundation-model/*` temporarily to confirm it's a scoping issue, then
> tighten back.

## 2. Push the repo to GitHub

The `v2` branch must be on a **public** GitHub repo (Streamlit Community Cloud requires it).
Confirm no secrets are committed: `git log -p | grep -iE 'AKIA|secret|password'` should be clean
(`.env` and `.streamlit/secrets.toml` are gitignored).

## 3. Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** → sign in with GitHub → **New app**.
2. Repo: your fork; **Branch: `v2`**; **Main file path: `app/streamlit_app.py`**.
3. **Advanced settings → Python version: 3.11** (or 3.12).
4. **Advanced settings → Secrets** — paste (TOML form, from `.streamlit/secrets.toml.example`):
   ```toml
   AWS_REGION = "us-east-1"
   AWS_ACCESS_KEY_ID = "AKIA...(the new IAM user's key)"
   AWS_SECRET_ACCESS_KEY = "...(the new IAM user's secret)"
   SUPABASE_DB_URL = "postgresql://postgres.<ref>:<pw>@aws-...pooler.supabase.com:6543/postgres"
   LANGSMITH_API_KEY = "lsv2_pt_..."      # optional
   LANGSMITH_TRACING = "true"             # optional
   LOG_JSON = "true"
   ```
   Do **not** set `AWS_PROFILE` here (that's local-only).
5. **Deploy**. First boot installs deps (~2-3 min) and cold-starts (~30-60 s).

## 4. Dependencies on Cloud

Streamlit Cloud installs from `requirements.txt` if present, else from `pyproject.toml`.
This repo's runtime deps (the `[project].dependencies`) are intentionally lean (no ragas/
ingestion heavyweights) to fit the 1 GB RAM ceiling. If you want to pin exactly, export a
runtime-only requirements file:

```bash
uv export --no-dev --no-emit-project --format requirements-txt > requirements.txt
```

(Commit it if you prefer reproducible Cloud builds; otherwise Streamlit resolves from pyproject.)

## 5. Warm-up before a live demo

```bash
uv run python scripts/warm_up_demo.py --url https://<your-app>.streamlit.app
```

Run ~5 min before the session to wake the app from cold-sleep and prime caches. Put the
resulting URL in the README's "Live demo" line.
