"""Standalone entry point for the insight chatbot.

Reads configuration from environment variables and starts the FastAPI app on
port 8000 by default.

Required env:
  MARINA_URL              -- base URL of the Marina deployment
  LLM_BASE_URL            -- OpenAI-compatible LLM endpoint (e.g. MindRouter)
  LLM_API_KEY             -- API key for the LLM endpoint
  LLM_MODEL               -- model identifier

Optional env:
  INSIGHT_CREDENTIALS_FILE  -- path to client_id + private key JSON
                               (default: /var/lib/insight/insight_credentials.json)
  AUTH_TOKEN_AUDIENCE       -- expected aud claim for /auth/token (default: marina)
  PORT                      -- listen port (default: 8000)
"""

import os

import uvicorn

from insight import build_app

app = build_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
