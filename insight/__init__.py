"""Insight chatbot package.

Self-contained: depends only on env vars (MARINA_URL, LLM_BASE_URL,
LLM_API_KEY, LLM_MODEL, INSIGHT_CREDENTIALS_FILE) and stdlib + a small
set of third-party packages (fastapi, jinja2, requests, pyjwt, openai,
cryptography). No imports from shipyard.

Mount as a sub-app:

    from insight import build_app
    app.mount("/insight", build_app())

Or run standalone:

    import uvicorn
    from insight import build_app
    uvicorn.run(build_app(), host="0.0.0.0", port=8000)
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .routes import register

__all__ = ["build_app"]

_PACKAGE_DIR = Path(__file__).resolve().parent


def build_app() -> FastAPI:
    """Construct a FastAPI app with insight routes, templates, and (optional) static."""
    app = FastAPI(title="Insight", docs_url="/docs", redoc_url=None)
    templates = Jinja2Templates(directory=str(_PACKAGE_DIR / "templates"))
    # Jinja2 3.1.5+ keys its LRUCache on values that can include unhashable
    # globals (dicts), which breaks template lookup. Disable the cache.
    templates.env.cache = None
    static_dir = _PACKAGE_DIR / "static"
    if static_dir.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(static_dir)),
            name="insight-static",
        )
    register(app, templates)
    return app
