"""HTTP routes for the insight chatbot, registered on a FastAPI sub-app.

When mounted at `/insight` from shipyard, these route paths become
`/insight`, `/insight/llm-status`, etc. When this package is lifted into
its own repo, the same routes serve at the application root.
"""

import asyncio
import json
import logging

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates

from . import auth as _auth
from . import credentials as _credentials
from . import llm as _llm
from . import marina_client as _marina

logger = logging.getLogger(__name__)


def register(app: FastAPI, templates: Jinja2Templates) -> None:
    """Attach insight routes to a FastAPI app."""

    @app.get("/health", summary="Liveness probe")
    async def insight_health():
        return JSONResponse({"status": "ok", "service": "insight"})

    @app.get("/", response_class=HTMLResponse, summary="Insight -- AI-powered data exploration")
    async def insight_page(request: Request):
        saved_client_id, saved_key = _credentials.read()
        return templates.TemplateResponse("insight.html", {
            "request": request,
            "saved_client_id": saved_client_id,
            "is_configured": bool(saved_client_id and saved_key),
        })

    @app.get("/llm-status", summary="Test LLM connectivity")
    async def insight_llm_status():
        ok, message = _llm.test_connection()
        return JSONResponse({"ok": ok, "message": message})

    @app.get("/tables", summary="List tables and files visible to the configured client")
    async def insight_tables():
        client_id, private_key_pem = _credentials.read()
        if not client_id or not private_key_pem:
            return JSONResponse({"error": "Not configured."}, status_code=400)
        try:
            headers = _auth.auth_headers(client_id, private_key_pem)
            schema_info = _marina.fetch_schema(headers).get("tables", [])
            file_catalog = _marina.fetch_file_catalog(headers)
            return JSONResponse({"tables": schema_info, "files": file_catalog})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/config", summary="Save client_id and private key")
    async def insight_config(
        client_id: str = Form(...),
        private_key: str = Form(...),
    ):
        _credentials.write(client_id.strip(), private_key.strip())
        return JSONResponse({"ok": True})

    @app.get("/preview/{file_hash}", summary="Proxy a file from Marina for preview rendering")
    async def insight_preview(file_hash: str):
        client_id, private_key_pem = _credentials.read()
        if not client_id or not private_key_pem:
            return JSONResponse({"error": "Not configured."}, status_code=400)
        try:
            headers = _auth.auth_headers(client_id, private_key_pem)
            content, content_type = _marina.fetch_file_raw(headers, file_hash)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=502)
        return Response(content=content, media_type=content_type)

    @app.post("/chat", summary="Ask the insight chatbot a question")
    async def insight_chat(
        question: str = Form(...),
        schema: str = Form(None),
        files: str = Form(None),
        history: str = Form(None),
    ):
        client_id, private_key_pem = _credentials.read()
        if not client_id or not private_key_pem:
            logger.warning("insight/chat 400: insight credentials not configured")
            return JSONResponse({"error": "Insight not configured."}, status_code=400)
        if not _llm.is_configured():
            logger.warning("insight/chat 400: LLM not configured")
            return JSONResponse({"error": "LLM not configured."}, status_code=400)

        cached_schema = _maybe_parse_list(schema)
        cached_file_catalog = _maybe_parse_list(files) or []
        conversation_history = _parse_history(history)

        try:
            headers = _auth.auth_headers(client_id, private_key_pem)
        except Exception as e:
            logger.warning("insight/chat 502: token exchange failed: %s", e)
            return JSONResponse({"error": f"Marina auth failed: {e}"}, status_code=502)

        return StreamingResponse(
            _stream(question, headers, cached_schema, cached_file_catalog, conversation_history),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(event_type: str, **kwargs) -> str:
    return f"data: {json.dumps({'type': event_type, **kwargs})}\n\n"


def _maybe_parse_list(raw: str | None) -> list | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return None


def _parse_history(raw: str | None) -> list[dict]:
    parsed = _maybe_parse_list(raw)
    if not parsed:
        return []
    return [
        {"role": m["role"], "text": m["text"]}
        for m in parsed
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and m.get("text")
    ]


async def _stream(question, headers, cached_schema, cached_file_catalog, history):
    if cached_schema is not None:
        schema_info = cached_schema
    else:
        yield _sse("status", text="Loading schema...")
        try:
            schema_info = _marina.fetch_schema(headers).get("tables", [])
        except Exception as e:
            yield _sse("error", text=str(e))
            return
        if not schema_info:
            yield _sse("error", text="No tables configured. Edit the insight querying stream to add allowed tables.")
            return

    if cached_file_catalog:
        file_catalog = cached_file_catalog
    else:
        file_catalog = _marina.fetch_file_catalog(headers)

    async def dispatch_query(table, filters, limit, offset=None, group_by=None, aggregate=None):
        return await _marina.query(
            headers, table, filters, limit,
            offset=offset, group_by=group_by, aggregate=aggregate,
        )

    async def dispatch_file(file_hash):
        return await _marina.fetch_file_text(headers, file_hash)

    try:
        agent_iter = _llm.insight_agent(
            question, schema_info, file_catalog, history or None,
            dispatch_query, dispatch_file,
        ).__aiter__()
        pending_next: asyncio.Task | None = None
        while True:
            if pending_next is None:
                pending_next = asyncio.ensure_future(agent_iter.__anext__())
            done, _pending = await asyncio.wait({pending_next}, timeout=3.0)
            if not done:
                yield _sse("heartbeat")
                continue
            try:
                event_type, data = pending_next.result()
            except StopAsyncIteration:
                break
            finally:
                pending_next = None
            if event_type == "status":
                yield _sse("status", text=data)
            elif event_type == "token":
                yield _sse("token", text=data)
            elif event_type == "plot":
                yield _sse("plot", **data)
            elif event_type == "preview":
                yield _sse("preview", **data)
            elif event_type == "done":
                yield _sse("done", results=data)
            elif event_type == "error":
                yield _sse("error", text=data)
    except Exception as e:
        yield _sse("error", text=f"Agent error: {e}")
