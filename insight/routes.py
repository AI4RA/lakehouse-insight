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
from . import llm_config as _llm_config
from . import marina_client as _marina
from . import marina_sql as _marina_sql

logger = logging.getLogger(__name__)


def register(app: FastAPI, templates: Jinja2Templates) -> None:
    """Attach insight routes to a FastAPI app."""

    @app.get("/health", summary="Liveness probe")
    async def insight_health():
        return JSONResponse({"status": "ok", "service": "insight"})

    @app.get("/", response_class=HTMLResponse, summary="Insight: AI-powered data exploration")
    async def insight_page(request: Request):
        saved_client_id, saved_key, saved_stream, saved_transport = _credentials.read()
        llm_cfg = _llm_config.read()
        # SQL transport scopes by client_id (no stream concept), so stream_name
        # is only required when transport is REST.
        if saved_transport == "sql":
            marina_configured = bool(saved_client_id and saved_key)
        else:
            marina_configured = bool(saved_client_id and saved_key and saved_stream)
        llm_configured = bool(
            llm_cfg.get("base_url", "").strip() and llm_cfg.get("model", "").strip()
        )
        return templates.TemplateResponse(
            request,
            "insight.html",
            {
                "saved_client_id": saved_client_id,
                "saved_stream": saved_stream,
                "saved_transport": saved_transport,
                "is_configured": marina_configured,
                "llm_configured": llm_configured,
                "llm_base_url": llm_cfg.get("base_url", ""),
                "llm_model": llm_cfg.get("model", ""),
                "llm_api_key": llm_cfg.get("api_key", ""),
                "llm_vision_base_url": llm_cfg.get("vision_base_url", ""),
                "llm_vision_model": llm_cfg.get("vision_model", ""),
                "llm_vision_api_key": llm_cfg.get("vision_api_key", ""),
            },
        )

    @app.post("/llm-config", summary="Save LLM endpoint config (partial)")
    async def insight_llm_save(
        base_url: str = Form(""),
        api_key: str = Form(""),
        model: str = Form(""),
        vision_base_url: str = Form(""),
        vision_api_key: str = Form(""),
        vision_model: str = Form(""),
    ):
        existing = _llm_config.read()
        merged_base = base_url.strip() or existing.get("base_url", "")
        merged_key = api_key.strip() or existing.get("api_key", "")
        merged_model = model.strip() or existing.get("model", "")
        merged_vbase = vision_base_url.strip() or existing.get("vision_base_url", "")
        merged_vkey = vision_api_key.strip() or existing.get("vision_api_key", "")
        merged_vmodel = vision_model.strip() or existing.get("vision_model", "")
        _llm_config.write(
            merged_base, merged_key, merged_model,
            merged_vbase, merged_vkey, merged_vmodel,
        )
        return JSONResponse({"ok": True})

    @app.get("/llm-status", summary="Test LLM connectivity")
    async def insight_llm_status():
        ok, message = _llm.test_connection()
        return JSONResponse({"ok": ok, "message": message})

    @app.get("/streams", summary="List querying streams visible to the saved credentials")
    async def insight_list_streams():
        client_id, private_key_pem, _, _ = _credentials.read()
        if not client_id or not private_key_pem:
            return JSONResponse({"error": "Not configured."}, status_code=400)
        try:
            headers = _auth.auth_headers(client_id, private_key_pem)
            return JSONResponse(_marina.fetch_streams(headers))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/streams/preview",
              summary="Preview streams using a client_id + private key passed in the form")
    async def insight_preview_streams(
        client_id: str = Form(""),
        private_key: str = Form(""),
    ):
        # Used when the settings UI wants to test what a freshly-typed
        # key can see *before* the user clicks Save. Falls back to
        # whatever is saved when a field is left blank, so partial
        # edits (just rotating the key, just changing the client_id)
        # still work.
        saved_id, saved_key, _, _ = _credentials.read()
        eff_id = (client_id or "").strip() or saved_id
        eff_key = (private_key or "").strip() or saved_key
        if not eff_id or not eff_key:
            return JSONResponse({"error": "client_id and private_key required"}, status_code=400)
        try:
            headers = _auth.auth_headers(eff_id, eff_key)
            return JSONResponse(_marina.fetch_streams(headers))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/tables", summary="List tables visible to the configured Marina client (transport-aware)")
    async def insight_tables():
        client_id, private_key_pem, stream_name, transport = _credentials.read()
        if transport == "sql":
            if not client_id or not private_key_pem:
                return JSONResponse({"error": "Not configured."}, status_code=400)
            try:
                sql_headers = _marina_sql.auth_headers_sql(client_id, private_key_pem)
                tables = await _marina_sql.discover_schema(sql_headers, client_id)
                # File catalog is REST-only; SQL clients have no equivalent.
                return JSONResponse({"tables": tables, "files": []})
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
        if not client_id or not private_key_pem or not stream_name:
            return JSONResponse({"error": "Not configured."}, status_code=400)
        try:
            headers = _auth.auth_headers(client_id, private_key_pem)
            schema_info = _marina.fetch_schema(headers, stream_name).get("tables", [])
            file_catalog = _marina.fetch_file_catalog(headers, stream_name)
            return JSONResponse({"tables": schema_info, "files": file_catalog})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/config", summary="Save client_id, private key, stream, and transport (partial)")
    async def insight_config(
        client_id: str = Form(""),
        private_key: str = Form(""),
        stream_name: str = Form(""),
        transport: str = Form(""),
    ):
        existing_id, existing_key, existing_stream, existing_transport = _credentials.read()
        merged_id = client_id.strip() or existing_id
        merged_key = private_key.strip() or existing_key
        merged_stream = stream_name.strip() or existing_stream
        merged_transport = transport.strip() if transport.strip() in ("rest", "sql") else existing_transport
        _credentials.write(merged_id, merged_key, merged_stream, merged_transport)
        return JSONResponse({"ok": True})

    @app.get("/preview/{file_hash}", summary="Proxy a file from Marina for preview rendering")
    async def insight_preview(file_hash: str):
        client_id, private_key_pem, stream_name, _ = _credentials.read()
        if not client_id or not private_key_pem or not stream_name:
            return JSONResponse({"error": "Not configured."}, status_code=400)
        try:
            headers = _auth.auth_headers(client_id, private_key_pem)
            content, content_type = _marina.fetch_file_raw(headers, stream_name, file_hash)
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
        client_id, private_key_pem, stream_name, transport = _credentials.read()
        if transport == "sql":
            if not client_id or not private_key_pem:
                logger.warning("insight/chat 400: SQL transport but no client_id/key")
                return JSONResponse({"error": "Insight not configured."}, status_code=400)
        else:
            if not client_id or not private_key_pem or not stream_name:
                logger.warning("insight/chat 400: insight credentials not configured")
                return JSONResponse({"error": "Insight not configured."}, status_code=400)
        if not _llm.is_configured():
            logger.warning("insight/chat 400: LLM not configured")
            return JSONResponse({"error": "LLM not configured."}, status_code=400)

        cached_schema = _maybe_parse_list(schema)
        cached_file_catalog = _maybe_parse_list(files) or []
        conversation_history = _parse_history(history)

        try:
            if transport == "sql":
                headers = _marina_sql.auth_headers_sql(client_id, private_key_pem)
            else:
                headers = _auth.auth_headers(client_id, private_key_pem)
        except Exception as e:
            logger.warning("insight/chat 502: token exchange failed: %s", e)
            return JSONResponse({"error": f"Marina auth failed: {e}"}, status_code=502)

        return StreamingResponse(
            _stream(question, headers, stream_name, transport, client_id,
                    cached_schema, cached_file_catalog, conversation_history),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )


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


async def _stream(question, headers, stream_name, transport, client_id,
                   cached_schema, cached_file_catalog, history):
    sql_mode = transport == "sql"

    if cached_schema is not None:
        schema_info = cached_schema
    else:
        yield _sse("status", text="Loading schema...")
        try:
            if sql_mode:
                schema_info = await _marina_sql.discover_schema(headers, client_id)
            else:
                schema_info = _marina.fetch_schema(headers, stream_name).get("tables", [])
        except Exception as e:
            yield _sse("error", text=str(e))
            return
        if not schema_info:
            if sql_mode:
                yield _sse("error", text=f"No tables visible to client_id '{client_id}'.")
            else:
                yield _sse("error", text=f"No tables configured on stream '{stream_name}'.")
            return

    if sql_mode:
        # SQL gateway has no file catalog; the read_document / preview_file
        # tools aren't in the SQL tool surface.
        file_catalog: list = []
    elif cached_file_catalog:
        file_catalog = cached_file_catalog
    else:
        file_catalog = _marina.fetch_file_catalog(headers, stream_name)

    if sql_mode:
        async def dispatch_sql(sql):
            return await _marina_sql.run_sql(headers, sql)
        agent_kwargs = {
            "dispatch_sql": dispatch_sql,
            "transport": "sql",
            "client_id": client_id,
        }
    else:
        async def dispatch_query(table, filters, limit, offset=None, group_by=None, aggregate=None):
            return await _marina.query(
                headers, stream_name, table, filters, limit,
                offset=offset, group_by=group_by, aggregate=aggregate,
            )
        async def dispatch_file(file_hash):
            return await _marina.fetch_file_text(headers, stream_name, file_hash)
        agent_kwargs = {
            "dispatch_query": dispatch_query,
            "dispatch_file": dispatch_file,
        }

    try:
        agent_iter = _llm.insight_agent(
            question, schema_info, file_catalog, history or None,
            **agent_kwargs,
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
            elif event_type == "thinking":
                yield _sse("thinking", text=data)
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
