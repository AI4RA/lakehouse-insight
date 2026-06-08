"""Marina's Trino-style SQL gateway via /sql/v1/statement.

Auth reuses the RFC 7523 client_credentials flow from `insight.auth`, but
Marina's SQL endpoint expects HTTP Basic with `base64(client_id:bearer)`
rather than `Authorization: Bearer <token>`. The SQL response model is
Trino-flavored: the initial POST returns a body with `nextUri` and we
GET-poll that URI (each step also returning a `nextUri`) until the query
finishes or a final body has no `nextUri`.
"""

import asyncio
import base64
import os
import time

import requests

from . import auth as _auth

MARINA_URL = os.environ.get("MARINA_URL", "http://marina:7010")
SQL_POLL_DEADLINE_SECS = 60.0


def auth_headers_sql(client_id: str, private_key_pem: str) -> dict:
    """Headers for Marina's SQL gateway: HTTP Basic with bearer-as-password."""
    token = _auth.get_bearer(client_id, private_key_pem)
    creds = f"{client_id}:{token}"
    encoded = base64.b64encode(creds.encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "text/plain",
    }


def schema_for(client_id: str, stream_name: str) -> str:
    """Schema-qualified prefix the LLM must use,
    e.g. `lakehouse."client_foo__personnel"`.

    Marina now scopes the SQL surface per `(client, stream)` -- a single
    statement may only reference one stream-schema, and cross-schema queries
    are rejected at the gateway with HTTP 403 (ui-insight/lakehouse#276).
    """
    return f'lakehouse."client_{client_id}__{stream_name}"'


async def run_sql(headers: dict, sql: str) -> dict:
    """Submit `sql`, poll nextUri until done, return rows + columns.

    Returns: {"columns": [str], "rows": [dict], "row_count": int}.
    Raises RuntimeError on non-200 responses, on a Trino `error` payload,
    or if polling doesn't terminate within SQL_POLL_DEADLINE_SECS.
    """
    url = f"{MARINA_URL}/sql/v1/statement"
    resp = await asyncio.to_thread(
        requests.post, url, data=sql.encode("utf-8"),
        headers=headers, timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Marina /sql/v1/statement returned {resp.status_code}: {resp.text[:300]}"
        )
    body = resp.json()

    columns = body.get("columns") or []
    rows: list[list] = list(body.get("data") or [])
    next_uri = body.get("nextUri")
    last_body: dict = body
    deadline = time.time() + SQL_POLL_DEADLINE_SECS

    while next_uri and time.time() < deadline:
        r = await asyncio.to_thread(
            requests.get, next_uri, headers=headers, timeout=15,
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"Marina SQL poll returned {r.status_code}: {r.text[:300]}"
            )
        poll_body = r.json()
        if poll_body.get("columns") and not columns:
            columns = poll_body["columns"]
        if poll_body.get("data"):
            rows.extend(poll_body["data"])
        last_body = poll_body
        next_uri = poll_body.get("nextUri")

    if next_uri:
        raise RuntimeError(
            f"SQL query did not finish within {SQL_POLL_DEADLINE_SECS:.0f}s"
        )

    err = last_body.get("error")
    if err:
        # Trino errors look like {message, errorName, errorType, ...}.
        msg = err.get("message") or err.get("errorName") or str(err)
        raise RuntimeError(f"SQL error: {msg}")

    col_names = [c.get("name", "") for c in columns]
    rows_dicts = [dict(zip(col_names, row)) for row in rows]
    return {
        "columns": col_names,
        "rows": rows_dicts,
        "row_count": len(rows_dicts),
    }


async def discover_schema(headers: dict, client_id: str, stream_name: str) -> list[dict]:
    """Build a schema listing equivalent to Marina's REST /query/schema,
    but populated via SHOW TABLES / DESCRIBE against the SQL endpoint.

    Scoped to the single `(client_id, stream_name)` schema -- post-#276 the
    SQL gateway no longer allows cross-stream queries, so each Insight
    session targets exactly one stream just like the REST path does.

    Falls back to the pre-#276 per-client schema `lakehouse."client_<id>"`
    when the per-stream schema doesn't exist yet (transitional Marinas).

    Returns: [{name, description, columns: [{column_name, data_type, description}]}].
    Failures to DESCRIBE a single table are skipped so one bad table doesn't
    nuke the whole listing.
    """
    schema = schema_for(client_id, stream_name)
    try:
        tables_result = await run_sql(headers, f"SHOW TABLES IN {schema}")
    except Exception:
        tables_result = None

    # Fall back to pre-#276 per-client schema if per-stream one is absent.
    fallback_schema = f'lakehouse."client_{client_id}"'
    if tables_result is None or not tables_result.get("rows"):
        try:
            tables_result = await run_sql(headers, f"SHOW TABLES IN {fallback_schema}")
            schema = fallback_schema
        except Exception:
            return []

    # SHOW TABLES returns rows with a single column (Trino calls it "Table").
    table_names: list[str] = []
    for r in tables_result["rows"]:
        if not r:
            continue
        # Take the first value regardless of column name casing.
        table_names.append(str(next(iter(r.values()))))

    schemas: list[dict] = []
    for tname in table_names:
        try:
            desc = await run_sql(headers, f'DESCRIBE {schema}."{tname}"')
        except Exception:
            continue
        cols: list[dict] = []
        for r in desc["rows"]:
            # DESCRIBE columns: Column, Type, Extra, Comment (Trino convention).
            col_name = r.get("Column") or r.get("column") or ""
            data_type = r.get("Type") or r.get("type") or ""
            comment = r.get("Comment") or r.get("comment") or ""
            if not col_name:
                continue
            cols.append({
                "column_name": col_name,
                "data_type": data_type,
                "description": comment,
            })
        schemas.append({
            "name": tname,
            "description": "",
            "columns": cols,
        })
    return schemas
