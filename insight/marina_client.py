"""Thin wrappers over Marina's data path: /query/schema, /query, /files/*.

All wrappers take an explicit `headers` dict (typically built via
`insight.auth.auth_headers`) so this module has no auth state of its own.
"""

import asyncio
import os

import requests

MARINA_URL = os.environ.get("MARINA_URL", "http://marina:7010")


def fetch_schema(headers: dict) -> dict:
    """Return the parsed `/query/schema` body. Raises on non-200."""
    resp = requests.get(f"{MARINA_URL}/query/schema", headers=headers, timeout=30)
    if resp.status_code != 200:
        detail = ""
        try:
            detail = resp.json().get("error", "")
        except Exception:
            pass
        msg = f"Marina /query/schema returned {resp.status_code}"
        if detail:
            msg += f": {detail}"
        raise RuntimeError(msg)
    return resp.json()


def fetch_file_catalog(headers: dict) -> list[dict]:
    """Best-effort fetch. 403 (no allowed_file_tags) and errors return []."""
    try:
        resp = requests.get(
            f"{MARINA_URL}/files/catalog", headers=headers, timeout=10
        )
        if resp.status_code == 200:
            return resp.json().get("files", [])
    except Exception:
        pass
    return []


async def query(headers: dict, table: str, filters: dict | None, limit: int,
                offset: int | None = None,
                group_by: list | None = None,
                aggregate: list | None = None) -> dict:
    """POST /query for one table. Returns rows/columns/row_count, plus
    total_count when Marina provides it."""
    table_req: dict = {"table": table, "limit": limit}
    if filters:
        table_req["filters"] = filters
    if offset:
        table_req["offset"] = offset
    if group_by:
        table_req["group_by"] = group_by
    if aggregate:
        table_req["aggregate"] = aggregate
    resp = await asyncio.to_thread(
        requests.post,
        f"{MARINA_URL}/query",
        headers=headers,
        json={"tables": [table_req]},
        timeout=30,
    )
    if resp.status_code != 200:
        if resp.content:
            try:
                err = resp.json().get("error", f"Marina returned {resp.status_code}")
            except Exception:
                err = f"Marina returned {resp.status_code}"
        else:
            err = f"Marina returned {resp.status_code}"
        raise RuntimeError(err)
    data = resp.json().get(table, {})
    out = {
        "rows": data.get("rows", []),
        "columns": data.get("columns", []),
        "row_count": data.get("rowCount", 0),
    }
    if "totalCount" in data:
        out["total_count"] = data["totalCount"]
    return out


def fetch_file_raw(headers: dict, file_hash: str) -> tuple[bytes, str]:
    """Synchronous fetch of a file's raw bytes + content-type from Marina."""
    resp = requests.get(
        f"{MARINA_URL}/files",
        headers=headers,
        params={"hash": file_hash},
        timeout=30,
        stream=False,
    )
    resp.raise_for_status()
    return resp.content, resp.headers.get("content-type", "application/octet-stream")


async def fetch_file_text(headers: dict, file_hash: str) -> str:
    """GET /files for a single file_hash. Refuses non-text content types."""
    resp = await asyncio.to_thread(
        requests.get,
        f"{MARINA_URL}/files",
        headers=headers,
        params={"hash": file_hash},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Marina returned {resp.status_code}")
    ct = resp.headers.get("content-type", "")
    if not any(t in ct for t in ("text/", "application/json", "application/xml")):
        raise RuntimeError(f"File type not readable: {ct}")
    return resp.text
