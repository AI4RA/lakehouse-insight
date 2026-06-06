"""LLM glue for the insight chatbot: tool definitions, system prompt, and the
agentic streaming loop.

Self-contained on purpose so the insight package can be lifted into its own
repo without dragging shipyard's broader llm_client along.
"""

import json
import logging
import os

import httpx
from openai import AsyncOpenAI, OpenAI

# Errors that justify a one-shot retry of an LLM streaming call. Each
# means "the upstream HTTP/1.1 chunked stream was cut" rather than
# "the model returned a bad answer."
_RETRYABLE_STREAM_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.ConnectError,
    httpx.ConnectTimeout,
)

logger = logging.getLogger(__name__)

from . import llm_config as _llm_config


# ---------------------------------------------------------------------------
# Configuration
#
# Resolution order: file (set via Insight UI) wins, env var falls through.
# ---------------------------------------------------------------------------

def _base_url() -> str:
    cfg = _llm_config.read()
    return (cfg.get("base_url", "").strip()
            or os.environ.get("LLM_BASE_URL", "").strip())


def _api_key() -> str:
    cfg = _llm_config.read()
    return (cfg.get("api_key", "").strip()
            or os.environ.get("LLM_API_KEY", "").strip()
            or "not-required")


def get_model() -> str:
    cfg = _llm_config.read()
    return (cfg.get("model", "").strip()
            or os.environ.get("LLM_MODEL", "").strip())


def vision_base_url() -> str:
    cfg = _llm_config.read()
    return (cfg.get("vision_base_url", "").strip()
            or os.environ.get("LLM_VISION_BASE_URL", "").strip()
            or _base_url())


def vision_api_key() -> str:
    cfg = _llm_config.read()
    return (cfg.get("vision_api_key", "").strip()
            or os.environ.get("LLM_VISION_API_KEY", "").strip()
            or _api_key())


def get_vision_model() -> str:
    cfg = _llm_config.read()
    return (cfg.get("vision_model", "").strip()
            or os.environ.get("LLM_VISION_MODEL", "").strip())


def is_configured() -> bool:
    return bool(_base_url() and get_model())


def test_connection() -> tuple[bool, str]:
    """Probe the configured LLM endpoint. Used by the /llm-status route."""
    if not _base_url():
        return False, "LLM_BASE_URL not configured"
    model = get_model()
    if not model:
        return False, "LLM_MODEL not configured"
    try:
        client = OpenAI(base_url=_base_url(), api_key=_api_key())
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the word 'ok' and nothing else."}],
            max_tokens=10,
        )
        text = (resp.choices[0].message.content or "").strip().lower()
        return True, f"Connected. Model responded: {text or '(empty)'}"
    except Exception as e:
        return False, f"Connection failed: {e}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_INSIGHT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_table",
            "description": (
                "Fetch rows from a table, or fetch aggregated results using group_by. "
                "Use group_by + aggregate for distribution/count/summary questions -- this is "
                "more accurate than fetching raw rows and counting client-side. "
                "For raw row fetches, use a high limit (e.g. 10000) to get all rows when needed. "
                "Returns column names, a sample of rows, and total row count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name"},
                    "filters": {
                        "type": "object",
                        "description": (
                            "Column filters. Simple equality: {\"col\": \"value\"}. "
                            "Operator form: {\"col\": {\"op\": \"value\"}} where op is one of: "
                            "eq, neq, gt, gte, lt, lte, like, ilike, in, is_null. "
                            "For 'in', value is a comma-separated string. "
                            "For 'is_null', value is 'true' or 'false'."
                        ),
                        "additionalProperties": {},
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to fetch. Use a high value (e.g. 10000) to retrieve all rows.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of rows to skip for pagination. Use with limit to page through results.",
                    },
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to GROUP BY for aggregate queries (e.g. ['account_type'])",
                    },
                    "aggregate": {
                        "type": "array",
                        "description": "Aggregate expressions to compute alongside group_by columns",
                        "items": {
                            "type": "object",
                            "properties": {
                                "fn": {
                                    "type": "string",
                                    "enum": ["COUNT", "SUM", "AVG", "MIN", "MAX"],
                                    "description": "Aggregate function",
                                },
                                "column": {
                                    "type": "string",
                                    "description": "Column to aggregate, or '*' for COUNT(*)",
                                },
                                "alias": {
                                    "type": "string",
                                    "description": "Result column name (e.g. 'cnt', 'total')",
                                },
                            },
                            "required": ["fn", "column", "alias"],
                        },
                    },
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_plot",
            "description": (
                "Render a Plotly chart from the data returned by the most recent "
                "query_table call. Call this after query_table when the user wants a chart or plot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "scatter", "pie", "histogram", "box", "violin"],
                        "description": "Chart type",
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Column for x-axis (or category labels for pie charts)",
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Column for y-axis (or numeric values for pie charts)",
                    },
                    "title": {"type": "string", "description": "Chart title"},
                    "x_label": {"type": "string", "description": "X-axis label (optional)"},
                    "y_label": {"type": "string", "description": "Y-axis label (optional)"},
                    "color_column": {
                        "type": "string",
                        "description": "Column for color grouping to produce multi-series charts (optional)",
                    },
                    "trendline": {
                        "type": "boolean",
                        "description": "Add a linear trendline to scatter plots (optional, scatter only)",
                    },
                },
                "required": ["chart_type", "x_column", "y_column", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read the text content of a document from the file catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_hash": {
                        "type": "string",
                        "description": "SHA-256 hash of the file",
                    },
                },
                "required": ["file_hash"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preview_file",
            "description": (
                "Open a file from the file catalog in a preview modal for the user to view. "
                "Use this when the user asks to open, show, display, or preview a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_hash": {
                        "type": "string",
                        "description": "SHA-256 hash of the file to preview",
                    },
                },
                "required": ["file_hash"],
            },
        },
    },
]


_SQL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Run a read-only Trino SQL statement against the lakehouse. Use this for "
                "every data lookup: simple SELECTs, joins, GROUP BY aggregates, window "
                "functions, CTEs -- whatever the question needs. Table references must be "
                "schema-qualified (see the system prompt for the exact schema). "
                "Returns column names, a sample of rows, and total row count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "The SQL statement to execute. Read-only -- INSERT/UPDATE/DELETE/"
                            "DROP/ALTER are rejected by Marina."
                        ),
                    },
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_plot",
            "description": (
                "Render a Plotly chart from the data returned by the most recent run_sql "
                "call. Call this after run_sql when the user wants a chart or plot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "scatter", "pie", "histogram", "box", "violin"],
                        "description": "Chart type",
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Column for x-axis (or category labels for pie charts)",
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Column for y-axis (or numeric values for pie charts)",
                    },
                    "title": {"type": "string", "description": "Chart title"},
                    "x_label": {"type": "string", "description": "X-axis label (optional)"},
                    "y_label": {"type": "string", "description": "Y-axis label (optional)"},
                    "color_column": {
                        "type": "string",
                        "description": "Column for color grouping (multi-series charts, optional)",
                    },
                    "trendline": {
                        "type": "boolean",
                        "description": "Add a linear trendline (optional, scatter only)",
                    },
                },
                "required": ["chart_type", "x_column", "y_column", "title"],
            },
        },
    },
]


def _build_agent_system(schema_info: list[dict], file_catalog: list[dict],
                         transport: str = "rest", client_id: str = "") -> str:
    lines = ["You are a data assistant. Answer questions using the tools available to you.\n"]
    sql_mode = transport == "sql"
    schema_qual = f'lakehouse."client_{client_id}"' if sql_mode else ""
    if sql_mode:
        lines.append(
            f"All data lives in the Trino schema `{schema_qual}`. "
            f"Reference tables schema-qualified, e.g. `{schema_qual}.\"table_name\"`. "
            f"You are talking to Marina's SQL gateway -- standard Trino SQL works.\n"
        )
    if schema_info:
        lines.append("Available tables:")
        for t in schema_info:
            desc = t.get("description", "")
            col_parts = []
            for c in t.get("columns", []):
                col_str = f"{c['column_name']} ({c['data_type']})"
                if c.get("description"):
                    col_str += f" \u2014 {c['description']}"
                col_parts.append(col_str)
            desc_str = f" \u2014 {desc}" if desc else ""
            name = f'{schema_qual}."{t["name"]}"' if sql_mode else t["name"]
            lines.append(f"  {name}{desc_str}: {', '.join(col_parts)}")
    if file_catalog and not sql_mode:
        lines.append("\nAvailable documents:")
        for f in file_catalog:
            desc = f.get("description", "")
            desc_str = f" \u2014 {desc}" if desc else ""
            lines.append(f"  {f['file_hash']}: {f['filename']}{desc_str}")
    if sql_mode:
        lines.append(
            "\nGuidelines:\n"
            "- IMPORTANT: Only answer questions using the tables listed above. Do not use "
            "general knowledge. If a question cannot be answered from the available data, "
            "say so clearly.\n"
            "- Use run_sql for every data lookup. Write standard Trino SQL.\n"
            "- Always schema-qualify table references with the schema shown above.\n"
            "- For chart or plot requests: call run_sql to fetch the data, then create_plot, "
            "then briefly explain what the chart shows. Use trendline=true on scatter plots "
            "when the user asks for a trendline or regression.\n"
            "- For data questions: call run_sql, then answer in prose.\n"
            "- For questions about what data is available: answer directly from the table list "
            "above without calling tools.\n"
            "- Read-only: INSERT/UPDATE/DELETE/DROP/ALTER are rejected by Marina with HTTP 403. "
            "Don't attempt them.\n"
            "- Keep prose answers concise."
        )
    else:
        lines.append(
            "\nGuidelines:\n"
            "- IMPORTANT: Only answer questions using the tables, documents, and data listed above. "
            "Do not use general knowledge. If a question cannot be answered from the available data "
            "or documents, say so clearly and suggest what data might be needed.\n"
            "- For distribution/count/summary questions (e.g. 'how many X by Y', 'breakdown of X'): "
            "use query_table with group_by and aggregate instead of fetching raw rows. "
            "Example: group_by=['account_type'], aggregate=[{fn:'COUNT', column:'*', alias:'cnt'}].\n"
            "- For chart or plot requests: call query_table (with group_by/aggregate when appropriate), "
            "then create_plot, then briefly explain what the chart shows. "
            "Use trendline=true on scatter plots when the user asks for a trendline or regression.\n"
            "- Filters support operators: eq, neq, gt, gte, lt, lte, like, ilike, in, is_null. "
            "Use {\"col\": {\"gt\": \"100\"}} for comparisons, {\"col\": {\"like\": \"%pattern%\"}} for text search, "
            "{\"col\": {\"in\": \"a,b,c\"}} for set membership.\n"
            "- For large result sets, use offset with limit to paginate through data.\n"
            "- For questions requiring all rows: use a high limit (e.g. 10000) in query_table.\n"
            "- For data questions: call query_table, then answer in prose.\n"
            "- For questions about what data is available: answer directly from the table and document lists above without calling tools.\n"
            "- When the user asks to open, show, display, or preview a file: call preview_file with the file_hash. "
            "Briefly confirm which file you are opening.\n"
            "- When referring to documents in your response, use the filename, not the hash.\n"
            "- Keep prose answers concise."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def insight_agent(
    question: str,
    schema_info: list[dict],
    file_catalog: list[dict],
    history: list[dict] | None,
    dispatch_query=None,   # REST: async (table, filters, limit, offset, group_by, aggregate) -> dict
    dispatch_file=None,    # REST: async (file_hash: str) -> str
    dispatch_sql=None,     # SQL:  async (sql: str) -> dict (columns, rows, row_count)
    transport: str = "rest",
    client_id: str = "",
):
    """Agentic tool-calling loop for insight chat.

    Yields (event_type, data) tuples:
      ("status", str)   -- progress message shown in UI
      ("token",  str)   -- LLM text token(s) to stream
      ("plot",   dict)  -- Plotly spec + row data for frontend rendering
      ("preview", dict) -- file preview metadata for UI modal
      ("done",   list)  -- list of table_results on completion
      ("error",  str)   -- fatal error message

    `transport` selects the tool surface: "rest" exposes query_table +
    create_plot + read_document + preview_file; "sql" exposes run_sql +
    create_plot only (files aren't reachable without a stream).
    """
    if not is_configured() or not get_model():
        yield ("error", "LLM not configured")
        return

    client = AsyncOpenAI(base_url=_base_url(), api_key=_api_key())
    model = get_model()

    sql_mode = transport == "sql"
    tools = _SQL_TOOLS if sql_mode else _INSIGHT_TOOLS

    messages: list[dict] = [
        {"role": "system",
         "content": _build_agent_system(schema_info, file_catalog,
                                          transport=transport, client_id=client_id)}
    ]
    if history:
        # Trim to last 10 messages (5 exchanges) to stay within context limits.
        for m in history[-10:]:
            messages.append({"role": m["role"], "content": m["text"]})
    messages.append({"role": "user", "content": question})

    last_query_result: dict | None = None
    table_results: list[dict] = []
    sql_query_counter = 0
    any_output = False

    for _ in range(8):  # cap iterations to prevent runaway loops
        content_chunks: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        finish_reason = None

        for attempt in range(2):  # one retry on transient stream cut
            content_chunks = []
            tool_calls_acc = {}
            finish_reason = None
            emitted_in_attempt = False

            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=4000,
                    stream=True,
                )
            except _RETRYABLE_STREAM_ERRORS as e:
                if attempt == 0:
                    logger.warning("LLM connect failed, retrying: %s", e)
                    yield ("status", "Reconnecting...")
                    continue
                yield ("error", f"Streaming error: {e}")
                return
            except Exception as e:
                yield ("error", str(e))
                return

            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    if delta.content:
                        content_chunks.append(delta.content)
                        # Speculatively stream content when no tool calls are accumulating.
                        if not tool_calls_acc:
                            any_output = True
                            emitted_in_attempt = True
                            yield ("token", delta.content)
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            i = tc_delta.index
                            if i not in tool_calls_acc:
                                tool_calls_acc[i] = {"id": "", "name": "", "arguments": ""}
                            if tc_delta.id:
                                tool_calls_acc[i]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_acc[i]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_acc[i]["arguments"] += tc_delta.function.arguments
                break  # success
            except _RETRYABLE_STREAM_ERRORS as e:
                # Retry only if we have not yielded anything yet this attempt.
                # Retrying after partial output would duplicate tokens or split a tool call.
                if (
                    attempt == 0
                    and not emitted_in_attempt
                    and not tool_calls_acc
                ):
                    logger.warning("LLM stream cut, retrying: %s", e)
                    yield ("status", "Reconnecting after stream interruption...")
                    continue
                yield ("error", f"Streaming error: {e}")
                return
            except Exception as e:
                yield ("error", f"Streaming error: {e}")
                return

        if finish_reason == "stop" or not tool_calls_acc:
            break

        full_content = "".join(content_chunks)
        messages.append({
            "role": "assistant",
            "content": full_content or "",
            "tool_calls": [
                {
                    "id": tool_calls_acc[i]["id"],
                    "type": "function",
                    "function": {
                        "name": tool_calls_acc[i]["name"],
                        "arguments": tool_calls_acc[i]["arguments"],
                    },
                }
                for i in sorted(tool_calls_acc)
            ],
        })

        for i in sorted(tool_calls_acc):
            tc = tool_calls_acc[i]
            name = tc["name"]
            try:
                args = json.loads(tc["arguments"] or "{}")
            except Exception:
                args = {}

            if name == "run_sql":
                sql = (args.get("sql") or "").strip()
                if not sql:
                    tool_content = json.dumps({"error": "sql argument was empty"})
                elif dispatch_sql is None:
                    tool_content = json.dumps({"error": "SQL transport not available in this session"})
                else:
                    sql_query_counter += 1
                    label = f"sql_{sql_query_counter}"
                    interaction = {
                        "kind": "sql",
                        "label": "run_sql",
                        "request": sql,
                    }
                    yield ("status", "Running SQL...")
                    try:
                        result = await dispatch_sql(sql)
                        last_query_result = result
                        table_results.append({
                            "table": label,
                            "interaction": interaction,
                            "rows": result.get("rows", [])[:20],
                            "columns": result.get("columns", []),
                            "row_count": result.get("row_count", 0),
                        })
                        sample = result.get("rows", [])[:20]
                        tool_content = json.dumps({
                            "columns": result.get("columns", []),
                            "rows": sample,
                            "row_count": result.get("row_count", 0),
                            "note": f"Showing {len(sample)} of {result.get('row_count', 0)} rows",
                        })
                    except Exception as e:
                        table_results.append({
                            "table": label,
                            "interaction": interaction,
                            "error": str(e),
                            "rows": [],
                            "columns": [],
                            "row_count": 0,
                        })
                        tool_content = json.dumps({"error": str(e)})

            elif name == "query_table":
                if dispatch_query is None:
                    tool_content = json.dumps({"error": "REST transport not available in this session"})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_content,
                    })
                    continue
                table = args.get("table", "")
                filters = args.get("filters") or {}
                limit = int(args.get("limit") or 500)
                offset = int(args.get("offset") or 0) or None
                group_by = args.get("group_by") or None
                aggregate = args.get("aggregate") or None
                interaction = {
                    "kind": "rest",
                    "label": "query_table",
                    "request": {
                        "table": table,
                        "filters": filters,
                        "limit": limit,
                        "offset": offset,
                        "group_by": group_by,
                        "aggregate": aggregate,
                    },
                }
                yield ("status", f"Querying {table}...")
                try:
                    result = await dispatch_query(
                        table, filters, limit,
                        offset=offset, group_by=group_by, aggregate=aggregate,
                    )
                    last_query_result = result
                    table_results = [r for r in table_results if r["table"] != table]
                    table_results.append({
                        "table": table,
                        "interaction": interaction,
                        "rows": result.get("rows", [])[:20],
                        "columns": result.get("columns", []),
                        "row_count": result.get("row_count", 0),
                    })
                    sample = result.get("rows", [])[:20]
                    tool_content = json.dumps({
                        "columns": result.get("columns", []),
                        "rows": sample,
                        "row_count": result.get("row_count", 0),
                        "note": f"Showing {len(sample)} of {result.get('row_count', 0)} rows",
                    })
                except Exception as e:
                    table_results = [r for r in table_results if r["table"] != table]
                    table_results.append({
                        "table": table,
                        "interaction": interaction,
                        "error": str(e),
                        "rows": [],
                        "columns": [],
                        "row_count": 0,
                    })
                    tool_content = json.dumps({"error": str(e)})

            elif name == "create_plot":
                if last_query_result is None:
                    prior_tool = "run_sql" if sql_mode else "query_table"
                    tool_content = json.dumps({"error": f"No data available. Call {prior_tool} first."})
                else:
                    yield ("plot", {
                        "chart_type": args.get("chart_type", "bar"),
                        "x_column": args.get("x_column", ""),
                        "y_column": args.get("y_column", ""),
                        "title": args.get("title", ""),
                        "x_label": args.get("x_label", ""),
                        "y_label": args.get("y_label", ""),
                        "color_column": args.get("color_column", ""),
                        "trendline": bool(args.get("trendline")),
                        "rows": last_query_result.get("rows", []),
                        "columns": last_query_result.get("columns", []),
                    })
                    tool_content = json.dumps({"status": "plot rendered"})

            elif name == "read_document":
                if dispatch_file is None:
                    tool_content = json.dumps({"error": "Document reading not available in SQL mode"})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_content,
                    })
                    continue
                file_hash = args.get("file_hash", "")
                match = next(
                    (f for f in file_catalog if f.get("file_hash") == file_hash),
                    None,
                )
                filename = match.get("filename", "") if match else ""
                interaction = {
                    "kind": "rest",
                    "label": "read_document",
                    "request": {"file_hash": file_hash, "filename": filename},
                }
                yield ("status", "Reading document...")
                try:
                    content = await dispatch_file(file_hash)
                    table_results.append({
                        "table": f"doc: {filename or file_hash[:8]}",
                        "interaction": interaction,
                        "rows": [],
                        "columns": [],
                        "row_count": 0,
                    })
                    tool_content = json.dumps({"content": content[:50000]})
                except Exception as e:
                    table_results.append({
                        "table": f"doc: {filename or file_hash[:8]}",
                        "interaction": interaction,
                        "error": str(e),
                        "rows": [],
                        "columns": [],
                        "row_count": 0,
                    })
                    tool_content = json.dumps({"error": str(e)})

            elif name == "preview_file":
                file_hash = args.get("file_hash", "")
                match = next(
                    (f for f in file_catalog if f.get("file_hash") == file_hash),
                    None,
                )
                filename = match.get("filename", "") if match else ""
                interaction = {
                    "kind": "rest",
                    "label": "preview_file",
                    "request": {"file_hash": file_hash, "filename": filename},
                }
                if match:
                    yield ("preview", {
                        "file_hash": file_hash,
                        "filename": filename,
                        "content_type": match.get("content_type", ""),
                    })
                    table_results.append({
                        "table": f"preview: {filename}",
                        "interaction": interaction,
                        "rows": [],
                        "columns": [],
                        "row_count": 0,
                    })
                    tool_content = json.dumps({"status": "preview opened"})
                else:
                    table_results.append({
                        "table": f"preview: {file_hash[:8]}",
                        "interaction": interaction,
                        "error": "File not found in catalog",
                        "rows": [],
                        "columns": [],
                        "row_count": 0,
                    })
                    tool_content = json.dumps({"error": "File not found in catalog"})

            else:
                tool_content = json.dumps({"error": f"Unknown tool: {name}"})

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_content,
            })

    if not any_output and not table_results:
        yield ("error", "No response from model. The conversation may be too long -- try starting a new chat.")
        return
    yield ("done", table_results)
