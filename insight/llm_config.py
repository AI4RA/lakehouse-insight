"""LLM endpoint configuration. File-backed, env vars are the fallback."""

import json
import os
from pathlib import Path

CONFIG_FILE = os.environ.get(
    "INSIGHT_LLM_CONFIG_FILE", "/var/lib/insight/insight_llm.json"
)


def read() -> dict:
    """Return {base_url, api_key, model}. Empty dict if no file."""
    try:
        data = json.loads(Path(CONFIG_FILE).read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def write(base_url: str, api_key: str, model: str) -> None:
    """Persist LLM config to disk with mode 0600."""
    path = Path(CONFIG_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
    }))
    path.chmod(0o600)
