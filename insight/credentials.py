"""Insight credential storage. File-based, no shipyard coupling."""

import json
import os
from pathlib import Path

CREDENTIALS_FILE = os.environ.get(
    "INSIGHT_CREDENTIALS_FILE", "/var/lib/shipyard/insight_credentials.json"
)


def read() -> tuple[str, str]:
    """Return (client_id, private_key_pem). Either may be empty if not configured."""
    try:
        data = json.loads(Path(CREDENTIALS_FILE).read_text())
        return data.get("client_id", ""), data.get("private_key", "")
    except Exception:
        return "", ""


def write(client_id: str, private_key_pem: str) -> None:
    """Persist credentials to disk with mode 0600."""
    path = Path(CREDENTIALS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "client_id": client_id,
        "private_key": private_key_pem,
    }))
    path.chmod(0o600)
