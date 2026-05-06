"""OAuth client_credentials flow against Marina /auth/token.

Mints an RFC 7523 client_assertion JWT, exchanges it at /auth/token, caches
the returned bearer in-process until near expiry.
"""

import os
import secrets
import time

import jwt
import requests

MARINA_URL = os.environ.get("MARINA_URL", "http://marina:7010")
ASSERTION_AUDIENCE = os.environ.get("AUTH_TOKEN_AUDIENCE", "marina")

JWT_BEARER_ASSERTION_TYPE = (
    "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
)
REFRESH_LEEWAY_SECONDS = 30

_bearer_cache: dict = {"client_id": None, "token": None, "expires_at": 0.0}


def _create_assertion(client_id: str, private_key_pem: str) -> str:
    now = int(time.time())
    payload = {
        "iss": client_id,
        "sub": client_id,
        "aud": ASSERTION_AUDIENCE,
        "iat": now,
        "exp": now + 300,
        "jti": secrets.token_urlsafe(16),
    }
    return jwt.encode(payload, private_key_pem, algorithm="RS256")


def _exchange(client_id: str, private_key_pem: str) -> tuple[str, int]:
    assertion = _create_assertion(client_id, private_key_pem)
    resp = requests.post(
        f"{MARINA_URL}/auth/token",
        data={
            "grant_type": "client_credentials",
            "client_assertion_type": JWT_BEARER_ASSERTION_TYPE,
            "client_assertion": assertion,
        },
        timeout=15,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Marina /auth/token rejected insight credentials "
            f"(HTTP {resp.status_code}): {resp.text[:300]}"
        )
    body = resp.json()
    return body["access_token"], int(body.get("expires_in", 600))


def get_bearer(client_id: str, private_key_pem: str) -> str:
    """Return a usable bearer, exchanging for a new one when stale."""
    now = time.time()
    if (
        _bearer_cache["client_id"] == client_id
        and _bearer_cache["token"]
        and _bearer_cache["expires_at"] - now > REFRESH_LEEWAY_SECONDS
    ):
        return _bearer_cache["token"]
    token, expires_in = _exchange(client_id, private_key_pem)
    _bearer_cache.update({
        "client_id": client_id,
        "token": token,
        "expires_at": now + expires_in,
    })
    return token


def auth_headers(client_id: str, private_key_pem: str) -> dict:
    return {
        "Authorization": f"Bearer {get_bearer(client_id, private_key_pem)}",
        "Content-Type": "application/json",
    }
