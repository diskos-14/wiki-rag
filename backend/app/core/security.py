"""API key check for the endpoints that cost money (they call Vertex AI) or hold user data.

Clients send `X-API-Key: <key>`. The key comes from the API_KEY setting (Secret Manager in the
deployment). /health and /docs stay open. With API_KEY unset or empty, the check is off: only
for local development, and a warning is logged.
"""

import logging
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.settings import get_settings

logger = logging.getLogger(__name__)

# auto_error=False: a missing header reaches our check, so the error message is ours.
# Using APIKeyHeader also gives Swagger UI an "Authorize" button.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_warned = False


def get_expected_api_key() -> str | None:
    """The configured key, or None when auth is off. A dependency, so tests can override it."""
    key = get_settings().api_key
    value = key.get_secret_value().strip() if key else ""
    return value or None


async def require_api_key(
    provided: Annotated[str | None, Security(api_key_header)],
    expected: Annotated[str | None, Depends(get_expected_api_key)],
) -> None:
    global _warned
    if expected is None:
        if not _warned:
            logger.warning("API_KEY is not set: endpoints are open (local development only)")
            _warned = True
        return
    # Constant-time comparison, so response timing doesn't leak how much of the key matched.
    if provided is None or not secrets.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Missing or invalid API key (X-API-Key header).",
            headers={"WWW-Authenticate": "APIKey"},
        )
