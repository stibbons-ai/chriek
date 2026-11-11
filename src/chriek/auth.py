"""API-key authentication: `Authorization: Bearer <key>` resolves to the key's owner name."""

import hashlib
import hmac
from collections.abc import Mapping

from fastapi import HTTPException, Request, status


def _digest(value: str) -> bytes:
    # Hashing first gives compare_digest equal-length inputs, so key length doesn't leak.
    return hashlib.sha256(value.encode()).digest()


class ApiKeyAuth:
    def __init__(self, api_keys: Mapping[str, str]) -> None:
        self._owners = [(name, _digest(key)) for name, key in api_keys.items()]

    def __call__(self, request: Request) -> str:
        """FastAPI dependency returning the owner name; 401 on missing or invalid key."""
        scheme, _, presented = request.headers.get("authorization", "").partition(" ")
        candidate = _digest(presented.strip())
        owner = None
        for name, digest in self._owners:  # no early exit: every key is compared
            if hmac.compare_digest(candidate, digest):
                owner = name
        if scheme.lower() != "bearer" or owner is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return owner
