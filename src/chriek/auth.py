"""Request authentication: resolves every /jobs request to the name of a job owner.

`api-key` mode: `Authorization: Bearer <key>` resolves to the key's owner name.
`proxy` mode: a trusted reverse proxy has already authenticated the client; every request
resolves to one shared owner, whatever headers it carries.
"""

import hashlib
import hmac
from collections.abc import Callable, Mapping

from fastapi import HTTPException, Request, status

from chriek.config import AuthMode

# The ':' can't occur in an API key name, so proxy-mode jobs never share an owner with
# api-key-mode jobs in the same database.
PROXY_OWNER = "proxy:shared"

Authenticate = Callable[[Request], str]


def authenticator(auth_mode: AuthMode, api_keys: Mapping[str, str]) -> Authenticate:
    if auth_mode is AuthMode.PROXY:
        return _proxy_owner
    return ApiKeyAuth(api_keys)


def _proxy_owner(_request: Request) -> str:
    return PROXY_OWNER


def _digest(value: str) -> bytes:
    # Hashing first gives compare_digest equal-length inputs, so key length doesn't leak.
    return hashlib.sha256(value.encode()).digest()


class ApiKeyAuth:
    def __init__(self, api_keys: Mapping[str, str]) -> None:
        self._owners = [(name, _digest(key)) for name, key in api_keys.items()]

    def __call__(self, request: Request) -> str:
        """Return the owner name of the presented key; 401 on missing or invalid key."""
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
