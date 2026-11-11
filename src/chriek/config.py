"""Environment-variable configuration. Every name and default is documented in README.md."""

import re
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

MIN_SECRET_LENGTH = 32
KEY_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")


class ConfigError(ValueError):
    """Invalid configuration. Messages never contain secret values."""


class AuthMode(StrEnum):
    API_KEY = "api-key"  # chriek checks named static keys; each key owns its jobs
    PROXY = "proxy"  # a trusted reverse proxy authenticates; all clients share one owner


@dataclass(frozen=True)
class Limits:
    max_images: int = 4
    max_dimension: int = 2048
    max_pixels: int = 1024 * 1024
    max_prompt_chars: int = 2000
    max_queued_jobs: int = 100  # per owner; running and finished jobs don't count


@dataclass(frozen=True)
class Tls:
    certfile: Path
    keyfile: Path


@dataclass(frozen=True)
class Settings:
    db_path: Path
    storage_dir: Path
    auth_mode: AuthMode
    api_keys: Mapping[str, str]  # owner name -> key; empty in proxy mode
    signing_secret: str
    url_ttl_seconds: int
    limits: Limits
    host: str
    port: int
    tls: Tls | None  # None serves plain HTTP

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        limits = Limits(
            max_images=_int(env, "CHRIEK_MAX_IMAGES", Limits.max_images, 1, 64),
            max_dimension=_int(env, "CHRIEK_MAX_DIMENSION", Limits.max_dimension, 16, 8192),
            max_pixels=_int(env, "CHRIEK_MAX_PIXELS", Limits.max_pixels, 256, 8192 * 8192),
            max_prompt_chars=_int(
                env, "CHRIEK_MAX_PROMPT_CHARS", Limits.max_prompt_chars, 1, 100_000
            ),
            max_queued_jobs=_int(env, "CHRIEK_MAX_QUEUED_JOBS", Limits.max_queued_jobs, 1, 100_000),
        )
        if limits.max_dimension % 16:
            raise ConfigError("CHRIEK_MAX_DIMENSION must be a multiple of 16")
        auth_mode = _auth_mode(env.get("CHRIEK_AUTH_MODE", AuthMode.API_KEY))
        api_keys = env.get("CHRIEK_API_KEYS", "")
        return cls(
            db_path=Path(env.get("CHRIEK_DB_PATH", "chriek-data/chriek.db")),
            storage_dir=Path(env.get("CHRIEK_STORAGE_DIR", "chriek-data/images")),
            auth_mode=auth_mode,
            api_keys=(
                _no_api_keys(api_keys) if auth_mode is AuthMode.PROXY else _api_keys(api_keys)
            ),
            signing_secret=_signing_secret(env.get("CHRIEK_SIGNING_SECRET", "")),
            url_ttl_seconds=_int(env, "CHRIEK_URL_TTL_SECONDS", 900, 1, 7 * 24 * 3600),
            limits=limits,
            host=env.get("CHRIEK_HOST", "127.0.0.1"),
            port=_int(env, "CHRIEK_PORT", 8000, 1, 65535),
            tls=_tls(env),
        )


def _int(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer") from None
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}")
    return value


def _auth_mode(raw: str) -> AuthMode:
    try:
        return AuthMode(raw)
    except ValueError:
        raise ConfigError("CHRIEK_AUTH_MODE must be 'api-key' or 'proxy'") from None


def _no_api_keys(raw: str) -> dict[str, str]:
    # Proxy mode never checks keys; refusing them keeps operators from relying on them.
    if raw.strip():
        raise ConfigError("CHRIEK_API_KEYS must be unset or empty when CHRIEK_AUTH_MODE is 'proxy'")
    return {}


def _api_keys(raw: str) -> dict[str, str]:
    """Parse `name:key,name:key`. Names identify job owners; keys are secrets."""
    keys: dict[str, str] = {}
    for position, entry in enumerate(raw.split(","), start=1):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, key = entry.partition(":")
        name = name.strip()
        if not sep or not KEY_NAME_PATTERN.fullmatch(name):
            raise ConfigError(
                f"CHRIEK_API_KEYS entry {position} must be name:key with a name of "
                "1-64 characters from A-Z a-z 0-9 _ . -"
            )
        # Starlette decodes headers as Latin-1, and clients differ in how they encode
        # anything beyond ASCII: only printable ASCII keys reliably authenticate.
        if len(key) < MIN_SECRET_LENGTH or not all("!" <= c <= "~" for c in key):
            raise ConfigError(
                f"CHRIEK_API_KEYS key for {name!r} must be at least "
                f"{MIN_SECRET_LENGTH} printable ASCII characters without whitespace"
            )
        if name in keys:
            raise ConfigError(f"CHRIEK_API_KEYS names {name!r} more than once")
        if key in keys.values():
            raise ConfigError(f"CHRIEK_API_KEYS key for {name!r} is shared with another name")
        keys[name] = key
    if not keys:
        raise ConfigError("CHRIEK_API_KEYS must name at least one key")
    return keys


def _signing_secret(raw: str) -> str:
    if len(raw) < MIN_SECRET_LENGTH:
        raise ConfigError(f"CHRIEK_SIGNING_SECRET must be at least {MIN_SECRET_LENGTH} characters")
    return raw


def _tls(env: Mapping[str, str]) -> Tls | None:
    """Both files or neither. A variable that is set counts even when empty, so an unset
    shell variable expanded into it stops startup instead of silently serving plain HTTP."""
    certfile, keyfile = env.get("CHRIEK_TLS_CERTFILE"), env.get("CHRIEK_TLS_KEYFILE")
    if certfile is None and keyfile is None:
        return None
    if certfile is None or keyfile is None:
        raise ConfigError("CHRIEK_TLS_CERTFILE and CHRIEK_TLS_KEYFILE must be set together")
    if not certfile:
        raise ConfigError("CHRIEK_TLS_CERTFILE must not be empty")
    if not keyfile:
        raise ConfigError("CHRIEK_TLS_KEYFILE must not be empty")
    tls = Tls(certfile=Path(certfile), keyfile=Path(keyfile))
    try:
        # Loading the pair here turns a bad file into a configuration error before the
        # database or port opens. The callback rejects encrypted keys instead of prompting.
        ssl.create_default_context(ssl.Purpose.CLIENT_AUTH).load_cert_chain(
            tls.certfile, tls.keyfile, password=_reject_encrypted_key
        )
    except OSError as error:  # ssl.SSLError is an OSError
        raise ConfigError(
            "CHRIEK_TLS_CERTFILE and CHRIEK_TLS_KEYFILE must name a readable PEM certificate "
            f"and its unencrypted private key: {error}"
        ) from None
    return tls


def _reject_encrypted_key() -> str:
    raise ConfigError("CHRIEK_TLS_KEYFILE must be an unencrypted private key")
