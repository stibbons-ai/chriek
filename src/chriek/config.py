"""Environment-variable configuration. Every name and default is documented in README.md."""

import ipaddress
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

MIN_SECRET_LENGTH = 32
KEY_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")
# Host is a DNS name, an IPv4 address or a bracketed IPv6 literal; port is optional digits.
# The bracket class excludes IPvFuture and zone IDs; urlsplit rejects other invalid literals.
AUTHORITY_PATTERN = re.compile(
    r"(?:(?P<name>[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*)\.?|\[[0-9A-Fa-f:.]+\])"
    r"(?::(?P<port>[0-9]{1,5}))?"
)
# WHATWG clients parse a host whose last label is numeric as IPv4 (017.0.0.1 -> 15.0.0.1).
NUMERIC_LABEL = re.compile(r"[0-9]+|0[xX][0-9A-Fa-f]*")
PUBLIC_URL_FORM = (
    "CHRIEK_PUBLIC_URL must be an absolute http or https URL with a host, an optional port "
    "1-65535 and an optional path prefix"
)


class ConfigError(ValueError):
    """Invalid configuration. Messages never contain secret values."""


@dataclass(frozen=True)
class Limits:
    max_images: int = 4
    max_dimension: int = 2048
    max_pixels: int = 1024 * 1024
    max_prompt_chars: int = 2000


@dataclass(frozen=True)
class Settings:
    db_path: Path
    public_url: str  # scheme://authority[/prefix], no trailing slash
    storage_dir: Path
    api_keys: Mapping[str, str]  # owner name -> key
    signing_secret: str
    url_ttl_seconds: int
    limits: Limits
    host: str
    port: int

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        limits = Limits(
            max_images=_int(env, "CHRIEK_MAX_IMAGES", Limits.max_images, 1, 64),
            max_dimension=_int(env, "CHRIEK_MAX_DIMENSION", Limits.max_dimension, 16, 8192),
            max_pixels=_int(env, "CHRIEK_MAX_PIXELS", Limits.max_pixels, 256, 8192 * 8192),
            max_prompt_chars=_int(
                env, "CHRIEK_MAX_PROMPT_CHARS", Limits.max_prompt_chars, 1, 100_000
            ),
        )
        if limits.max_dimension % 16:
            raise ConfigError("CHRIEK_MAX_DIMENSION must be a multiple of 16")
        api_keys = _api_keys(env.get("CHRIEK_API_KEYS", ""))
        signing_secret = _signing_secret(env.get("CHRIEK_SIGNING_SECRET", ""))
        if signing_secret in api_keys or signing_secret in api_keys.values():
            raise ConfigError(
                "CHRIEK_SIGNING_SECRET must differ from every CHRIEK_API_KEYS name and key"
            )
        return cls(
            public_url=_public_url(env.get("CHRIEK_PUBLIC_URL", "")),
            db_path=Path(env.get("CHRIEK_DB_PATH", "chriek-data/chriek.db")),
            storage_dir=Path(env.get("CHRIEK_STORAGE_DIR", "chriek-data/images")),
            api_keys=api_keys,
            signing_secret=signing_secret,
            url_ttl_seconds=_int(env, "CHRIEK_URL_TTL_SECONDS", 900, 1, 7 * 24 * 3600),
            limits=limits,
            host=env.get("CHRIEK_HOST", "127.0.0.1"),
            port=_int(env, "CHRIEK_PORT", 8000, 1, 65535),
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
        # Errors name the entry position, never the name: a name may equal a secret.
        if len(key) < MIN_SECRET_LENGTH or not all("!" <= c <= "~" for c in key):
            raise ConfigError(
                f"CHRIEK_API_KEYS entry {position} key must be at least "
                f"{MIN_SECRET_LENGTH} printable ASCII characters without whitespace"
            )
        if name in keys:
            raise ConfigError(f"CHRIEK_API_KEYS entry {position} repeats an earlier name")
        if key in keys.values():
            raise ConfigError(f"CHRIEK_API_KEYS entry {position} key is shared with another name")
        keys[name] = key
    if not keys:
        raise ConfigError("CHRIEK_API_KEYS must name at least one key")
    return keys


def _signing_secret(raw: str) -> str:
    if len(raw) < MIN_SECRET_LENGTH:
        raise ConfigError(f"CHRIEK_SIGNING_SECRET must be at least {MIN_SECRET_LENGTH} characters")
    return raw


def _public_url(raw: str) -> str:
    """Validate the operator-chosen download origin. Errors never echo the value: it may hold
    credentials or tokens pasted by mistake."""
    if not raw:
        raise ConfigError("CHRIEK_PUBLIC_URL is required: the public base URL of download links")
    # urlsplit silently strips some whitespace and control characters; reject them first.
    if not all("!" <= c <= "~" for c in raw):
        raise ConfigError(
            "CHRIEK_PUBLIC_URL must contain only printable ASCII characters without whitespace"
        )
    if "?" in raw or "#" in raw:
        raise ConfigError("CHRIEK_PUBLIC_URL must not contain a query or fragment")
    try:
        parts = urlsplit(raw)
    except ValueError:
        raise ConfigError(PUBLIC_URL_FORM) from None
    if parts.scheme not in ("http", "https"):
        raise ConfigError(PUBLIC_URL_FORM)
    if "@" in parts.netloc:
        raise ConfigError("CHRIEK_PUBLIC_URL must not contain credentials")
    authority = AUTHORITY_PATTERN.fullmatch(parts.netloc)
    if authority is None or (authority["port"] and not 1 <= int(authority["port"]) <= 65535):
        raise ConfigError(PUBLIC_URL_FORM)
    name = authority["name"]
    if name and NUMERIC_LABEL.fullmatch(name.rpartition(".")[2]):
        try:
            ipaddress.IPv4Address(name)  # rejects leading zeros, short and hex forms
        except ValueError:
            raise ConfigError(PUBLIC_URL_FORM) from None
    # Clients normalize backslashes and dot segments, which would change the prefix.
    segments = parts.path.lower().replace("%2e", ".").split("/")
    if "\\" in parts.path or "." in segments or ".." in segments:
        raise ConfigError(PUBLIC_URL_FORM)
    return raw.rstrip("/")
