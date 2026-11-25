"""Environment-variable configuration. Every name and default is documented in README.md."""

import ipaddress
import os
import re
import ssl
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

MIN_SECRET_LENGTH = 32
POSTGRES_DRIVERS = ("postgresql", "postgresql+psycopg")
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

# Every supported CHRIEK_* name; any other name with that prefix stops startup.
SETTING_NAMES = frozenset(
    {
        "CHRIEK_ALLOWED_HOSTS",
        "CHRIEK_API_KEYS",
        "CHRIEK_AUTH_MODE",
        "CHRIEK_BACKEND",
        "CHRIEK_DATABASE_URL",
        "CHRIEK_DB_PATH",
        "CHRIEK_HOST",
        "CHRIEK_JOB_TIMEOUT_SECONDS",
        "CHRIEK_MAX_DIMENSION",
        "CHRIEK_MAX_IMAGES",
        "CHRIEK_MAX_PIXELS",
        "CHRIEK_MAX_PROMPT_CHARS",
        "CHRIEK_MAX_QUEUED_JOBS",
        "CHRIEK_MAX_REFERENCES",
        "CHRIEK_MAX_UPLOAD_BYTES",
        "CHRIEK_OFFLOAD",
        "CHRIEK_PORT",
        "CHRIEK_PUBLIC_URL",
        "CHRIEK_SIGNING_SECRET",
        "CHRIEK_STORAGE_DIR",
        "CHRIEK_TLS_CERTFILE",
        "CHRIEK_TLS_KEYFILE",
        "CHRIEK_URL_TTL_SECONDS",
    }
)
DEFAULT_ALLOWED_HOSTS = "localhost,127.0.0.1,[::1]"
_LABEL = r"[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?"
HOSTNAME_PATTERN = re.compile(rf"{_LABEL}(?:\.{_LABEL})*")  # IPv4 addresses match too
BRACKETED_IPV6_PATTERN = re.compile(r"\[([0-9a-f:.]+)\]")


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
    max_references: int = 4
    max_upload_bytes: int = 20 * 1024 * 1024


@dataclass(frozen=True)
class Tls:
    certfile: Path
    keyfile: Path


@dataclass(frozen=True)
class BackendSettings:
    name: str = "fake"
    offload: str = "none"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> BackendSettings:
        unknown = sorted(
            name for name in env if name.upper().startswith("CHRIEK_") and name not in SETTING_NAMES
        )
        if unknown:
            raise ConfigError(f"unknown setting {', '.join(unknown)}; see README.md for names")
        name = env.get("CHRIEK_BACKEND", "fake")
        if name not in ("fake", "mlx", "cuda"):
            raise ConfigError("CHRIEK_BACKEND must be 'fake', 'mlx' or 'cuda'")
        offload = env.get("CHRIEK_OFFLOAD", "none")
        if offload != "none" and (name, offload) != ("cuda", "sequential"):
            raise ConfigError(
                "CHRIEK_OFFLOAD must be 'none', or 'sequential' with CHRIEK_BACKEND=cuda"
            )
        return cls(name, offload)


@dataclass(frozen=True)
class Settings:
    database: Path | URL  # SQLite file (or ":memory:") or a Postgres URL
    public_url: str  # scheme://authority[/prefix], no trailing slash
    storage_dir: Path
    auth_mode: AuthMode
    api_keys: Mapping[str, str]  # owner name -> key; empty in proxy mode
    signing_secret: str
    url_ttl_seconds: int
    job_timeout_seconds: int  # running time, from started_at, after which a job fails
    limits: Limits
    host: str
    port: int
    tls: Tls | None  # None serves plain HTTP
    allowed_hosts: tuple[str, ...]  # proxy mode's accepted Host names; empty in api-key mode
    backend: BackendSettings = BackendSettings()

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        backend = BackendSettings.from_env(env)
        limits = Limits(
            max_images=_int(env, "CHRIEK_MAX_IMAGES", Limits.max_images, 1, 64),
            max_dimension=_int(env, "CHRIEK_MAX_DIMENSION", Limits.max_dimension, 16, 8192),
            max_pixels=_int(env, "CHRIEK_MAX_PIXELS", Limits.max_pixels, 256, 8192 * 8192),
            max_prompt_chars=_int(
                env, "CHRIEK_MAX_PROMPT_CHARS", Limits.max_prompt_chars, 1, 100_000
            ),
            max_queued_jobs=_int(env, "CHRIEK_MAX_QUEUED_JOBS", Limits.max_queued_jobs, 1, 100_000),
            # 10 is flux3-mlx's hard reference limit: never admit a job the backend cannot render.
            max_references=_int(env, "CHRIEK_MAX_REFERENCES", Limits.max_references, 0, 10),
            max_upload_bytes=_int(
                env, "CHRIEK_MAX_UPLOAD_BYTES", Limits.max_upload_bytes, 1024, 512 * 1024 * 1024
            ),
        )
        if limits.max_dimension % 16:
            raise ConfigError("CHRIEK_MAX_DIMENSION must be a multiple of 16")
        auth_mode = _auth_mode(env.get("CHRIEK_AUTH_MODE", AuthMode.API_KEY))
        if auth_mode is AuthMode.API_KEY and "CHRIEK_ALLOWED_HOSTS" in env:
            # Host checking runs only in proxy mode; refusing it here keeps it from looking active.
            raise ConfigError("CHRIEK_ALLOWED_HOSTS applies only when CHRIEK_AUTH_MODE is 'proxy'")
        raw_keys = env.get("CHRIEK_API_KEYS", "")
        api_keys = _no_api_keys(raw_keys) if auth_mode is AuthMode.PROXY else _api_keys(raw_keys)
        signing_secret = _signing_secret(env.get("CHRIEK_SIGNING_SECRET", ""))
        if signing_secret in api_keys or signing_secret in api_keys.values():
            raise ConfigError(
                "CHRIEK_SIGNING_SECRET must differ from every CHRIEK_API_KEYS name and key"
            )
        return cls(
            public_url=_public_url(env.get("CHRIEK_PUBLIC_URL", "")),
            database=_database(env),
            storage_dir=Path(env.get("CHRIEK_STORAGE_DIR", "chriek-data/images")),
            auth_mode=auth_mode,
            api_keys=api_keys,
            signing_secret=signing_secret,
            url_ttl_seconds=_int(env, "CHRIEK_URL_TTL_SECONDS", 900, 1, 7 * 24 * 3600),
            job_timeout_seconds=_int(env, "CHRIEK_JOB_TIMEOUT_SECONDS", 3600, 1, 7 * 24 * 3600),
            limits=limits,
            host=env.get("CHRIEK_HOST", "127.0.0.1"),
            port=_int(env, "CHRIEK_PORT", 8000, 1, 65535),
            tls=_tls(env),
            allowed_hosts=(
                _allowed_hosts(env.get("CHRIEK_ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS))
                if auth_mode is AuthMode.PROXY
                else ()
            ),
            backend=backend,
        )


def _database(env: Mapping[str, str]) -> Path | URL:
    """CHRIEK_DATABASE_URL (Postgres) takes precedence over CHRIEK_DB_PATH (SQLite)."""
    raw = env.get("CHRIEK_DATABASE_URL")
    if raw is None:
        return Path(env.get("CHRIEK_DB_PATH", "chriek-data/chriek.db"))
    # The URL carries the database password: errors name the variable, never the value.
    # A raw `@` outside the authority (a second one, or one after the first `/` or `?` past
    # `://`) makes the parser split credentials into host or database names, which drivers echo.
    if raw.count("@") > 1 or re.search(r"://[^/?]*[/?].*@", raw, re.DOTALL):
        raise ConfigError(
            "CHRIEK_DATABASE_URL must percent-encode every @ as %40, except the one before the host"
        )
    try:
        url = make_url(raw)
    except ArgumentError, ValueError:
        raise ConfigError("CHRIEK_DATABASE_URL is not a valid database URL") from None
    if url.drivername not in POSTGRES_DRIVERS:
        raise ConfigError(
            "CHRIEK_DATABASE_URL must start with postgresql:// or postgresql+psycopg://"
        )
    if not all(_is_conninfo_text(text) for text in _url_texts(url)):
        raise ConfigError(
            "CHRIEK_DATABASE_URL must not contain NUL characters (also as %00) or raw invalid UTF-8"
        )
    return url


def _url_texts(url: URL) -> Iterator[str]:
    """Every URL part the driver receives (percent-decoded except the host); repeated query
    keys hold tuples."""
    yield from (text for text in (url.username, url.password, url.host, url.database) if text)
    for key, value in url.query.items():
        yield key
        yield from (value,) if isinstance(value, str) else value


def _is_conninfo_text(text: str) -> bool:
    """libpq takes NUL-terminated UTF-8 strings: a NUL truncates the connection settings."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return "\x00" not in text


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
    # Diagnostics cite entry positions only: a reversed key:name puts a secret in the name.
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
    # Diagnostics only, naming the bad variable but not its value, which may be pasted key
    # material: the pair load below stays authoritative.
    for name, path in (("CHRIEK_TLS_CERTFILE", tls.certfile), ("CHRIEK_TLS_KEYFILE", tls.keyfile)):
        if not path.is_file() or not os.access(path, os.R_OK):
            raise ConfigError(
                f"{name} must name a readable file; the configured path is missing, "
                "not a regular file or unreadable"
            )
    try:
        # Loading the pair here turns a bad file into a configuration error before the
        # database or port opens. The callback rejects encrypted keys instead of prompting.
        # A bare server context, like uvicorn's: create_default_context would also open
        # SSLKEYLOGFILE and blame its failure on the certificate pair.
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(
            tls.certfile, tls.keyfile, password=_reject_encrypted_key
        )
    except OSError as error:  # ssl.SSLError is an OSError
        raise ConfigError(
            "CHRIEK_TLS_CERTFILE and CHRIEK_TLS_KEYFILE must name a readable PEM certificate "
            f"and its unencrypted private key: {error}"
        ) from None
    return tls


def _allowed_hosts(raw: str) -> tuple[str, ...]:
    """Parse exact host names and IP addresses, IPv6 in brackets as in a Host header."""
    hosts = []
    for position, entry in enumerate(raw.split(","), start=1):
        host = entry.strip().lower()
        if not _is_valid_host(host):
            raise ConfigError(
                f"CHRIEK_ALLOWED_HOSTS entry {position} must be a host name, an IPv4 address or "
                "a bracketed IPv6 address, without wildcard, scheme, port or path"
            )
        hosts.append(host)
    return tuple(hosts)


def _is_valid_host(host: str) -> bool:
    ipv6 = BRACKETED_IPV6_PATTERN.fullmatch(host)  # no zone IDs, which Host can't carry
    if ipv6 is None:
        return HOSTNAME_PATTERN.fullmatch(host) is not None
    try:
        ipaddress.IPv6Address(ipv6[1])
    except ValueError:
        return False
    return True


def _reject_encrypted_key() -> str:
    raise ConfigError("CHRIEK_TLS_KEYFILE must be an unencrypted private key")
