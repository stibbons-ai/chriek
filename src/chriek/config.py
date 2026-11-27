"""Environment-variable configuration. Every name and default is documented in README.md."""

import ipaddress
import json
import os
import re
import ssl
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
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
URL_FORM = (
    "{} must be an absolute http or https URL with a host, an optional port "
    "1-65535 and an optional path prefix"
)
ORIGIN_FORM = (
    "{} must be an absolute http or https URL with a host and an optional port 1-65535, "
    "without a path"
)
S3_BUCKET_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
S3_REGION_PATTERN = re.compile(r"[a-z0-9-]{1,32}")
S3_PREFIX_PATTERN = re.compile(r"(?=.{1,256}$)[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")

# Every supported CHRIEK_* name; any other name with that prefix stops startup.
SETTING_NAMES = frozenset(
    {
        "CHRIEK_ALLOWED_HOSTS",
        "CHRIEK_API_KEYS",
        "CHRIEK_AUTH_MODE",
        "CHRIEK_BACKEND",
        "CHRIEK_DATABASE_URL",
        "CHRIEK_DB_PATH",
        "CHRIEK_DISPATCH_HEADERS",
        "CHRIEK_DISPATCH_ID_HEADER",
        "CHRIEK_DISPATCH_TIMEOUT_SECONDS",
        "CHRIEK_DISPATCH_TOKEN",
        "CHRIEK_DISPATCH_URL",
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
        "CHRIEK_S3_ACCESS_KEY_ID",
        "CHRIEK_S3_BUCKET",
        "CHRIEK_S3_ENDPOINT_URL",
        "CHRIEK_S3_PREFIX",
        "CHRIEK_S3_REGION",
        "CHRIEK_S3_SECRET_ACCESS_KEY",
        "CHRIEK_SIGNING_SECRET",
        "CHRIEK_STORAGE",
        "CHRIEK_STORAGE_DIR",
        "CHRIEK_TLS_CERTFILE",
        "CHRIEK_TLS_KEYFILE",
        "CHRIEK_URL_TTL_SECONDS",
        "CHRIEK_WORKER_TOKEN",
    }
)
DISPATCH_NAMES = frozenset(
    {
        "CHRIEK_DISPATCH_HEADERS",
        "CHRIEK_DISPATCH_ID_HEADER",
        "CHRIEK_DISPATCH_TIMEOUT_SECONDS",
        "CHRIEK_DISPATCH_TOKEN",
        "CHRIEK_DISPATCH_URL",
    }
)
# What `chriek worker` reads; it refuses every other setting, so no client API key, signing
# secret or dispatch credential has to reach the GPU container.
WORKER_SETTING_NAMES = frozenset(
    {
        "CHRIEK_BACKEND",
        "CHRIEK_DATABASE_URL",
        "CHRIEK_HOST",
        "CHRIEK_JOB_TIMEOUT_SECONDS",
        "CHRIEK_OFFLOAD",
        "CHRIEK_PORT",
        "CHRIEK_S3_ACCESS_KEY_ID",
        "CHRIEK_S3_BUCKET",
        "CHRIEK_S3_ENDPOINT_URL",
        "CHRIEK_S3_PREFIX",
        "CHRIEK_S3_REGION",
        "CHRIEK_S3_SECRET_ACCESS_KEY",
        "CHRIEK_STORAGE",
        "CHRIEK_TLS_CERTFILE",
        "CHRIEK_TLS_KEYFILE",
        "CHRIEK_WORKER_TOKEN",
    }
)
# The worker invocation credential travels in its own header, so a provider gateway may consume
# `Authorization` (CHRIEK_DISPATCH_TOKEN) without chriek's worker losing its check.
WORKER_TOKEN_HEADER = "X-Chriek-Worker-Token"  # noqa: S105 - a header name, not a secret
HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")  # RFC 9110 token
RESERVED_DISPATCH_HEADERS = (
    "Authorization",
    "Content-Type",
    "Idempotency-Key",
    WORKER_TOKEN_HEADER,
)
# Never settable as CHRIEK_DISPATCH_ID_HEADER: a per-job value there would break the request's
# framing, connection handling or routing.
FRAMING_HEADERS = (
    "Connection",
    "Content-Length",
    "Expect",
    "Host",
    "Keep-Alive",
    "Proxy-Connection",
    "TE",
    "Trailer",
    "Transfer-Encoding",
    "Upgrade",
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


class Role(StrEnum):
    """The `chriek` command a configuration is for; values appear in error messages."""

    HOME = "serve"  # API and in-process worker, one process
    API = "api"  # serverless API: queues jobs, dispatches each to the worker endpoint
    WORKER = "worker"  # serverless worker: runs one dispatched job per invocation


# Refused, not ignored: a setting that does nothing in a role must not look active there.
_REFUSED_SETTINGS = {
    Role.HOME: DISPATCH_NAMES | {"CHRIEK_WORKER_TOKEN"},
    Role.API: frozenset({"CHRIEK_BACKEND", "CHRIEK_OFFLOAD"}),  # the API never loads a model
    Role.WORKER: SETTING_NAMES - WORKER_SETTING_NAMES,
}


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

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> BackendSettings:
        unknown = sorted(
            name for name in env if name.upper().startswith("CHRIEK_") and name not in SETTING_NAMES
        )
        if unknown:
            raise ConfigError(f"unknown setting {', '.join(unknown)}; see README.md for names")
        name = env.get("CHRIEK_BACKEND", "fake")
        if name not in ("fake", "mlx"):
            raise ConfigError("CHRIEK_BACKEND must be 'fake' or 'mlx'")
        if env.get("CHRIEK_OFFLOAD", "none") != "none":
            raise ConfigError("CHRIEK_OFFLOAD must be 'none'")
        return cls(name)


@dataclass(frozen=True)
class S3Settings:
    bucket: str
    endpoint_url: str | None  # None: the AWS endpoint for `region`
    region: str
    prefix: str  # key prefix without slashes at either end; "" stores at the bucket root
    access_key_id: str
    secret_access_key: str = field(repr=False)


@dataclass(frozen=True)
class Dispatch:
    """Where `chriek api` sends each job ID, and with which headers."""

    url: str = field(repr=False)  # a provider may put a token in the path
    worker_token: str = field(repr=False)
    headers: Mapping[str, str] = field(repr=False)  # extra headers, incl. a provider credential
    # Per connect and per read, not a total deadline; a synchronous endpoint answers after
    # rendering.
    timeout_seconds: float
    id_header: str | None = None  # also carries the job ID, e.g. a provider's inference ID


@dataclass(frozen=True)
class Settings:
    database: Path | URL  # SQLite file (or ":memory:") or a Postgres URL
    public_url: str  # scheme://authority[/prefix], no trailing slash
    storage: Path | S3Settings  # local image directory or S3-compatible bucket
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
    dispatch: Dispatch | None = None  # set only for `chriek api`

    @classmethod
    def from_env(cls, env: Mapping[str, str], role: Role = Role.HOME) -> Settings:
        """Home (`chriek serve`) or serverless API (`chriek api`) settings."""
        _refuse(env, role)
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
        database, storage, dispatch = _database(env), _storage(env), None
        if role is Role.API:
            database, storage = _shared_storage(role, database, storage)
            dispatch = _dispatch(env, {signing_secret, *api_keys, *api_keys.values()})
        return cls(
            public_url=_public_url(env.get("CHRIEK_PUBLIC_URL", "")),
            database=database,
            storage=storage,
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
            dispatch=dispatch,
        )


@dataclass(frozen=True)
class WorkerSettings:
    """`chriek worker`: shared Postgres and bucket, the model backend and the invocation token."""

    database: URL
    storage: S3Settings
    worker_token: str = field(repr=False)
    job_timeout_seconds: int
    host: str
    port: int
    tls: Tls | None
    backend: BackendSettings

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> WorkerSettings:
        _refuse(env, Role.WORKER)
        backend = BackendSettings.from_env(env)
        database, storage = _shared_storage(Role.WORKER, _database(env), _storage(env))
        return cls(
            database=database,
            storage=storage,
            worker_token=_worker_token(env),
            job_timeout_seconds=_int(env, "CHRIEK_JOB_TIMEOUT_SECONDS", 3600, 1, 7 * 24 * 3600),
            host=env.get("CHRIEK_HOST", "127.0.0.1"),
            port=_int(env, "CHRIEK_PORT", 8000, 1, 65535),
            tls=_tls(env),
            backend=backend,
        )


def _refuse(env: Mapping[str, str], role: Role) -> None:
    refused = sorted(name for name in env if name in _REFUSED_SETTINGS[role])
    if refused:
        raise ConfigError(f"{', '.join(refused)} does not apply to chriek {role}; unset it")


def _shared_storage(
    role: Role, database: Path | URL, storage: Path | S3Settings
) -> tuple[URL, S3Settings]:
    """API and worker run in different places: both need the one Postgres and bucket."""
    if not isinstance(database, URL):
        raise ConfigError(
            f"chriek {role} requires CHRIEK_DATABASE_URL (Postgres shared by api and worker)"
        )
    if not isinstance(storage, S3Settings):
        raise ConfigError(
            f"chriek {role} requires CHRIEK_STORAGE=s3 (a bucket shared by api and worker)"
        )
    return database, storage


def _dispatch(env: Mapping[str, str], other_secrets: set[str]) -> Dispatch:
    url = env.get("CHRIEK_DISPATCH_URL", "")
    if not url:
        raise ConfigError(
            "CHRIEK_DISPATCH_URL is required for chriek api: the worker invocation URL"
        )
    _http_url(url, "CHRIEK_DISPATCH_URL")  # validated, kept verbatim: a trailing / may matter
    worker_token = _worker_token(env)
    headers = _dispatch_headers(env.get("CHRIEK_DISPATCH_HEADERS", "{}"))
    id_header = _dispatch_id_header(env.get("CHRIEK_DISPATCH_ID_HEADER"), headers)
    token = env.get("CHRIEK_DISPATCH_TOKEN")
    if token is not None and (not token or not all("!" <= c <= "~" for c in token)):
        raise ConfigError(
            "CHRIEK_DISPATCH_TOKEN must be printable ASCII characters without whitespace"
        )
    # Raw values, before "Bearer " is added: the provider credential must never reach the worker.
    if (
        worker_token in other_secrets
        or worker_token == token
        or any(worker_token in value for value in headers.values())
    ):
        raise ConfigError(
            "CHRIEK_WORKER_TOKEN must differ from CHRIEK_SIGNING_SECRET, every CHRIEK_API_KEYS "
            "name and key and CHRIEK_DISPATCH_TOKEN, and appear in no CHRIEK_DISPATCH_HEADERS value"
        )
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    raw_timeout = env.get("CHRIEK_DISPATCH_TIMEOUT_SECONDS", "30")
    try:
        timeout = float(raw_timeout)
    except ValueError:
        timeout = 0.0
    # socket.settimeout overflows on huge finite values; a week bounds any real cold start
    if not 0 < timeout <= 604800:  # also false for NaN and inf
        raise ConfigError(
            "CHRIEK_DISPATCH_TIMEOUT_SECONDS must be a number of seconds above 0 and at most "
            "604800 (7 days)"
        )
    return Dispatch(
        url=url,
        worker_token=worker_token,
        headers=headers,
        timeout_seconds=timeout,
        id_header=id_header,
    )


def _dispatch_headers(raw: str) -> dict[str, str]:
    """A JSON object of extra header names to values. Errors never echo either: a value may be a
    credential, and a name may be one pasted in the wrong place."""
    form = "CHRIEK_DISPATCH_HEADERS must be a JSON object of header names to string values"
    try:
        headers = json.loads(raw)
    except ValueError:
        raise ConfigError(form) from None
    if not isinstance(headers, dict) or not all(isinstance(v, str) for v in headers.values()):
        raise ConfigError(form)
    reserved = {name.lower() for name in RESERVED_DISPATCH_HEADERS}
    for name, value in headers.items():
        if not HEADER_NAME_PATTERN.fullmatch(name) or not all(" " <= c <= "~" for c in value):
            raise ConfigError(
                "CHRIEK_DISPATCH_HEADERS names must be HTTP header tokens and values printable "
                "ASCII characters"
            )
        if name.lower() in reserved:
            raise ConfigError(
                f"CHRIEK_DISPATCH_HEADERS must not set {', '.join(RESERVED_DISPATCH_HEADERS)}; "
                "chriek sets them (CHRIEK_DISPATCH_TOKEN sends Authorization)"
            )
    return headers


def _dispatch_id_header(name: str | None, headers: Mapping[str, str]) -> str | None:
    """The optional extra header carrying the job ID. Errors never echo the name."""
    if name is None:
        return None
    if not HEADER_NAME_PATTERN.fullmatch(name):
        raise ConfigError("CHRIEK_DISPATCH_ID_HEADER must be one HTTP header token")
    taken = {n.lower() for n in (*RESERVED_DISPATCH_HEADERS, *FRAMING_HEADERS, *headers)}
    if name.lower() in taken:
        raise ConfigError(
            "CHRIEK_DISPATCH_ID_HEADER must not name a header chriek sets "
            f"({', '.join(RESERVED_DISPATCH_HEADERS)}), one in CHRIEK_DISPATCH_HEADERS, or "
            f"{', '.join(FRAMING_HEADERS)}; names are case-insensitive"
        )
    return name


def _worker_token(env: Mapping[str, str]) -> str:
    token = env.get("CHRIEK_WORKER_TOKEN", "")
    if len(token) < MIN_SECRET_LENGTH or not all("!" <= c <= "~" for c in token):
        raise ConfigError(
            f"CHRIEK_WORKER_TOKEN must be at least {MIN_SECRET_LENGTH} printable ASCII "
            "characters without whitespace"
        )
    return token


def _storage(env: Mapping[str, str]) -> Path | S3Settings:
    kind = env.get("CHRIEK_STORAGE", "local")
    s3_names = sorted(name for name in env if name.startswith("CHRIEK_S3_"))
    if kind == "local":
        if s3_names:
            # Refused, not ignored: a bucket setting must never look active while images stay local.
            raise ConfigError(f"{', '.join(s3_names)} applies only when CHRIEK_STORAGE is 's3'")
        return Path(env.get("CHRIEK_STORAGE_DIR", "chriek-data/images"))
    if kind != "s3":
        raise ConfigError("CHRIEK_STORAGE must be 'local' or 's3'")
    if "CHRIEK_STORAGE_DIR" in env:
        raise ConfigError("CHRIEK_STORAGE_DIR applies only when CHRIEK_STORAGE is 'local'")
    for name in ("CHRIEK_S3_BUCKET", "CHRIEK_S3_ACCESS_KEY_ID", "CHRIEK_S3_SECRET_ACCESS_KEY"):
        if not env.get(name):
            raise ConfigError(f"{name} is required when CHRIEK_STORAGE is 's3'")
    bucket = env["CHRIEK_S3_BUCKET"]
    if not S3_BUCKET_PATTERN.fullmatch(bucket) or ".." in bucket:
        raise ConfigError(
            "CHRIEK_S3_BUCKET must be an existing bucket name of 3-63 characters from a-z 0-9 . -"
        )
    endpoint_url = env.get("CHRIEK_S3_ENDPOINT_URL")
    if endpoint_url is not None:
        if not endpoint_url:
            raise ConfigError("CHRIEK_S3_ENDPOINT_URL must not be empty; unset it for AWS")
        endpoint_url = _http_url(endpoint_url, "CHRIEK_S3_ENDPOINT_URL", ORIGIN_FORM)
        if urlsplit(endpoint_url).path:
            raise ConfigError("CHRIEK_S3_ENDPOINT_URL must not contain a path")
    region = env.get("CHRIEK_S3_REGION", "us-east-1")
    if not S3_REGION_PATTERN.fullmatch(region):
        raise ConfigError("CHRIEK_S3_REGION must be 1-32 characters from a-z 0-9 -")
    prefix = env.get("CHRIEK_S3_PREFIX", "")
    if prefix and (not S3_PREFIX_PATTERN.fullmatch(prefix) or {".", ".."} & set(prefix.split("/"))):
        raise ConfigError(
            "CHRIEK_S3_PREFIX must be at most 256 characters of /-separated segments from "
            "A-Z a-z 0-9 _ . -, without empty, . or .. segments"
        )
    # Credentials are checked for shape only; errors name the variable, never the value.
    for name in ("CHRIEK_S3_ACCESS_KEY_ID", "CHRIEK_S3_SECRET_ACCESS_KEY"):
        if not all("!" <= c <= "~" for c in env[name]):
            raise ConfigError(f"{name} must be printable ASCII characters without whitespace")
    return S3Settings(
        bucket=bucket,
        endpoint_url=endpoint_url,
        region=region,
        prefix=prefix,
        access_key_id=env["CHRIEK_S3_ACCESS_KEY_ID"],
        secret_access_key=env["CHRIEK_S3_SECRET_ACCESS_KEY"],
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
    if not raw:
        raise ConfigError("CHRIEK_PUBLIC_URL is required: the public base URL of download links")
    return _http_url(raw, "CHRIEK_PUBLIC_URL")


def _http_url(raw: str, name: str, form: str = URL_FORM) -> str:
    """Validate a non-empty, operator-chosen http(s) origin with optional path prefix; `form`
    describes the accepted shape. Errors never echo the value: it may hold credentials or
    tokens pasted by mistake."""
    url_form = form.format(name)
    # urlsplit silently strips some whitespace and control characters; reject them first.
    if not all("!" <= c <= "~" for c in raw):
        raise ConfigError(f"{name} must contain only printable ASCII characters without whitespace")
    if "?" in raw or "#" in raw:
        raise ConfigError(f"{name} must not contain a query or fragment")
    try:
        parts = urlsplit(raw)
    except ValueError:
        raise ConfigError(url_form) from None
    if parts.scheme not in ("http", "https"):
        raise ConfigError(url_form)
    if "@" in parts.netloc:
        raise ConfigError(f"{name} must not contain credentials")
    authority = AUTHORITY_PATTERN.fullmatch(parts.netloc)
    if authority is None or (authority["port"] and not 1 <= int(authority["port"]) <= 65535):
        raise ConfigError(url_form)
    host = authority["name"]
    if host and NUMERIC_LABEL.fullmatch(host.rpartition(".")[2]):
        try:
            ipaddress.IPv4Address(host)  # rejects leading zeros, short and hex forms
        except ValueError:
            raise ConfigError(url_form) from None
    # Clients normalize backslashes and dot segments, which would change the prefix.
    segments = parts.path.lower().replace("%2e", ".").split("/")
    if "\\" in parts.path or "." in segments or ".." in segments:
        raise ConfigError(url_form)
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
