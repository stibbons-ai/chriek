"""`chriek` command line."""

import argparse
import json
import logging
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import uvicorn
from PIL import Image
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError

from chriek.api import create_app, create_worker_app
from chriek.backend import JobMetrics, create_backend, run_record
from chriek.config import (
    WORKER_TOKEN_HEADER,
    AuthMode,
    BackendSettings,
    ConfigError,
    Limits,
    Role,
    Settings,
    WorkerSettings,
)
from chriek.png import encode_png
from chriek.storage import StorageError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chriek", description="Self-hosted image job server.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "serve", help="Run the API and the in-process worker (home mode). Configured via CHRIEK_*."
    )
    commands.add_parser(
        "api",
        help="Run the serverless API: queue jobs and dispatch each to the worker endpoint.",
    )
    commands.add_parser(
        "worker", help="Run the serverless worker: one dispatched job per HTTP invocation."
    )
    smoke = commands.add_parser("smoke", help="Render one PNG locally, without starting a server.")
    smoke.add_argument("--out", required=True, type=Path)
    smoke.add_argument("--prompt", default="A red bicycle in the rain")
    smoke.add_argument("--reference", action="append", type=Path, default=[])
    smoke.add_argument("--width", type=int, default=1024)
    smoke.add_argument("--height", type=int, default=1024)
    smoke.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    try:
        if args.command == "smoke":
            for axis in ("width", "height"):
                size = getattr(args, axis)
                if size < 16 or size > Limits.max_dimension or size % 16:
                    smoke.error(
                        f"--{axis} must be a multiple of 16 between 16 and {Limits.max_dimension}"
                    )
            if args.width * args.height > Limits.max_pixels:
                smoke.error(f"width * height must be at most {Limits.max_pixels} pixels")
            if not 0 <= args.seed < 2**32:
                smoke.error("--seed must be between 0 and 4294967295")
            return _smoke(args)
        if args.command == "worker":
            return _serve(WorkerSettings.from_env(os.environ))
        role = Role.API if args.command == "api" else Role.HOME
        return _serve(Settings.from_env(os.environ, role))
    except ConfigError as error:
        print(f"chriek: configuration error: {error}", file=sys.stderr)
        return 2


def _serve(settings: Settings | WorkerSettings) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        if isinstance(settings, WorkerSettings):
            app = create_worker_app(settings)
        else:
            app = create_app(settings)
    except StorageError as error:
        print(f"chriek: storage error: {error}", file=sys.stderr)
        return 2
    except (SQLAlchemyError, UnicodeError) as error:  # psycopg's IDNA host encoding: UnicodeError
        print(
            f"chriek: database error: {_database_error(error, settings.database)}", file=sys.stderr
        )
        return 1
    log = logging.getLogger("chriek")
    if isinstance(settings, WorkerSettings):
        log.info("worker: POST /invoke requires the %s header", WORKER_TOKEN_HEADER)
    elif settings.auth_mode is AuthMode.PROXY:
        log.warning(
            "auth mode proxy: no application authentication; every client that reaches this "
            "port shares one owner (allowed Host names: %s)",
            ", ".join(settings.allowed_hosts),
        )
    else:
        log.info("auth mode api-key: %d API keys", len(settings.api_keys))
    # No access log: download URLs carry their signature, a bearer credential until expiry.
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        access_log=False,
        ssl_certfile=settings.tls.certfile if settings.tls else None,
        ssl_keyfile=settings.tls.keyfile if settings.tls else None,
    )
    return 0


def _smoke(args: argparse.Namespace) -> int:
    backend = create_backend(BackendSettings.from_env(os.environ))
    with ExitStack() as stack:
        try:
            references = [stack.enter_context(Image.open(path)) for path in args.reference]
        except (OSError, Image.DecompressionBombError) as error:
            print(f"chriek: cannot open reference: {error}", file=sys.stderr)
            return 1

        started = perf_counter()
        metrics = backend if isinstance(backend, JobMetrics) else None
        try:
            if metrics is not None:
                metrics.start_job()
            image = backend.render(args.prompt, args.width, args.height, args.seed, references)
            png = encode_png(image, model=backend.model, revision=backend.revision, seed=args.seed)
            details = metrics.run_details() if metrics is not None else {}
        except Exception as error:
            print(f"chriek: smoke render failed ({type(error).__name__}): {error}", file=sys.stderr)
            return 1
        try:
            _write_smoke_png(args.out, png)
        except OSError as error:
            print(f"chriek: cannot write PNG: {error}", file=sys.stderr)
            return 1

        record = run_record(backend, perf_counter() - started, details)
        record.setdefault("steps", None)
        record.update(
            prompt=args.prompt,
            seed=args.seed,
            width=image.width,
            height=image.height,
            references=len(references),
            output=str(args.out),
        )
        print(json.dumps(record))
    return 0


def _write_smoke_png(output: Path, png: bytes) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(png)
        os.replace(temporary, output)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _database_error(error: SQLAlchemyError | UnicodeError, database: Path | URL) -> str:
    """The driver's message without SQLAlchemy's wrapper, and never the database password."""
    message = str(getattr(error, "orig", None) or error)
    if isinstance(database, URL):
        for secret in (database.password, database.query.get("password")):
            if isinstance(secret, str) and secret:
                message = message.replace(secret, "***")
    return message
