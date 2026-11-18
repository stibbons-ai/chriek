"""`chriek` command line."""

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError

from chriek.api import create_app
from chriek.config import AuthMode, ConfigError, Settings
from chriek.storage import StorageError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chriek", description="Self-hosted image job server.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "serve", help="Run the API and the in-process worker (home mode). Configured via CHRIEK_*."
    )
    parser.parse_args(argv)

    try:
        settings = Settings.from_env(os.environ)
    except ConfigError as error:
        print(f"chriek: configuration error: {error}", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
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
    if settings.auth_mode is AuthMode.PROXY:
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


def _database_error(error: SQLAlchemyError | UnicodeError, database: Path | URL) -> str:
    """The driver's message without SQLAlchemy's wrapper, and never the database password."""
    message = str(getattr(error, "orig", None) or error)
    if isinstance(database, URL):
        for secret in (database.password, database.query.get("password")):
            if isinstance(secret, str) and secret:
                message = message.replace(secret, "***")
    return message
