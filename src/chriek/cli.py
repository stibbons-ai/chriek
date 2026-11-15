"""`chriek` command line."""

import argparse
import logging
import os
import sys

import uvicorn

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
