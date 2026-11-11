"""`chriek` command line."""

import argparse
import logging
import os
import sys

import uvicorn

from chriek.api import create_app
from chriek.config import ConfigError, Settings


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
    # No access log: download URLs carry their signature, a bearer credential until expiry.
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, access_log=False)
    return 0
