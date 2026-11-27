"""Serverless push dispatch: `chriek api` sends each committed job's ID to the worker endpoint.

No retry and no polling (ADR-0001): a job whose delivery fails is failed while still queued,
so a client polling it sees why instead of waiting forever.
"""

import http.client
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable

from chriek.config import WORKER_TOKEN_HEADER, Dispatch
from chriek.store import JobStore

log = logging.getLogger(__name__)
# Covers connect plus response; with `Prefer: respond-async` providers answer right away. A
# synchronous endpoint that holds the request longer already claimed the job, which a late
# failure then can't touch.
DISPATCH_TIMEOUT_SECONDS = 30.0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would resend the worker token and provider credential to another URL."""

    def redirect_request(self, *_args, **_kwargs) -> None:
        return None


def dispatch(settings: Dispatch, store: JobStore, job_id: str, clock: Callable[[], float]) -> None:
    """Deliver `job_id` once; on failure fail the job if it is still queued."""
    request = urllib.request.Request(  # noqa: S310 - http(s) URL checked by configuration
        settings.url,
        data=json.dumps({"job_id": job_id}).encode(),
        method="POST",
        headers={
            **settings.headers,
            "Content-Type": "application/json",
            # The job ID doubles as the request's idempotency key: a provider retry or a
            # duplicate delivery claims the same job, which the worker runs at most once.
            "Idempotency-Key": job_id,
            WORKER_TOKEN_HEADER: settings.worker_token,
        },
    )
    try:
        # Built per call, not at import: the HTTPS handler reads SSLKEYLOGFILE when created.
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(request, timeout=DISPATCH_TIMEOUT_SECONDS):
            pass  # any 2xx is a delivery; the body carries nothing chriek uses
        return
    except urllib.error.HTTPError as error:
        reason = f"the worker endpoint answered HTTP {error.code}"
        error.close()
    except (OSError, ValueError, http.client.HTTPException) as error:  # timeouts, refusals
        reason = f"the worker endpoint could not be reached ({type(error).__name__})"
    log.error("dispatching job %s failed: %s", job_id, reason)
    try:
        failed = store.fail_queued(job_id, f"dispatch failed: {reason}", clock())
    except Exception:
        log.exception("could not record the dispatch failure of job %s", job_id)
        return
    if not failed:
        log.info("job %s already left the queue; its dispatch failure changes nothing", job_id)
