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
            # Validated distinct from every other header name, case-insensitively.
            **({settings.id_header: job_id} if settings.id_header else {}),
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
        with opener.open(request, timeout=settings.timeout_seconds):
            pass  # any 2xx is a delivery; the body carries nothing chriek uses
        return
    except urllib.error.HTTPError as error:
        reason = f"the worker endpoint answered HTTP {error.code}"
        error.close()
    except (OSError, ValueError, http.client.HTTPException) as error:  # timeouts, refusals
        # urllib wraps a connect timeout in URLError; a read timeout arrives bare.
        if isinstance(error, TimeoutError) or isinstance(
            getattr(error, "reason", None), TimeoutError
        ):
            reason = (
                f"the worker endpoint gave no answer within {settings.timeout_seconds:g} seconds"
            )
        else:
            reason = f"the worker endpoint could not be reached ({type(error).__name__})"
    try:
        failed = store.fail_queued(job_id, f"dispatch failed: {reason}", clock())
    except Exception:
        log.exception("dispatching job %s failed (%s); recording that failed too", job_id, reason)
        return
    if failed:
        log.error("dispatching job %s failed: %s; the job failed", job_id, reason)
    else:  # a worker claimed it meanwhile, e.g. a synchronous endpoint answering after rendering
        log.info(
            "dispatching job %s got no 2xx answer (%s) after a worker claimed it; its state stands",
            job_id,
            reason,
        )
