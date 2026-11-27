"""Job execution. Home mode runs a thread, one job at a time, oldest queued job first; the
serverless worker calls `Worker.run` for each job it claims on invocation, without the thread."""

import io
import logging
import threading
from collections.abc import Callable

from PIL import Image

from chriek.backend import ImageBackend, JobMetrics, run_record
from chriek.images import LocalImages, S3Images
from chriek.png import encode_png
from chriek.store import Job, JobStore

log = logging.getLogger(__name__)
IDLE_RECHECK_SECONDS = 5.0


class Worker:
    def __init__(
        self,
        store: JobStore,
        backend: ImageBackend,
        images: LocalImages | S3Images,
        clock: Callable[[], float],
        timeout_seconds: int,
    ) -> None:
        self._store = store
        self._backend = backend
        self._images = images
        self._clock = clock
        self._timeout_seconds = timeout_seconds
        self._wake = threading.Event()
        self._stopping = False
        self.stopped_on_error = False
        self._thread = threading.Thread(target=self._loop, name="chriek-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def notify(self) -> None:
        self._wake.set()

    def stop(self, timeout: float) -> None:
        """Stop after the current job. A job still running at exit fails on the next startup."""
        self._stopping = True
        self._wake.set()
        self._thread.join(timeout)

    def _loop(self) -> None:
        while not self._stopping:
            self._wake.clear()
            try:
                job = self._store.claim_next(self._clock())
            except Exception:
                log.exception("claiming the next job failed")
                self._wake.wait(IDLE_RECHECK_SECONDS)
                continue
            if job is None:
                self._wake.wait(IDLE_RECHECK_SECONDS)
                continue
            try:
                self.run(job)
            except Exception:
                # Only the final state write can raise here. Fail closed: the job stays
                # `running` until a restart or an overdue poll fails it, and no later job may run
                # alongside it.
                log.exception("could not record the result of job %s; worker stopped", job.id)
                self.stopped_on_error = True
                return

    def run(self, job: Job) -> None:
        """Render a claimed (running) job and record its result. Raises only when that final
        state write fails."""
        started = self._clock()
        references: list[Image.Image] = []
        for ref in job.references:
            try:
                data = self._images.read_reference(job.id, ref.index)
                with Image.open(io.BytesIO(data), formats=("PNG",)) as opened:
                    references.append(opened.copy())
            except Exception:
                # Like render failures: the traceback stays in the log, never in the public error.
                log.exception("job %s failed loading reference %d", job.id, ref.index)
                finished = self._clock()
                error = f"reference {ref.index + 1} of {len(job.references)} could not be loaded"
                if not self._fail_if_timed_out(job, finished):
                    # The backend never ran, so it has no job metrics to report.
                    record = run_record(self._backend, finished - started, {})
                    self._store.fail(job.id, error, record, finished)
                return
        metrics = self._backend if isinstance(self._backend, JobMetrics) else None
        details: dict[str, object] = {}
        index = 0
        try:
            if metrics is not None:
                metrics.start_job()
            try:
                self._images.begin_outputs(job.id)
                for index in range(job.image_count):
                    seed = job.seed + index
                    image = self._backend.render(
                        job.prompt, job.width, job.height, seed, references
                    )
                    png = encode_png(
                        image, model=self._backend.model, revision=self._backend.revision, seed=seed
                    )
                    self._images.stage_output(job.id, index, png)
            finally:
                if metrics is not None:
                    details = metrics.run_details()
            self._images.publish_outputs(job.id, job.image_count)
        except Exception:
            # Any image failure fails the whole job; already rendered images are discarded.
            # The traceback stays in the server log; the public error never carries exception text.
            log.exception("job %s failed at image %d", job.id, index)
            # One completion time decides the timeout and is the one persisted; taken before the
            # cleanup so a slow delete cannot turn this failure into a timeout.
            finished = self._clock()
            self._images.remove_outputs(job.id, job.image_count)
            error = f"image {index + 1} of {job.image_count} failed"
            if not self._fail_if_timed_out(job, finished):
                self._store.fail(
                    job.id, error, run_record(self._backend, finished - started, details), finished
                )
            return
        finished = self._clock()
        if self._fail_if_timed_out(job, finished) or not self._store.succeed(
            job.id, run_record(self._backend, finished - started, details), finished
        ):
            # The job is no longer running (timed out, or failed by restart recovery); never
            # publish it. References stay with their job; only outputs go.
            self._images.remove_outputs(job.id, job.image_count)

    def _fail_if_timed_out(self, job: Job, now: float) -> bool:
        """A result finished past the deadline fails the job as timed out, polled or not."""
        # Local check first: a job finished in time costs no extra write.
        return job.started_at < now - self._timeout_seconds and self._store.fail_timed_out(
            job.id, self._timeout_seconds, now
        )
