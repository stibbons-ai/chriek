"""Home-mode worker: one thread, one job at a time, oldest queued job first."""

import logging
import shutil
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from chriek.backend import ImageBackend, JobMetrics, run_record
from chriek.png import encode_png
from chriek.storage import PRIVATE_DIR_MODE, private_image_root, write_private
from chriek.store import Job, JobStore

log = logging.getLogger(__name__)
IDLE_RECHECK_SECONDS = 5.0


def image_path(storage_dir: Path, job_id: uuid.UUID | str, index: int) -> Path:
    return storage_dir / str(job_id) / f"{index}.png"


def reference_path(storage_dir: Path, job_id: str, index: int) -> Path:
    return storage_dir / job_id / "references" / f"{index}.png"


def write_references(storage_dir: Path, job_id: str, references: list[bytes]) -> None:
    if not references:
        return
    references_dir = reference_path(storage_dir, job_id, 0).parent
    references_dir.parent.mkdir(mode=PRIVATE_DIR_MODE)
    references_dir.mkdir(mode=PRIVATE_DIR_MODE)
    for index, png_bytes in enumerate(references):
        # The job row doesn't exist yet and a failed submission removes `<id>/`: no rename needed.
        write_private(reference_path(storage_dir, job_id, index), png_bytes)


class Worker:
    def __init__(
        self,
        store: JobStore,
        backend: ImageBackend,
        storage_dir: Path,
        clock: Callable[[], float],
        timeout_seconds: int,
    ) -> None:
        self._store = store
        self._backend = backend
        self._storage_dir = private_image_root(storage_dir)
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
                self._run(job)
            except Exception:
                # Only the final state write can raise here. Fail closed: the job stays
                # `running` until a restart or an overdue poll fails it, and no later job may run
                # alongside it.
                log.exception("could not record the result of job %s; worker stopped", job.id)
                self.stopped_on_error = True
                return

    def _run(self, job: Job) -> None:
        started = self._clock()
        partial = self._storage_dir / f"{job.id}.partial"
        final = self._storage_dir / job.id
        references: list[Image.Image] = []
        for ref in job.references:
            try:
                path = reference_path(self._storage_dir, job.id, ref.index)
                with Image.open(path, formats=("PNG",)) as opened:
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
                shutil.rmtree(partial, ignore_errors=True)
                partial.mkdir(mode=PRIVATE_DIR_MODE)
                for index in range(job.image_count):
                    seed = job.seed + index
                    image = self._backend.render(
                        job.prompt, job.width, job.height, seed, references
                    )
                    png = encode_png(
                        image, model=self._backend.model, revision=self._backend.revision, seed=seed
                    )
                    write_private(partial / f"{index}.png", png)
            finally:
                if metrics is not None:
                    details = metrics.run_details()
            # Reference jobs already own `<id>/`, so outputs move in one by one.
            final.mkdir(mode=PRIVATE_DIR_MODE, exist_ok=True)
            for index in range(job.image_count):
                (partial / f"{index}.png").replace(image_path(self._storage_dir, job.id, index))
            shutil.rmtree(partial, ignore_errors=True)
        except Exception:
            # Any image failure fails the whole job; already rendered images are discarded.
            # The traceback stays in the server log; the public error never carries exception text.
            log.exception("job %s failed at image %d", job.id, index)
            shutil.rmtree(partial, ignore_errors=True)
            self._remove_outputs(job.id, job.image_count)
            # One completion time decides the timeout and is the one persisted.
            finished = self._clock()
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
            # publish it. Reference jobs keep `<id>/references/`, so only outputs go.
            self._remove_outputs(job.id, job.image_count)
            if not job.references:
                shutil.rmtree(final, ignore_errors=True)

    def _remove_outputs(self, job_id: str, image_count: int) -> None:
        for i in range(image_count):
            path = image_path(self._storage_dir, job_id, i)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.exception("could not remove unpublished output %s for job %s", path, job_id)

    def _fail_if_timed_out(self, job: Job, now: float) -> bool:
        """A result finished past the deadline fails the job as timed out, polled or not."""
        # Local check first: a job finished in time costs no extra write.
        return job.started_at < now - self._timeout_seconds and self._store.fail_timed_out(
            job.id, self._timeout_seconds, now
        )
