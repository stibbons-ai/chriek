"""Home-mode worker: one thread, one job at a time, oldest queued job first."""

import logging
import shutil
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from chriek.backend import ImageBackend
from chriek.png import encode_png
from chriek.store import Job, JobStore

log = logging.getLogger(__name__)
IDLE_RECHECK_SECONDS = 5.0


def image_path(storage_dir: Path, job_id: uuid.UUID, index: int) -> Path:
    return storage_dir / str(job_id) / f"{index}.png"


class Worker:
    def __init__(
        self,
        store: JobStore,
        backend: ImageBackend,
        storage_dir: Path,
        clock: Callable[[], float],
    ) -> None:
        self._store = store
        self._backend = backend
        self._storage_dir = storage_dir
        self._clock = clock
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
                # `running` until a restart fails it, and no later job may run alongside it.
                log.exception("could not record the result of job %s; worker stopped", job.id)
                self.stopped_on_error = True
                return

    def _run(self, job: Job) -> None:
        started = self._clock()
        partial = self._storage_dir / f"{job.id}.partial"
        final = self._storage_dir / job.id
        index = 0
        try:
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir(parents=True)
            for index in range(job.image_count):
                seed = job.seed + index
                image = self._backend.render(job.prompt, job.width, job.height, seed)
                png = encode_png(
                    image, model=self._backend.model, revision=self._backend.revision, seed=seed
                )
                (partial / f"{index}.png").write_bytes(png)
            partial.rename(final)
        except Exception as exc:
            # Any image failure fails the whole job; already rendered images are discarded.
            log.exception("job %s failed at image %d", job.id, index)
            shutil.rmtree(partial, ignore_errors=True)
            error = f"image {index + 1} of {job.image_count} failed: {exc}"[:500]
            self._store.fail(job.id, error, self._run_record(started), self._clock())
            return
        if not self._store.succeed(job.id, self._run_record(started), self._clock()):
            # The job is no longer running (failed by a restart recovery); never publish it.
            shutil.rmtree(final, ignore_errors=True)

    def _run_record(self, started: float) -> dict[str, object]:
        finished = self._clock()
        return {
            "backend": self._backend.name,
            "model": self._backend.model,
            "revision": self._backend.revision,
            "render_seconds": round(finished - started, 3),
        }
