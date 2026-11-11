"""HTTP API: job submission and polling (API key required), signed image downloads (no key)."""

import math
import re
import secrets
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from chriek.auth import ApiKeyAuth
from chriek.backend import FakeBackend, ImageBackend
from chriek.config import Limits, Settings
from chriek.signing import UrlSigner
from chriek.store import SUCCEEDED, Job, JobStore
from chriek.worker import Worker, image_path

MAX_SEED = 2**32 - 1
WORKER_STOP_TIMEOUT_SECONDS = 30.0
INDEX_PATTERN = re.compile(r"[0-9]{1,6}")
EXPIRES_PATTERN = re.compile(r"[0-9]{1,12}")  # bounded: int() of huge digit strings raises
WORKER_STOPPED = "worker stopped: a job result could not be saved; restart chriek"


class JobRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str
    image_count: int = 1
    seed: int | None = None
    width: int = 1024
    height: int = 1024


def create_app(
    settings: Settings,
    *,
    backend: ImageBackend | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    """Build the home-mode app. `backend` and `clock` are test seams; `chriek serve` omits them."""
    storage_dir = settings.storage_dir.resolve()
    store = JobStore(settings.db_path)
    worker = Worker(store, backend or FakeBackend(), storage_dir, clock)
    signer = UrlSigner(settings.signing_secret)
    authenticate = ApiKeyAuth(settings.api_keys)
    Owner = Annotated[str, Depends(authenticate)]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        storage_dir.mkdir(parents=True, exist_ok=True)
        store.fail_interrupted(clock())
        worker.start()
        yield
        worker.stop(WORKER_STOP_TIMEOUT_SECONDS)
        store.close()

    app = FastAPI(title="chriek", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default handler echoes rejected input verbatim; JSON can't encode
        # nonfinite numbers (NaN, 1e999), lone surrogates or non-UTF-8 bytes (a raw
        # non-JSON body), which would turn a 422 into a 500.
        errors = jsonable_encoder(exc.errors(), custom_encoder={bytes: _decode_lossless})
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": _json_safe(errors)},
        )

    def job_body(job: Job, request: Request) -> dict[str, Any]:
        images = []
        if job.status == SUCCEEDED:
            expires = int(clock()) + settings.url_ttl_seconds
            base = str(request.base_url)
            for image in job.images:
                signature = signer.sign(uuid.UUID(job.id), image.index, expires)
                images.append(
                    {
                        "index": image.index,
                        "seed": image.seed,
                        "url": f"{base}images/{job.id}/{image.index}.png"
                        f"?expires={expires}&sig={signature}",
                        "expires_at": _iso(expires),
                    }
                )
        return {
            "id": job.id,
            "status": job.status,
            "params": {
                "prompt": job.prompt,
                "image_count": job.image_count,
                "seed": job.seed,
                "width": job.width,
                "height": job.height,
            },
            "created_at": _iso(job.created_at),
            "started_at": _iso(job.started_at),
            "finished_at": _iso(job.finished_at),
            "error": job.error,
            "run": job.run_record,
            "images": images,
        }

    @app.get("/health")
    def health(response: Response) -> dict[str, str]:
        if worker.stopped_on_error:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "error", "detail": WORKER_STOPPED}
        return {"status": "ok"}

    @app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_job(
        body: JobRequest, owner: Owner, request: Request, response: Response
    ) -> dict[str, Any]:
        errors = _limit_errors(body, settings.limits)
        if errors:
            raise RequestValidationError(errors)
        seed = body.seed
        if seed is None:
            seed = secrets.randbelow(MAX_SEED - body.image_count + 2)
        job = store.create(
            owner=owner,
            prompt=body.prompt,
            image_count=body.image_count,
            width=body.width,
            height=body.height,
            seed=seed,
            now=clock(),
        )
        worker.notify()
        response.headers["Location"] = f"/jobs/{job.id}"
        return job_body(job, request)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, owner: Owner, request: Request) -> dict[str, Any]:
        # Malformed IDs, missing jobs and other owners' jobs all get the same 404.
        job_uuid = _parse_uuid(job_id)
        job = store.get(owner, str(job_uuid)) if job_uuid else None
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        return job_body(job, request)

    @app.get("/images/{job_id}/{index}.png")
    def download_image(job_id: str, index: str, expires: str = "", sig: str = "") -> FileResponse:
        job_uuid = _parse_uuid(job_id)
        if job_uuid is None or not INDEX_PATTERN.fullmatch(index):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "image not found")
        image_index = int(index)
        valid = EXPIRES_PATTERN.fullmatch(expires) and signer.verify(
            job_uuid, image_index, int(expires), sig
        )
        if not valid:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid signature")
        if clock() >= int(expires):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "link expired; poll the job again")
        path = image_path(storage_dir, job_uuid, image_index)
        if not store.is_published(str(job_uuid), image_index) or not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "image not found")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    return app


def _limit_errors(body: JobRequest, limits: Limits) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def reject(field: str, message: str) -> None:
        value = getattr(body, field)
        errors.append(
            {"type": "value_error", "loc": ("body", field), "msg": message, "input": value}
        )

    if not _is_utf8(body.prompt):
        reject("prompt", "prompt must be valid Unicode text (no unpaired surrogates)")
    elif not body.prompt.strip():
        reject("prompt", "prompt must not be blank")
    elif len(body.prompt) > limits.max_prompt_chars:
        reject("prompt", f"prompt must be at most {limits.max_prompt_chars} characters")
    if not 1 <= body.image_count <= limits.max_images:
        reject("image_count", f"image_count must be between 1 and {limits.max_images}")
    for field in ("width", "height"):
        value = getattr(body, field)
        if value < 16 or value > limits.max_dimension or value % 16:
            reject(field, f"{field} must be a multiple of 16 between 16 and {limits.max_dimension}")
    if body.width * body.height > limits.max_pixels:
        reject("width", f"width * height must be at most {limits.max_pixels} pixels")
    if body.seed is not None and not 0 <= body.seed <= MAX_SEED - max(body.image_count - 1, 0):
        reject("seed", f"seed + image_count - 1 must be between 0 and {MAX_SEED}")
    return errors


def _parse_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _decode_lossless(data: bytes) -> str:
    return data.decode("utf-8", "backslashreplace")


def _json_safe(value: Any) -> Any:
    """Make decoded request data strictly JSON-serializable without dropping it."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, str):
        return value.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(value, dict):
        return {_json_safe(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _iso(timestamp: float | None) -> str | None:
    return None if timestamp is None else datetime.fromtimestamp(timestamp, UTC).isoformat()
