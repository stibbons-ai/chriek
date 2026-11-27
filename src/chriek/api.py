"""HTTP API: job submission and polling (authenticated), signed image downloads (no key)."""

import io
import json
import logging
import re
import secrets
import struct
import threading
import time
import uuid
import zlib
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header
from sqlalchemy.exc import OperationalError
from starlette.requests import ClientDisconnect
from starlette.types import Message

from chriek.auth import Authenticate, authenticator, worker_authenticator
from chriek.backend import ImageBackend, create_backend
from chriek.config import AuthMode, Limits, Settings, WorkerSettings
from chriek.dispatch import dispatch
from chriek.images import LocalImages, S3Images, image_path, open_images
from chriek.signing import UrlSigner
from chriek.storage import check_database
from chriek.store import (
    MEMORY_DATABASE,
    RUNNING,
    SUCCEEDED,
    CommitOutcomeUnknown,
    Job,
    JobStore,
)
from chriek.worker import Worker

log = logging.getLogger(__name__)
MAX_SEED = 2**32 - 1
WORKER_STOP_TIMEOUT_SECONDS = 30.0
INDEX_PATTERN = re.compile(r"[0-9]{1,6}")
EXPIRES_PATTERN = re.compile(r"[0-9]{1,12}")  # bounded: int() of huge digit strings raises
WORKER_STOPPED = "worker stopped: a job result could not be saved; restart chriek"
# Fixed wire budgets, not configuration: a 100,000-character prompt fits the body budget even
# fully JSON-escaped in UTF-8, and every 422 response fits the error budget whatever the request
# held. Multipart bodies carry reference images and are bounded by `max_upload_bytes` instead.
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_ERROR_BYTES = 16 * 1024
# Per-entry caps keep one worst-case 422 entry within MAX_ERROR_BYTES, so the first error always
# fits: type, msg and MAX_ERROR_LOC loc parts are each clipped to MAX_ERROR_TEXT characters, and a
# character JSON-encodes to at most 7 bytes (a lone surrogate becomes `\udXXX` text whose
# backslash is escaped again), plus 2 quotes per string, punctuation and the response envelope.
# Integer loc parts are not clipped; counting them as strings holds because the capped body only
# yields small ones (field indexes, JSON error offsets below MAX_BODY_BYTES), not giant integers.
MAX_ERROR_TEXT = 200
MAX_ERROR_LOC = 8
_WORST_ERROR_BYTES = (
    len(b'{"detail":[{"type":,"loc":[],"msg":}]}')
    + (MAX_ERROR_LOC - 1)  # commas between loc parts
    + (2 + MAX_ERROR_LOC) * (2 + 7 * MAX_ERROR_TEXT)
)
assert _WORST_ERROR_BYTES <= MAX_ERROR_BYTES, "a worst-case first 422 entry must fit"
BODY_TOO_LARGE = f"request body must be at most {MAX_BODY_BYTES} bytes"
MAX_SCALAR_FIELD_BYTES = 1024 * 1024
MAX_FORM_FIELDS = 32
INTEGER_PATTERN = re.compile(r"[+-]?[0-9]{1,20}")
ALLOWED_REFERENCE_FORMATS = ("PNG", "JPEG", "WEBP")
# Pillow's JPEG header parser runs in Python and keeps every APPn segment, and libjpeg
# decodes every progressive scan in full, so a small file can cost far more memory and CPU
# than its pixel count suggests. These bound the JPEG structure before Pillow sees it.
MAX_JPEG_SEGMENTS = 256
MAX_JPEG_SCANS = 64
MAX_JPEG_FILL_BYTES = 256
# Pillow decodes EXIF into Python objects at several times its size, and its JPEG header
# parser already does so for the resolution tags when JFIF gives no DPI, so EXIF payloads
# are capped before open.
MAX_EXIF_BYTES = 64 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Text chunk keywords Pillow's PNG reader turns into EXIF (`getexif` decodes the second
# from hex when no `exif` is present).
_PNG_EXIF_TEXT_KEYWORDS = (b"exif", b"Raw profile type exif")
# Pillow inflates every iCCP, zTXt and compressed iTXt chunk (each up to 1 MiB, its own
# per-chunk ceiling) and reads every other chunk into memory, including after IDAT, so a
# small PNG can repeat them far beyond its pixel cost. IDAT chunks of at least
# MIN_UNCOUNTED_IDAT_BYTES are pixel data, not counted; shorter ones (empty or split into
# tiny pieces) cost per-chunk work without pixel data, so they count.
MAX_PNG_CHUNKS = 256
MAX_PNG_COMPRESSED_CHUNKS = 8
MIN_UNCOUNTED_IDAT_BYTES = 1024
# libwebp's demuxer keeps a record per chunk and the walk below loops once per chunk, so
# empty chunks or frames cost work without pixel data; frames and their chunks count too.
MAX_WEBP_CHUNKS = 256
# A marker outside entropy-coded data: optional 0xFF fill bytes, then the marker code.
_JPEG_MARKER = re.compile(rb"\xff(\xff*+)([^\x00\xff])")
# End of entropy-coded scan data: the next marker that is neither stuffing (FF00), fill
# (FFFF) nor a restart marker (FFD0-FFD7); fill bytes before it are skipped by the search.
_JPEG_SCAN_END = re.compile(rb"\xff[^\x00\xd0-\xd7\xff]")
_EXIF_ORIENTATION = 0x0112
_ORIENTATION_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}
# Pillow raises these for corrupt input. AssertionError: Image.convert() on a palette PNG
# with tRNS but no PLTE; under `python -O` that assert is gone and AttributeError follows.
_DECODER_ERRORS = (
    UnidentifiedImageError,
    AssertionError,
    AttributeError,
    OSError,
    ValueError,
    SyntaxError,
    RuntimeError,
    struct.error,
    TypeError,
    KeyError,
    IndexError,
    ZeroDivisionError,
    OverflowError,
    EOFError,
    zlib.error,
)


class JobRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str
    image_count: int = 1
    seed: int | None = None
    width: int = 1024
    height: int = 1024


SCALAR_FIELDS = tuple(JobRequest.model_fields)
_JOB_REQUEST_ADAPTER = TypeAdapter(JobRequest)
_JOB_REQUEST_SCHEMA = JobRequest.model_json_schema(ref_template="#/components/schemas/{model}")
_JOBS_POST_OPENAPI: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {"schema": {"$ref": "#/components/schemas/JobRequest"}},
            "multipart/form-data": {
                "schema": {
                    **_JOB_REQUEST_SCHEMA,
                    "title": "Body_submit_job_jobs_post",
                    "properties": {
                        **_JOB_REQUEST_SCHEMA["properties"],
                        # A form field cannot be null: omit `seed` or send it empty for a
                        # random seed.
                        "seed": {
                            "anyOf": [{"type": "integer"}, {"type": "string", "maxLength": 0}],
                            "title": "Seed",
                        },
                        "references": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                            "title": "References",
                        },
                    },
                }
            },
        },
    },
    "responses": {
        "422": {
            "description": "Validation Error",
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}
            },
        }
    },
}


def create_app(
    settings: Settings,
    *,
    backend: ImageBackend | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    """Build the home-mode app (`chriek serve`), or the serverless API (`chriek api`) when
    `settings.dispatch` is set. The serverless API constructs no model, runs no worker and
    recovers no running jobs: on the shared database those belong to workers elsewhere.
    `backend` and `clock` are test seams; the commands omit them."""
    if backend is None and settings.dispatch is None:
        backend = create_backend(settings.backend)
    # Both storage sides are checked before either is created: the database side here without
    # creating anything, then the image side by open_images (a local directory is checked and
    # created, a bucket must exist). JobStore checks its own side again, so it stays safe when
    # used without the app. A Postgres URL has no local database side.
    if isinstance(settings.database, Path) and str(settings.database) != MEMORY_DATABASE:
        check_database(settings.database)
    images = open_images(settings.storage, settings.url_ttl_seconds)
    store = JobStore(settings.database)
    worker = None
    if settings.dispatch is None:
        try:
            worker = Worker(store, backend, images, clock, settings.job_timeout_seconds)
        except BaseException:
            store.close()
            raise
    signer = UrlSigner(settings.signing_secret)
    jobs = APIRouter(
        route_class=_authenticated_route(authenticator(settings.auth_mode, settings.api_keys))
    )
    Owner = Annotated[str, Depends(_owner)]

    def worker_stopped() -> bool:
        return worker is not None and worker.stopped_on_error

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if worker is not None:
            store.fail_interrupted(clock())
            worker.start()
        yield
        if worker is not None:
            worker.stop(WORKER_STOP_TIMEOUT_SECONDS)
        store.close()

    app = FastAPI(title="chriek", lifespan=lifespan)
    if settings.auth_mode is AuthMode.PROXY:
        # Proxy mode only: with no authentication, a Host check is what stops DNS rebinding
        # from letting a browser page drive chriek. Missing or malformed Host also gets 400.
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts, www_redirect=False
        )

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
            schemas = schema.setdefault("components", {}).setdefault("schemas", {})
            schemas["JobRequest"] = _JOB_REQUEST_SCHEMA
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> Response:
        return Response(
            _error_detail(exc.errors()),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            media_type="application/json",
        )

    def image_link(job_id: str, index: int) -> tuple[str, int]:
        if isinstance(images, S3Images):
            return images.presigned_url(job_id, index)  # direct from the bucket, never via chriek
        expires = int(clock()) + settings.url_ttl_seconds
        signature = signer.sign(uuid.UUID(job_id), index, expires)
        url = f"{settings.public_url}/images/{job_id}/{index}.png?expires={expires}&sig={signature}"
        return url, expires

    def job_body(job: Job) -> dict[str, Any]:
        links = []
        if job.status == SUCCEEDED:
            for image in job.images:
                url, expires = image_link(job.id, image.index)
                links.append(
                    {
                        "index": image.index,
                        "seed": image.seed,
                        "url": url,
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
                "reference_count": job.reference_count,
            },
            "created_at": _iso(job.created_at),
            "started_at": _iso(job.started_at),
            "finished_at": _iso(job.finished_at),
            "error": job.error,
            "run": job.run_record,
            "images": links,
        }

    @app.get("/health")
    def health(response: Response) -> dict[str, str]:
        if worker_stopped():
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "error", "detail": WORKER_STOPPED}
        return {"status": "ok"}

    @jobs.post("/jobs", status_code=status.HTTP_202_ACCEPTED, openapi_extra=_JOBS_POST_OPENAPI)
    async def submit_job(
        owner: Owner, request: Request, response: Response, background: BackgroundTasks
    ) -> dict[str, Any]:
        complete_submission = await _stream_submission(request, settings.limits)

        def decode_and_persist() -> Job:
            body, references = complete_submission()
            # Observed-state check only: a result write failing after this point still admits.
            if worker_stopped():
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, WORKER_STOPPED)
            seed = body.seed
            if seed is None:
                seed = secrets.randbelow(MAX_SEED - body.image_count + 2)
            job_id = str(uuid.uuid7())
            try:
                images.put_references(job_id, references)
                job = store.create(
                    job_id=job_id,
                    owner=owner,
                    prompt=body.prompt,
                    image_count=body.image_count,
                    width=body.width,
                    height=body.height,
                    seed=seed,
                    reference_count=len(references),
                    now=clock(),
                    max_queued=settings.limits.max_queued_jobs,
                )
                if job is None:  # rejected: the cleanup below removes this upload's references
                    raise HTTPException(
                        status.HTTP_429_TOO_MANY_REQUESTS,
                        f"queue full: at most {settings.limits.max_queued_jobs} queued jobs "
                        "per owner; new jobs are accepted only while fewer are queued",
                    )
            except CommitOutcomeUnknown:
                # The job may be committed: keep its references, at worst as an orphan.
                raise
            except BaseException:
                images.remove_submission(job_id, len(references))
                raise
            if worker is not None:
                worker.notify()
            return job

        job = await run_in_threadpool(decode_and_persist)
        if settings.dispatch is not None:
            # Only now, with the job and its references committed, may a worker hear of it.
            # After the response: delivery never delays or changes the 202, but it holds a
            # shared threadpool thread until the endpoint answers or the dispatch timeout.
            background.add_task(dispatch, settings.dispatch, store, job.id, clock)
        response.headers["Location"] = f"/jobs/{job.id}"
        return job_body(job)

    @jobs.get("/jobs/{job_id}")
    def get_job(job_id: str, owner: Owner) -> dict[str, Any]:
        # Malformed IDs, missing jobs and other owners' jobs all get the same 404.
        job_uuid = _parse_uuid(job_id)
        job = store.get(owner, str(job_uuid)) if job_uuid else None
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        now = clock()
        if job.status == RUNNING and job.started_at < now - settings.job_timeout_seconds:
            # A stuck render can't finish the job; the poll fails it. The conditional update
            # loses to a result saved meanwhile, so the job is read again after it.
            try:
                store.fail_timed_out(job.id, settings.job_timeout_seconds, now)
            except OperationalError:
                # The row was readable: report it as read; a later poll retries the write.
                log.exception("could not time out job %s", job.id)
            else:
                job = store.get(owner, job.id)
        return job_body(job)

    # S3 links point at the bucket: chriek serves image bytes only from local storage.
    if isinstance(images, LocalImages):
        root = images.root

        @app.get("/images/{job_id}/{index}.png")
        def download_image(
            job_id: str, index: str, expires: str = "", sig: str = ""
        ) -> FileResponse:
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
            path = image_path(root, str(job_uuid), image_index)
            if not store.is_published(str(job_uuid), image_index) or not path.is_file():
                raise HTTPException(status.HTTP_404_NOT_FOUND, "image not found")
            return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    app.include_router(jobs)
    return app


class Invocation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    job_id: str


def create_worker_app(
    settings: WorkerSettings,
    *,
    backend: ImageBackend | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    """Build `chriek worker`: each `POST /invoke` runs the one queued job it names, from the
    shared database and bucket, and answers with the job's status. `backend` and `clock` are
    test seams; the command omits them."""
    if backend is None:
        backend = create_backend(settings.backend)
    images = open_images(settings.storage, 1)  # the worker presigns no download links
    store = JobStore(settings.database)
    worker = Worker(store, backend, images, clock, settings.job_timeout_seconds)
    # One render at a time per process: a second invocation waits, then reports the state its
    # job reached, so a duplicate delivery returns the first delivery's outcome.
    rendering = threading.Lock()
    invocations = APIRouter(
        route_class=_authenticated_route(worker_authenticator(settings.worker_token))
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # No startup recovery: other workers' running jobs share this database.
        yield
        store.close()

    app = FastAPI(title="chriek worker", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @invocations.post("/invoke")
    def invoke(invocation: Invocation) -> dict[str, str]:
        job_uuid = _parse_uuid(invocation.job_id)
        if job_uuid is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        job_id = str(job_uuid)
        with rendering:
            # Only a queued job is claimed: duplicates and failed dispatches render nothing.
            job = store.claim(job_id, clock())
            if job is not None:
                try:
                    worker.run(job)
                except Exception:
                    log.exception("could not record the result of job %s", job_id)
                    raise HTTPException(
                        status.HTTP_500_INTERNAL_SERVER_ERROR, "job result could not be saved"
                    ) from None
            job_status = store.status(job_id)
        if job_status is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        return {"id": job_id, "status": job_status}

    app.include_router(invocations)
    return app


def _authenticated_route(authenticate: Authenticate) -> type[APIRoute]:
    """Route class that authenticates, then caps bodies other than multipart `POST /jobs`
    uploads, before the handler runs.

    `/jobs` streams and parses its body itself, so authenticating first means an
    unauthenticated client gets 401 without chriek reading, parsing or storing any upload.
    """

    class AuthenticatedRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Any]:
            handler = super().get_route_handler()

            async def authenticated(request: Request) -> Response:
                request.state.owner = authenticate(request)
                content_type, _ = parse_options_header(request.headers.get("content-type", ""))
                if request.method == "POST" and content_type == b"multipart/form-data":
                    # Reference uploads: the multipart parser enforces `max_upload_bytes` (422).
                    return await handler(request)
                try:
                    declared = int(request.headers.get("content-length", 0))
                except ValueError:
                    declared = 0  # malformed header: the received-byte count still applies
                if declared > MAX_BODY_BYTES:
                    raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, BODY_TOO_LARGE)
                received = 0

                async def capped_receive() -> Message:
                    # Raising here drops the overflowing chunk before Starlette's Request.body()
                    # buffers it; FastAPI re-raises HTTPException from body reading unchanged.
                    nonlocal received
                    message = await request.receive()
                    received += len(message.get("body", b""))
                    if received > MAX_BODY_BYTES:
                        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, BODY_TOO_LARGE)
                    return message

                # Shares the scope, so the owner set in request.state above stays visible.
                return await handler(Request(request.scope, capped_receive))

            return authenticated

    return AuthenticatedRoute


def _owner(request: Request) -> str:
    """Owner name set by the authenticated route before this dependency runs."""
    return request.state.owner


def _limit_errors(body: JobRequest, limits: Limits) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def reject(field: str, message: str) -> None:
        errors.append({"type": "value_error", "loc": ("body", field), "msg": message})

    if not _is_utf8(body.prompt):
        reject("prompt", "prompt must be valid Unicode text (no unpaired surrogates)")
    elif "\x00" in body.prompt:
        # Postgres text can't hold U+0000; rejecting it keeps both databases equivalent.
        reject("prompt", "prompt must not contain NUL characters (U+0000)")
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


async def _stream_submission(
    request: Request, limits: Limits
) -> Callable[[], tuple[JobRequest, list[bytes]]]:
    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    if content_type == b"multipart/form-data":
        acc = await _stream_multipart_job(request, params, limits)
        return lambda: _finish_multipart_job(acc, limits)
    body_bytes = await _read_body(request)
    return lambda: (_decode_json_job(body_bytes, content_type, limits), [])


def _decode_json_job(body_bytes: bytes, content_type: bytes, limits: Limits) -> JobRequest:
    if not body_bytes:
        raise RequestValidationError(
            [{"type": "missing", "loc": ("body",), "msg": "Field required", "input": None}]
        )
    is_json = content_type == b"application/json" or (
        content_type.startswith(b"application/") and content_type.endswith(b"+json")
    )
    if is_json:
        try:
            raw: Any = json.loads(body_bytes)
        except json.JSONDecodeError as exc:
            raise RequestValidationError(
                [
                    {
                        "type": "json_invalid",
                        "loc": ("body", exc.pos),
                        "msg": "JSON decode error",
                        "input": {},
                        "ctx": {"error": exc.msg},
                    }
                ]
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "There was an error parsing the body"
            ) from exc
    else:
        raw = body_bytes
    try:
        body = _JOB_REQUEST_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        errors = [{**err, "loc": ("body", *err["loc"])} for err in exc.errors(include_url=False)]
        raise RequestValidationError(errors) from exc
    errors = _limit_errors(body, limits)
    if errors:
        raise RequestValidationError(errors)
    return body


async def _read_body(request: Request) -> bytes:
    """The non-multipart body; the authenticated route caps it at MAX_BODY_BYTES (413)."""
    try:
        return await request.body()
    except ClientDisconnect as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "There was an error parsing the body"
        ) from exc


class _MultipartAccumulator:
    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.fields: list[tuple[str, bytes]] = []
        self.files: list[bytes] = []
        self.ended = False
        self._header_name = bytearray()
        self._header_value = bytearray()
        self._content_disposition: bytes | None = None
        self._part_name = ""
        self._is_file = False
        self._part_data = bytearray()

    def on_part_begin(self) -> None:
        self._header_name.clear()
        self._header_value.clear()
        self._content_disposition = None
        self._part_name = ""
        self._is_file = False
        self._part_data.clear()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name.extend(data[start:end])

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value.extend(data[start:end])

    def on_header_end(self) -> None:
        if self._header_name.lower() == b"content-disposition":
            self._content_disposition = bytes(self._header_value)
        self._header_name.clear()
        self._header_value.clear()

    def on_headers_finished(self) -> None:
        if self._content_disposition is None:
            raise _validation_error(("body",), "multipart part is missing Content-Disposition")
        _, options = parse_options_header(self._content_disposition)
        raw_name = options.get(b"name")
        if raw_name is None:
            raise _validation_error(
                ("body",), 'multipart part Content-Disposition must include a "name" parameter'
            )
        try:
            self._part_name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _validation_error(("body",), "multipart field name must be valid UTF-8") from exc
        # A `references` part is an upload whether or not it carries the optional filename
        # (RFC 7578); image decoding, not the part's headers, decides whether it is valid.
        self._is_file = self._part_name == "references"
        if self._is_file:
            if len(self.files) + 1 > self.limits.max_references:
                raise _validation_error(
                    ("body", "references"),
                    f"at most {self.limits.max_references} reference images are allowed",
                )
        elif b"filename" in options:
            raise _validation_error(
                ("body", self._part_name),
                f"unexpected file field {self._part_name!r}; reference files must use "
                "field 'references'",
            )
        elif len(self.fields) + 1 > MAX_FORM_FIELDS:
            raise _validation_error(("body",), "too many form fields")

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if not self._is_file and len(self._part_data) + (end - start) > MAX_SCALAR_FIELD_BYTES:
            raise _validation_error(
                ("body", self._part_name), f"field {self._part_name!r} is too large"
            )
        self._part_data.extend(memoryview(data)[start:end])

    def on_part_end(self) -> None:
        payload = bytes(self._part_data)
        if self._is_file:
            self.files.append(payload)
        else:
            self.fields.append((self._part_name, payload))
        self._part_data.clear()

    def on_end(self) -> None:
        self.ended = True


async def _stream_multipart_job(
    request: Request, params: dict[bytes, bytes], limits: Limits
) -> _MultipartAccumulator:
    boundary = params.get(b"boundary")
    if not boundary:
        raise _validation_error(("body",), "missing boundary in multipart/form-data")
    acc = _MultipartAccumulator(limits)
    callbacks = {
        "on_part_begin": acc.on_part_begin,
        "on_header_field": acc.on_header_field,
        "on_header_value": acc.on_header_value,
        "on_header_end": acc.on_header_end,
        "on_headers_finished": acc.on_headers_finished,
        "on_part_data": acc.on_part_data,
        "on_part_end": acc.on_part_end,
        "on_end": acc.on_end,
    }
    try:
        parser = MultipartParser(boundary, callbacks)
    except (FormParserError, ValueError) as exc:
        raise _validation_error(("body",), "invalid multipart boundary") from exc

    total_bytes = 0
    try:
        async for chunk in request.stream():
            total_bytes += len(chunk)
            if total_bytes > limits.max_upload_bytes:
                raise _validation_error(
                    ("body",),
                    f"upload exceeds maximum size of {limits.max_upload_bytes} bytes",
                )
            parser.write(chunk)
        parser.finalize()
    except (FormParserError, ClientDisconnect) as exc:
        raise _validation_error(("body",), "invalid multipart/form-data body") from exc

    if not acc.ended:
        raise _validation_error(("body",), "incomplete multipart/form-data body")
    return acc


def _finish_multipart_job(
    acc: _MultipartAccumulator, limits: Limits
) -> tuple[JobRequest, list[bytes]]:
    body, errors = _build_multipart_job_request(acc.fields)
    if body is not None:
        errors.extend(_limit_errors(body, limits))
    references: list[bytes] = []
    for index, data in enumerate(acc.files):
        png_bytes, error = _decode_reference_png(index, data, limits)
        if error is not None:
            errors.append(error)
        elif png_bytes is not None:
            references.append(png_bytes)
    if errors or body is None:
        raise RequestValidationError(errors)
    return body, references


def _build_multipart_job_request(
    fields: list[tuple[str, bytes]],
) -> tuple[JobRequest | None, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    parsed: dict[str, Any] = {}
    for name, raw_bytes in fields:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            seen.add(name)
            errors.append(
                {
                    "type": "value_error",
                    "loc": ("body", name),
                    "msg": f"{name} must be valid UTF-8 text",
                    "input": raw_bytes,
                }
            )
            continue
        if name not in SCALAR_FIELDS:
            errors.append(
                {
                    "type": "extra_forbidden",
                    "loc": ("body", name),
                    "msg": "Extra inputs are not permitted",
                    "input": text,
                }
            )
            continue
        if name in seen:
            errors.append(
                {
                    "type": "value_error",
                    "loc": ("body", name),
                    "msg": f"duplicate form field {name!r}",
                    "input": text,
                }
            )
            continue
        seen.add(name)
        if name == "prompt":
            parsed["prompt"] = text
        elif name == "seed" and text == "":
            continue
        elif not INTEGER_PATTERN.fullmatch(text):
            errors.append(
                {
                    "type": "int_parsing",
                    "loc": ("body", name),
                    "msg": "Input should be a valid integer",
                    "input": text,
                }
            )
        else:
            parsed[name] = int(text)

    if "prompt" not in seen:
        errors.append(
            {
                "type": "missing",
                "loc": ("body", "prompt"),
                "msg": "Field required",
                "input": None,
            }
        )
    if errors:
        return None, errors
    return JobRequest(**parsed), []


def _decode_reference_png(
    index: int, data: bytes, limits: Limits
) -> tuple[bytes | None, dict[str, Any] | None]:
    loc = ("body", "references", index)
    if not data:
        return None, {
            "type": "value_error",
            "loc": loc,
            "msg": f"reference image {index} is empty",
            "input": None,
        }
    try:
        try:
            _check_before_decoding(data)
        except _RejectedBeforeDecoding:
            raise
        except Exception:
            # A bug in chriek's own walk, not bad input: log it, answer a generic 500.
            log.exception("reference image %d: structure check failed unexpectedly", index)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal error") from None
        with Image.open(io.BytesIO(data), formats=ALLOWED_REFERENCE_FORMATS) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > limits.max_pixels:
                raise Image.DecompressionBombError
            image.load()
            # getexif() decodes the EXIF capped above (falling back to an XMP
            # tiff:Orientation when EXIF has none); only the orientation is applied.
            # ImageOps.exif_transpose would also re-serialise the whole EXIF block.
            method = _ORIENTATION_TRANSPOSE.get(image.getexif().get(_EXIF_ORIENTATION))
            if method is not None:
                image = image.transpose(method)
            if image.mode.startswith("I;16"):
                # convert() clips 16-bit samples at 255; scale them to 8 bits first.
                image = image.point(lambda value: value / 256).convert("L")
            rgb = image if image.mode == "RGB" else image.convert("RGB")
            # Drop all source metadata; the PNG writer would re-emit `icc_profile` and
            # `transparency` from `info`. EXIF (with its orientation, already applied
            # above) and XMP are never written, so nothing rotates twice.
            # No colour management happens: pixel values pass through, and the profile that
            # described them is dropped (Pillow's PNG reader also refuses profiles whose
            # decompressed size exceeds 1 MiB). Without `transparency` no tRNS chunk is
            # written for the opaque RGB result; a 1-bit source's integer key would even
            # make the writer raise.
            rgb.info.clear()
            buf = io.BytesIO()
            rgb.save(buf, format="PNG")
    except _JpegTooComplex:
        return None, {
            "type": "value_error",
            "loc": loc,
            "msg": (
                f"reference image {index} has too complex a JPEG structure (at most "
                f"{MAX_JPEG_SEGMENTS} marker segments, {MAX_JPEG_SCANS} scans and "
                f"{MAX_JPEG_FILL_BYTES} fill bytes before markers outside scan data)"
            ),
            "input": None,
        }
    except _PngTooComplex:
        return None, {
            "type": "value_error",
            "loc": loc,
            "msg": (
                f"reference image {index} has too many PNG chunks (at most {MAX_PNG_CHUNKS} "
                f"chunks other than IDAT chunks of at least {MIN_UNCOUNTED_IDAT_BYTES} bytes, "
                f"of them at most {MAX_PNG_COMPRESSED_CHUNKS} iCCP, zTXt or compressed iTXt)"
            ),
            "input": None,
        }
    except _WebpTooComplex:
        return None, {
            "type": "value_error",
            "loc": loc,
            "msg": (
                f"reference image {index} has too many WebP chunks (at most {MAX_WEBP_CHUNKS}, "
                "counting each ANMF animation frame and every chunk inside it)"
            ),
            "input": None,
        }
    except _ExifTooLarge:
        return None, {
            "type": "value_error",
            "loc": loc,
            "msg": (
                f"reference image {index} has too much EXIF metadata "
                f"(at most {MAX_EXIF_BYTES} bytes)"
            ),
            "input": None,
        }
    except Image.DecompressionBombError, Image.DecompressionBombWarning:
        return None, {
            "type": "value_error",
            "loc": loc,
            "msg": f"reference image {index} exceeds maximum of {limits.max_pixels} pixels",
            "input": None,
        }
    except (_MalformedImage, *_DECODER_ERRORS):
        return None, {
            "type": "value_error",
            "loc": loc,
            "msg": (
                f"reference image {index} is not a valid image (supported formats: PNG, JPEG, WebP)"
            ),
            "input": None,
        }
    return buf.getvalue(), None


class _RejectedBeforeDecoding(Exception):
    """An upload the structure walk rejects; anything else it raises is a bug."""


class _JpegTooComplex(_RejectedBeforeDecoding):
    pass


class _PngTooComplex(_RejectedBeforeDecoding):
    pass


class _WebpTooComplex(_RejectedBeforeDecoding):
    pass


class _ExifTooLarge(_RejectedBeforeDecoding):
    pass


class _MalformedImage(_RejectedBeforeDecoding):
    """Structure Pillow or libjpeg would have to skip or guess at."""


def _check_before_decoding(data: bytes) -> None:
    """Reject uploads whose structure or EXIF would be expensive, before Pillow opens them.

    The signature checks accept everything Pillow's PNG, JPEG (and MPO) and WebP plugins
    accept, and slightly more.
    """
    if data.startswith(b"\xff\xd8\xff"):
        _check_jpeg_structure(data)
    elif data.startswith(_PNG_SIGNATURE):
        _check_png_structure(data)
    elif data.startswith(b"RIFF") and data.startswith(b"WEBP", 8):
        _check_webp_structure(data)


def _check_jpeg_structure(data: bytes) -> None:
    """Walk JPEG markers up to the first EOI, without decoding anything.

    Raises `_JpegTooComplex` above the segment, scan or fill-byte bounds, `_ExifTooLarge`
    when the APP1 `Exif` payloads (after their 6-byte `Exif\\0\\0` headers, which Pillow
    concatenates) total more than `MAX_EXIF_BYTES`, and `_MalformedImage` for structure Pillow
    or libjpeg would have to skip or guess at: bytes other than a marker between segments,
    a nested SOI, a length below 2, an EOI before the first SOS (Pillow's header parser
    reads on past it), a reserved JPG or JPGn marker (Pillow reads no length for it and
    parses its payload as markers), or a file that ends before EOI. Fill bytes count only
    before markers found by the walk, including after a bare RST or TEM outside scan data;
    those after scan data are skipped with it. Entropy-coded data after each SOS is skipped
    with one regex search, so the walk costs C-speed time linear in the upload plus at most
    `MAX_JPEG_SEGMENTS` iterations. Data after the first EOI (further MPO pictures) may be
    buffered by Pillow but is never parsed or decoded, and not walked.
    """
    pos = 2  # after SOI
    segments = scans = fill = exif = 0
    while True:
        marker = _JPEG_MARKER.match(data, pos)
        if marker is None:
            raise _MalformedImage("JPEG marker expected")
        fill += len(marker.group(1))
        if fill > MAX_JPEG_FILL_BYTES:
            raise _JpegTooComplex
        code = marker.group(2)[0]
        pos = marker.end()
        if code == 0xD9:  # EOI
            if scans == 0:
                raise _MalformedImage("JPEG EOI before the first scan")
            return
        if code == 0xD8:  # SOI
            raise _MalformedImage("nested JPEG SOI")
        segments += 1
        if segments > MAX_JPEG_SEGMENTS:
            raise _JpegTooComplex
        if 0xD0 <= code <= 0xD7 or code == 0x01:  # RSTn and TEM carry no length
            continue
        if code == 0xC8 or 0xF0 <= code <= 0xFD:
            # Reserved JPG and JPGn: Pillow reads no length and parses their payload as markers.
            raise _MalformedImage("reserved JPEG marker")
        length = int.from_bytes(data[pos : pos + 2])
        if length < 2 or pos + length > len(data):
            raise _MalformedImage("truncated JPEG segment")
        if code == 0xE1 and data.startswith(b"Exif\x00\x00", pos + 2, pos + length):
            exif += length - 8
            if exif > MAX_EXIF_BYTES:
                raise _ExifTooLarge
        pos += length
        if code == 0xDA:  # SOS: entropy-coded data follows
            scans += 1
            if scans > MAX_JPEG_SCANS:
                raise _JpegTooComplex
            scan_end = _JPEG_SCAN_END.search(data, pos)
            if scan_end is None:
                raise _MalformedImage("truncated JPEG scan")
            pos = scan_end.start()


def _check_png_structure(data: bytes) -> None:
    """Walk PNG chunks up to IEND, inflating only EXIF text and only up to the EXIF budget.

    Raises `_PngTooComplex` above `MAX_PNG_CHUNKS` chunks other than IDAT chunks of at least
    `MIN_UNCOUNTED_IDAT_BYTES` bytes, or above `MAX_PNG_COMPRESSED_CHUNKS` iCCP, zTXt or
    iTXt chunks with a nonzero compression flag (counted whatever their keyword or text, as
    Pillow inflates them first), and
    `_ExifTooLarge` when the EXIF payloads total more than `MAX_EXIF_BYTES`: the data of every
    `eXIf` chunk (Pillow keeps only the last one) and the text of every tEXt, zTXt or iTXt
    chunk with an EXIF keyword, decompressed for zTXt and compressed iTXt. A chunk cut off
    by the end of the file counts what is there; the walk stops at it.
    """
    pos, chunks, compressed, exif = len(_PNG_SIGNATURE), 0, 0, 0
    while pos + 8 <= len(data):
        start = pos + 8
        end = min(start + int.from_bytes(data[pos : pos + 4]), len(data))
        kind = data[pos + 4 : start]
        if kind != b"IDAT" or end - start < MIN_UNCOUNTED_IDAT_BYTES:
            chunks += 1
            if kind in (b"iCCP", b"zTXt") or (
                kind == b"iTXt" and _png_itxt_zipped(data, start, end)
            ):
                compressed += 1
            if chunks > MAX_PNG_CHUNKS or compressed > MAX_PNG_COMPRESSED_CHUNKS:
                raise _PngTooComplex
        if kind == b"eXIf":
            exif += end - start
        elif kind in (b"tEXt", b"zTXt", b"iTXt"):
            exif += _png_exif_text_size(kind, data, start, end, MAX_EXIF_BYTES - exif)
        if exif > MAX_EXIF_BYTES:
            raise _ExifTooLarge
        if kind == b"IEND":
            return
        pos = end + 4  # CRC


def _png_itxt_zipped(data: bytes, start: int, end: int) -> bool:
    """Whether an iTXt chunk's compression flag, the byte after its keyword, is nonzero."""
    flag = data.find(b"\x00", start, end) + 1
    return 0 < flag < end and data[flag] != 0


def _png_exif_text_size(kind: bytes, data: bytes, start: int, end: int, budget: int) -> int:
    """Size of a PNG text chunk's text if its keyword makes it EXIF, else 0.

    Compressed text is inflated to at most `budget + 1` bytes, enough to tell it is over.
    """
    keyword_end = data.find(b"\x00", start, end)
    if not any(
        keyword_end - start == len(keyword) and data.startswith(keyword, start)
        for keyword in _PNG_EXIF_TEXT_KEYWORDS
    ):
        return 0
    text = keyword_end + 1
    if kind == b"tEXt":
        return end - text
    if kind == b"zTXt":
        compressed = True
        text += 1  # compression method
    else:  # iTXt: compression flag and method, language tag, translated keyword
        # Pillow inflates the text for any nonzero compression flag.
        compressed = not data.startswith(b"\x00", text, end)
        language_end = data.find(b"\x00", text + 2, end)
        if language_end < 0:
            return 0
        translated_end = data.find(b"\x00", language_end + 1, end)
        if translated_end < 0:
            return 0
        text = translated_end + 1
    if not compressed:
        return end - text
    try:
        return len(zlib.decompressobj().decompress(memoryview(data)[text:end], budget + 1))
    except zlib.error:
        return 0


def _check_webp_structure(data: bytes) -> None:
    """Walk WebP chunks, top-level and inside `ANMF` frames, without decoding anything.

    Raises `_WebpTooComplex` above `MAX_WEBP_CHUNKS` chunks, counting every top-level chunk,
    every `ANMF` frame and every chunk inside one, and `_ExifTooLarge` when the `EXIF` chunks
    total more than `MAX_EXIF_BYTES`. libwebp reads on after a frame's image data as if at
    the top level, so an `EXIF` there is the image's EXIF too. A chunk cut off by the end of
    the file counts what is there.
    """
    pos, chunks, exif = 12, 0, 0  # after RIFF, size, WEBP
    while pos + 8 <= len(data):
        chunks += 1
        if chunks > MAX_WEBP_CHUNKS:
            raise _WebpTooComplex
        if data.startswith(b"ANMF", pos):
            pos += 8 + 16  # chunk header, frame header; then the frame's chunks
            continue
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        if data.startswith(b"EXIF", pos):
            exif += min(size, len(data) - pos - 8)
            if exif > MAX_EXIF_BYTES:
                raise _ExifTooLarge
        pos += 8 + size + (size & 1)  # chunks are padded to even sizes


def _validation_error(
    loc: tuple[str | int, ...], msg: str, input_val: Any = None
) -> RequestValidationError:
    return RequestValidationError(
        [{"type": "value_error", "loc": loc, "msg": msg, "input": input_val}]
    )


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


def _error_detail(errors: Sequence[Any]) -> bytes:
    """Encode `{"detail": [...]}` within MAX_ERROR_BYTES, keeping each error's type, loc and msg.

    Rejected input and validation ctx are never echoed; text is clipped and trailing errors that
    no longer fit are dropped. The counted bytes are the bytes sent, so the budget is exact.
    """
    entries: list[bytes] = []
    size = len(b'{"detail":[]}')
    for error in errors:
        loc = [part if type(part) is int else _clip(part) for part in error["loc"][:MAX_ERROR_LOC]]
        entry = {"type": _clip(error["type"]), "loc": loc, "msg": _clip(error["msg"])}
        encoded = json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode()
        size += len(encoded) + bool(entries)
        if size > MAX_ERROR_BYTES:
            break
        entries.append(encoded)
    return b'{"detail":[' + b",".join(entries) + b"]}"


def _clip(value: Any) -> str:
    """First MAX_ERROR_TEXT characters; unpaired surrogates then become (longer) escape text."""
    return str(value)[:MAX_ERROR_TEXT].encode("utf-8", "backslashreplace").decode("utf-8")


def _iso(timestamp: float | None) -> str | None:
    return None if timestamp is None else datetime.fromtimestamp(timestamp, UTC).isoformat()
