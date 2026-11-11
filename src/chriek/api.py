"""HTTP API: job submission and polling (API key required), signed image downloads (no key)."""

import io
import json
import math
import re
import secrets
import shutil
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.requests import ClientDisconnect

from chriek.auth import ApiKeyAuth
from chriek.backend import FakeBackend, ImageBackend
from chriek.config import Limits, Settings
from chriek.signing import UrlSigner
from chriek.store import SUCCEEDED, Job, JobStore
from chriek.worker import Worker, image_path, write_references

MAX_SEED = 2**32 - 1
WORKER_STOP_TIMEOUT_SECONDS = 30.0
INDEX_PATTERN = re.compile(r"[0-9]{1,6}")
EXPIRES_PATTERN = re.compile(r"[0-9]{1,12}")  # bounded: int() of huge digit strings raises
WORKER_STOPPED = "worker stopped: a job result could not be saved; restart chriek"
# Rejected input echoed in 422 errors is cut below this nesting depth: JSON encoders
# recurse once per level, so a small, deeply nested body would otherwise overflow the stack.
MAX_ECHO_DEPTH = 32
TOO_DEEP = "<nested too deeply>"
MAX_SCALAR_FIELD_BYTES = 1024 * 1024
MAX_FORM_FIELDS = 32
INTEGER_PATTERN = re.compile(r"[+-]?[0-9]{1,20}")


class JobRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str
    image_count: int = 1
    seed: int | None = None
    width: int = 1024
    height: int = 1024


SCALAR_FIELDS = tuple(JobRequest.model_fields)
_JOB_REQUEST_ADAPTER = TypeAdapter(JobRequest)


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
    jobs = APIRouter(route_class=_authenticated_route(ApiKeyAuth(settings.api_keys)))
    Owner = Annotated[str, Depends(_owner)]

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
        errors = jsonable_encoder(
            _depth_limited(exc.errors(), MAX_ECHO_DEPTH), custom_encoder={bytes: _decode_lossless}
        )
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
                "reference_count": job.reference_count,
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

    @jobs.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    async def submit_job(owner: Owner, request: Request, response: Response) -> dict[str, Any]:
        body, references = await _parse_submission(request, settings.limits)
        seed = body.seed
        if seed is None:
            seed = secrets.randbelow(MAX_SEED - body.image_count + 2)
        job_id = str(uuid.uuid7())
        try:
            write_references(storage_dir, job_id, references)
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
            )
        except BaseException:
            shutil.rmtree(storage_dir / job_id, ignore_errors=True)
            raise
        worker.notify()
        response.headers["Location"] = f"/jobs/{job.id}"
        return job_body(job, request)

    @jobs.get("/jobs/{job_id}")
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

    app.include_router(jobs)
    return app


def _authenticated_route(authenticate: ApiKeyAuth) -> type[APIRoute]:
    """Route class that authenticates before FastAPI reads the body or resolves parameters.

    FastAPI parses the request body before running dependencies, so a key check as a
    dependency would answer malformed bodies with 400/422 instead of 401.
    """

    class AuthenticatedRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Any]:
            handler = super().get_route_handler()

            async def authenticated(request: Request) -> Response:
                request.state.owner = authenticate(request)
                return await handler(request)

            return authenticated

    return AuthenticatedRoute


def _owner(request: Request) -> str:
    """Owner name set by the authenticated route before this dependency runs."""
    return request.state.owner


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


async def _parse_submission(request: Request, limits: Limits) -> tuple[JobRequest, list[bytes]]:
    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    if content_type == b"multipart/form-data":
        return await _parse_multipart_job(request, params, limits)
    return await _parse_json_job(request, content_type, limits), []


async def _parse_json_job(request: Request, content_type: bytes, limits: Limits) -> JobRequest:
    body_bytes = await _read_bounded_body(request, limits.max_upload_bytes)
    is_json = content_type == b"application/json" or (
        content_type.startswith(b"application/") and content_type.endswith(b"+json")
    )
    if body_bytes and is_json:
        try:
            raw: Any = json.loads(body_bytes.decode("utf-8"))
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
    elif body_bytes:
        raw = body_bytes
    else:
        raw = None
    try:
        body = _JOB_REQUEST_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        errors = [{**err, "loc": ("body", *err["loc"])} for err in exc.errors(include_url=False)]
        raise RequestValidationError(errors) from exc
    errors = _limit_errors(body, limits)
    if errors:
        raise RequestValidationError(errors)
    return body


async def _read_bounded_body(request: Request, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_bytes:
                raise _validation_error(
                    ("body",), f"upload exceeds maximum size of {max_bytes} bytes"
                )
            chunks.append(chunk)
    except ClientDisconnect as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "There was an error parsing the body"
        ) from exc
    return b"".join(chunks)


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
        self._is_file = b"filename" in options
        if self._is_file:
            if self._part_name != "references":
                raise _validation_error(
                    ("body", self._part_name),
                    f"unexpected file field {self._part_name!r}; reference files must use "
                    "field 'references'",
                )
            if len(self.files) + 1 > self.limits.max_references:
                raise _validation_error(
                    ("body", "references"),
                    f"at most {self.limits.max_references} reference images are allowed",
                )
        else:
            if self._part_name == "references":
                raise _validation_error(
                    ("body", "references"),
                    "references must be uploaded as image files, not text or URLs",
                )
            if len(self.fields) + 1 > MAX_FORM_FIELDS:
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


async def _parse_multipart_job(
    request: Request, params: dict[bytes, bytes], limits: Limits
) -> tuple[JobRequest, list[bytes]]:
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
                    ("body", "references"),
                    f"upload exceeds maximum size of {limits.max_upload_bytes} bytes",
                )
            parser.write(chunk)
        parser.finalize()
    except (FormParserError, ClientDisconnect) as exc:
        raise _validation_error(("body",), "invalid multipart/form-data body") from exc

    if not acc.ended:
        raise _validation_error(("body",), "incomplete multipart/form-data body")

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
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > limits.max_pixels:
                return None, {
                    "type": "value_error",
                    "loc": loc,
                    "msg": f"reference image {index} exceeds maximum of {limits.max_pixels} pixels",
                    "input": None,
                }
            image.load()
            ImageOps.exif_transpose(image, in_place=True)
            rgb = image if image.mode == "RGB" else image.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="PNG")
            return buf.getvalue(), None
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
        SyntaxError,
    ):
        return None, {
            "type": "value_error",
            "loc": loc,
            "msg": f"reference image {index} is not a valid image",
            "input": None,
        }


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


def _decode_lossless(data: bytes) -> str:
    return data.decode("utf-8", "backslashreplace")


def _depth_limited(value: Any, depth: int) -> Any:
    """Replace containers nested deeper than `depth` with a marker; recursion stops at `depth`."""
    if not isinstance(value, dict | list | tuple):
        return value
    if depth == 0:
        return TOO_DEEP
    if isinstance(value, dict):
        return {key: _depth_limited(item, depth - 1) for key, item in value.items()}
    return [_depth_limited(item, depth - 1) for item in value]


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
