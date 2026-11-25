"""Image storage: a private local directory or a private S3-compatible bucket.

Both use one layout below the root directory or key prefix: `<job>/references/<i>.png` holds
a job's uploaded references and `<job>/<i>.png` its outputs. Neither is ever reachable before
the job's success is stored: local downloads check publication, and S3 links are only
presigned for succeeded jobs. References are never downloadable.
"""

import logging
import shutil
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from chriek.config import S3Settings
from chriek.storage import PRIVATE_DIR_MODE, StorageError, private_image_root, write_private

log = logging.getLogger(__name__)


def image_path(root: Path, job_id: str, index: int) -> Path:
    return root / job_id / f"{index}.png"


def reference_path(root: Path, job_id: str, index: int) -> Path:
    return root / job_id / "references" / f"{index}.png"


class LocalImages:
    """Private files below a checked directory (see storage.py); outputs are staged in
    `<job>.partial/` and moved into `<job>/` only once every image is written."""

    def __init__(self, root: Path) -> None:
        self.root = private_image_root(root)

    def put_references(self, job_id: str, references: Sequence[bytes]) -> None:
        if not references:
            return
        references_dir = reference_path(self.root, job_id, 0).parent
        references_dir.parent.mkdir(mode=PRIVATE_DIR_MODE)
        references_dir.mkdir(mode=PRIVATE_DIR_MODE)
        for index, png_bytes in enumerate(references):
            # The job row doesn't exist yet and a failed submission removes `<id>/`: no rename.
            write_private(reference_path(self.root, job_id, index), png_bytes)

    def remove_submission(self, job_id: str, _reference_count: int) -> None:
        shutil.rmtree(self.root / job_id, onexc=_log_cleanup_error)

    def read_reference(self, job_id: str, index: int) -> bytes:
        return reference_path(self.root, job_id, index).read_bytes()

    def begin_outputs(self, job_id: str) -> None:
        partial = self._partial(job_id)
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(mode=PRIVATE_DIR_MODE)

    def stage_output(self, job_id: str, index: int, png: bytes) -> None:
        write_private(self._partial(job_id) / f"{index}.png", png)

    def publish_outputs(self, job_id: str, count: int) -> None:
        # Reference jobs already own `<id>/`, so outputs move in one by one.
        partial = self._partial(job_id)
        (self.root / job_id).mkdir(mode=PRIVATE_DIR_MODE, exist_ok=True)
        for index in range(count):
            (partial / f"{index}.png").replace(image_path(self.root, job_id, index))
        shutil.rmtree(partial, ignore_errors=True)

    def remove_outputs(self, job_id: str, count: int) -> None:
        """Remove staged and moved outputs; references stay with their job."""
        shutil.rmtree(self._partial(job_id), ignore_errors=True)
        for index in range(count):
            path = image_path(self.root, job_id, index)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.exception("could not remove unpublished output %s for job %s", path, job_id)
        with suppress(OSError):  # only when empty: a job without references
            (self.root / job_id).rmdir()

    def _partial(self, job_id: str) -> Path:
        return self.root / f"{job_id}.partial"


class S3Images:
    """Private objects in an existing bucket. Outputs are written to their final keys: without
    a presigned link, which only a succeeded job gets, a private object can't be read."""

    def __init__(self, settings: S3Settings, url_ttl_seconds: int) -> None:
        self._bucket = settings.bucket
        self._prefix = settings.prefix
        self._ttl = url_ttl_seconds
        compatible = settings.endpoint_url is not None
        # Many S3-compatible services reject the aws-chunked checksum trailers botocore sends
        # over https by default; AWS (None: botocore's defaults) supports them.
        checksums = "when_required" if compatible else None
        try:
            self._client = boto3.client(
                "s3",
                endpoint_url=settings.endpoint_url,
                region_name=settings.region,
                aws_access_key_id=settings.access_key_id,
                aws_secret_access_key=settings.secret_access_key,
                config=Config(
                    signature_version="s3v4",
                    # S3-compatible services rarely route bucket subdomains; AWS prefers them.
                    s3={"addressing_style": "path" if compatible else "auto"},
                    connect_timeout=10,
                    retries={"mode": "standard"},
                    # Only CHRIEK_S3_ENDPOINT_URL picks the host, never AWS_ENDPOINT_URL* or
                    # a profile's endpoint_url.
                    ignore_configured_endpoint_urls=True,
                    request_checksum_calculation=checksums,
                    response_checksum_validation=checksums,
                ),
            )
            response = self._client.head_bucket(Bucket=self._bucket)
        except (BotoCoreError, ClientError) as error:
            raise StorageError(
                f"CHRIEK_S3_BUCKET {self._bucket} cannot be used ({_describe(error)}); it must "
                "already exist and be reachable with the configured endpoint and credentials"
            ) from None
        # botocore follows AWS's region redirect for requests but presigns with the configured
        # region, so every link of a bucket in another region would be rejected.
        bucket_region = response.get("BucketRegion")
        if not compatible and bucket_region and bucket_region != settings.region:
            raise StorageError(
                f"CHRIEK_S3_BUCKET {self._bucket} is in region {bucket_region}; "
                f"set CHRIEK_S3_REGION={bucket_region}"
            )

    def put_references(self, job_id: str, references: Sequence[bytes]) -> None:
        for index, png_bytes in enumerate(references):
            self._put(self._key(job_id, "references", f"{index}.png"), png_bytes)

    def remove_submission(self, job_id: str, reference_count: int) -> None:
        for index in range(reference_count):
            self._delete(self._key(job_id, "references", f"{index}.png"), job_id)

    def read_reference(self, job_id: str, index: int) -> bytes:
        key = self._key(job_id, "references", f"{index}.png")
        return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    def begin_outputs(self, job_id: str) -> None:
        pass

    def stage_output(self, job_id: str, index: int, png: bytes) -> None:
        self._put(self._key(job_id, f"{index}.png"), png)

    def publish_outputs(self, job_id: str, count: int) -> None:
        pass

    def remove_outputs(self, job_id: str, count: int) -> None:
        for index in range(count):
            self._delete(self._key(job_id, f"{index}.png"), job_id)

    def presigned_url(self, job_id: str, index: int) -> tuple[str, int]:
        """A direct GET link to the object and its expiry, from the signature's own timestamp."""
        url = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": self._key(job_id, f"{index}.png")},
            ExpiresIn=self._ttl,
        )
        (signed_at,) = parse_qs(urlsplit(url).query)["X-Amz-Date"]
        signed = datetime.strptime(signed_at, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        return url, int(signed.timestamp()) + self._ttl

    def _key(self, *parts: str) -> str:
        return "/".join((self._prefix, *parts) if self._prefix else parts)

    def _put(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType="image/png")

    def _delete(self, key: str, job_id: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except (BotoCoreError, ClientError) as error:
            log.error("could not remove %s for job %s (%s)", key, job_id, _describe(error))


def open_images(storage: Path | S3Settings, url_ttl_seconds: int) -> LocalImages | S3Images:
    if isinstance(storage, Path):
        return LocalImages(storage)
    return S3Images(storage, url_ttl_seconds)


def _describe(error: BotoCoreError | ClientError) -> str:
    """The S3 error code, or botocore's message (endpoint URL, no credentials)."""
    if isinstance(error, ClientError):
        return f"S3 error {error.response.get('Error', {}).get('Code', 'unknown')}"
    return str(error)


def _log_cleanup_error(_function: Callable[..., Any], path: str, exc: BaseException) -> None:
    if not isinstance(exc, FileNotFoundError):
        log.error("could not remove %s after a failed submission", path, exc_info=exc)
