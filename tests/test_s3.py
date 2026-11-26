"""S3 storage: configuration, and against a real S3-compatible service (CHRIEK_TEST_S3_URL) the
direct presigned download walkthrough, link tampering and expiry, and object cleanup.

The rest of the API suite also runs on S3 when CHRIEK_TEST_S3_URL is set (see conftest.py).
"""

import hashlib
import io
import logging
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import boto3
import pytest
from botocore.awsrequest import AWSResponse
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from conftest import (
    ALICE,
    BOB,
    S3_BUCKET,
    S3_URL,
    SIGNING_SECRET,
    SMALL,
    FailingBackend,
    FakeClock,
    GatedBackend,
    base_env,
    ensure_bucket,
    fetch,
    make_image_bytes,
    running,
    s3_env,
    s3_keys,
    stored_owners,
    submit,
    wait_for,
)
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.exc import OperationalError

from chriek.api import create_app
from chriek.config import ConfigError, S3Settings, Settings
from chriek.images import S3Images
from chriek.png import DIGITAL_SOURCE_TYPE, XMP_KEY
from chriek.signing import UrlSigner
from chriek.storage import StorageError
from chriek.store import JobStore

SECRET = "s3-secret-SENTINEL-0123456789"
S3_BASE = {
    "CHRIEK_STORAGE": "s3",
    "CHRIEK_S3_BUCKET": "chriek-images",
    "CHRIEK_S3_ACCESS_KEY_ID": "AKIDSENTINEL",
    "CHRIEK_S3_SECRET_ACCESS_KEY": SECRET,
}
needs_s3 = pytest.mark.skipif(S3_URL is None, reason="CHRIEK_TEST_S3_URL not set")


def local_env(tmp_path: Path) -> dict[str, str]:
    env = base_env(tmp_path)
    for name in [name for name in env if name.startswith("CHRIEK_S3_")]:
        del env[name]
    env.pop("CHRIEK_STORAGE", None)
    env["CHRIEK_STORAGE_DIR"] = str(tmp_path / "images")
    return env


def s3_settings_env(tmp_path: Path, **changes: str) -> dict[str, str]:
    env = local_env(tmp_path) | S3_BASE
    del env["CHRIEK_STORAGE_DIR"]
    return env | changes


# Configuration (no service needed)


def test_local_storage_is_the_default(tmp_path):
    env = local_env(tmp_path)
    del env["CHRIEK_STORAGE_DIR"]
    assert Settings.from_env(env).storage == Path("chriek-data/images")


def test_s3_settings_with_defaults(tmp_path):
    storage = Settings.from_env(s3_settings_env(tmp_path)).storage
    assert storage == S3Settings(
        bucket="chriek-images",
        endpoint_url=None,
        region="us-east-1",
        prefix="",
        access_key_id="AKIDSENTINEL",
        secret_access_key=SECRET,
    )
    assert SECRET not in repr(storage)


def test_s3_settings_with_endpoint_region_and_prefix(tmp_path):
    storage = Settings.from_env(
        s3_settings_env(
            tmp_path,
            CHRIEK_S3_ENDPOINT_URL="https://s3.example.test:9000/",
            CHRIEK_S3_REGION="eu-central-1",
            CHRIEK_S3_PREFIX="team/chriek",
        )
    ).storage
    assert isinstance(storage, S3Settings)
    assert (storage.endpoint_url, storage.region, storage.prefix) == (
        "https://s3.example.test:9000",
        "eu-central-1",
        "team/chriek",
    )


@pytest.mark.parametrize(
    ("changes", "removed", "message"),
    [
        ({"CHRIEK_STORAGE": "gcs"}, (), "CHRIEK_STORAGE must be 'local' or 's3'"),
        ({}, ("CHRIEK_S3_BUCKET",), "CHRIEK_S3_BUCKET is required"),
        ({}, ("CHRIEK_S3_ACCESS_KEY_ID",), "CHRIEK_S3_ACCESS_KEY_ID is required"),
        ({}, ("CHRIEK_S3_SECRET_ACCESS_KEY",), "CHRIEK_S3_SECRET_ACCESS_KEY is required"),
        ({"CHRIEK_S3_BUCKET": "Bad_Bucket"}, (), "CHRIEK_S3_BUCKET must be"),
        ({"CHRIEK_S3_BUCKET": "a..b"}, (), "CHRIEK_S3_BUCKET must be"),
        ({"CHRIEK_S3_ENDPOINT_URL": ""}, (), "CHRIEK_S3_ENDPOINT_URL must not be empty"),
        ({"CHRIEK_S3_ENDPOINT_URL": "ftp://h"}, (), "CHRIEK_S3_ENDPOINT_URL must be an absolute"),
        (
            {"CHRIEK_S3_ENDPOINT_URL": f"https://key:{SECRET}@s3.example.test"},
            (),
            "CHRIEK_S3_ENDPOINT_URL must not contain credentials",
        ),
        (
            {"CHRIEK_S3_ENDPOINT_URL": "https://s3.example.test/bucket"},
            (),
            "CHRIEK_S3_ENDPOINT_URL must not contain a path",
        ),
        ({"CHRIEK_S3_REGION": "EU West"}, (), "CHRIEK_S3_REGION must be"),
        ({"CHRIEK_S3_PREFIX": "/abs"}, (), "CHRIEK_S3_PREFIX must be"),
        ({"CHRIEK_S3_PREFIX": "a/../b"}, (), "CHRIEK_S3_PREFIX must be"),
        (
            {"CHRIEK_S3_SECRET_ACCESS_KEY": f"{SECRET} x"},
            (),
            "CHRIEK_S3_SECRET_ACCESS_KEY must be printable ASCII",
        ),
        ({"CHRIEK_STORAGE_DIR": "images"}, (), "CHRIEK_STORAGE_DIR applies only when"),
    ],
)
def test_invalid_s3_settings_are_refused_without_echoing_secrets(
    tmp_path, changes, removed, message
):
    env = s3_settings_env(tmp_path, **changes)
    for name in removed:
        del env[name]
    with pytest.raises(ConfigError, match=message) as error:
        Settings.from_env(env)
    assert SECRET not in str(error.value)


def test_s3_settings_are_refused_with_local_storage(tmp_path):
    env = local_env(tmp_path) | {"CHRIEK_S3_SECRET_ACCESS_KEY": SECRET}
    with pytest.raises(ConfigError, match="CHRIEK_S3_SECRET_ACCESS_KEY applies only when") as error:
        Settings.from_env(env)
    assert SECRET not in str(error.value)


def test_s3_endpoint_errors_describe_an_origin_and_public_url_errors_a_prefix(tmp_path):
    for raw in ("ftp://h", "https://h:0"):
        with pytest.raises(ConfigError, match="CHRIEK_S3_ENDPOINT_URL .*port") as error:
            Settings.from_env(s3_settings_env(tmp_path, CHRIEK_S3_ENDPOINT_URL=raw))
        assert "prefix" not in str(error.value)
    with pytest.raises(ConfigError, match="CHRIEK_PUBLIC_URL .*path prefix"):
        Settings.from_env(local_env(tmp_path) | {"CHRIEK_PUBLIC_URL": "ftp://h"})


def _stubbed_s3() -> tuple[object, Stubber]:
    client = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="k", aws_secret_access_key="s"
    )
    return client, Stubber(client)


@pytest.mark.parametrize(("head_code", "created"), [(None, False), ("404", True)])
def test_test_bucket_is_created_only_when_missing(head_code, created):
    client, stubber = _stubbed_s3()
    params = {"Bucket": "chriek-test"}
    if head_code is None:
        stubber.add_response("head_bucket", {}, params)
    else:
        stubber.add_client_error("head_bucket", head_code, http_status_code=404)
    if created:
        stubber.add_response("create_bucket", {}, params)
    with stubber:
        ensure_bucket(client, "chriek-test")
    stubber.assert_no_pending_responses()


def test_test_bucket_access_errors_propagate_without_creating_it():
    client, stubber = _stubbed_s3()
    stubber.add_client_error("head_bucket", "403", http_status_code=403)
    # An unexpected create_bucket call would raise the stubber's own error instead.
    with stubber, pytest.raises(ClientError) as error:
        ensure_bucket(client, "chriek-test")
    assert error.value.response["Error"]["Code"] == "403"


def test_unreachable_storage_stops_startup_without_leaking_credentials(tmp_path):
    env = s3_settings_env(tmp_path, CHRIEK_S3_ENDPOINT_URL="http://127.0.0.1:9")
    settings = Settings.from_env(env)
    with pytest.raises(StorageError, match="CHRIEK_S3_BUCKET chriek-images cannot be used") as e:
        create_app(settings)
    assert SECRET not in str(e.value)
    assert not (tmp_path / "images").exists()


# The S3 client against a stubbed wire (no service, no developer AWS configuration)


class _Raw:
    def stream(self, **_kwargs):
        yield b""


@pytest.fixture
def wire(monkeypatch, tmp_path):
    """Every request the S3 client sends, answered by `responses` (status, headers) in order,
    then 200. Instead of the developer's AWS files, the client gets a profile whose ambient
    endpoint_url chriek must ignore."""
    for name in [name for name in os.environ if name.startswith("AWS_")]:
        monkeypatch.delenv(name)
    (tmp_path / "aws-config").write_text("[default]\nendpoint_url = https://evil.example\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws-credentials"))
    sent, responses = [], []

    def answer(request, **_kwargs):
        sent.append(request)
        status, headers = responses.pop(0) if responses else (200, {})
        return AWSResponse(request.url, status, headers, _Raw())

    session = boto3.Session()
    session.events.register("before-send.s3", answer)
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", session)
    return sent, responses


def _settings(endpoint_url: str | None = None, region: str = "us-east-1") -> S3Settings:
    return S3Settings("chriek-images", endpoint_url, region, "", "AKIDSENTINEL", SECRET)


@pytest.mark.parametrize("variable", ["AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL", None])
def test_ambient_aws_endpoints_never_redirect_s3_traffic(wire, monkeypatch, variable):
    sent, _ = wire
    if variable:
        monkeypatch.setenv(variable, "https://evil.example")
    images = S3Images(_settings(), 60)
    assert urlsplit(sent[0].url).hostname == "chriek-images.s3.amazonaws.com"
    assert urlsplit(images.presigned_url("job", 0)[0]).hostname == "chriek-images.s3.amazonaws.com"

    sent.clear()
    images = S3Images(_settings("https://s3.example.test:9000"), 60)
    assert sent[0].url == "https://s3.example.test:9000/chriek-images"
    assert images.presigned_url("job", 0)[0].startswith(
        "https://s3.example.test:9000/chriek-images/job/0.png?"
    )


def test_aws_bucket_in_another_region_stops_startup(wire):
    _, responses = wire
    # AWS answers the wrong region with a redirect, which botocore follows.
    redirect = {"x-amz-bucket-region": "eu-central-1"}
    responses.extend([(301, redirect), (200, redirect)])
    with pytest.raises(StorageError, match="set CHRIEK_S3_REGION=eu-central-1") as error:
        S3Images(_settings(), 60)
    assert SECRET not in str(error.value)


@pytest.mark.parametrize(
    ("endpoint_url", "bucket_region"),
    [(None, "eu-central-1"), (None, None), ("https://s3.example.test", "us-west-2")],
)
def test_bucket_region_is_checked_only_when_aws_reports_it(wire, endpoint_url, bucket_region):
    _, responses = wire
    responses.append((200, {"x-amz-bucket-region": bucket_region} if bucket_region else {}))
    url, _ = S3Images(_settings(endpoint_url, "eu-central-1"), 60).presigned_url("job", 0)
    assert _query(url)["X-Amz-Credential"].endswith("/eu-central-1/s3/aws4_request")


def test_compatible_endpoint_uploads_over_https_without_checksum_trailers(wire):
    sent, _ = wire
    png = make_image_bytes()
    S3Images(_settings("https://s3.example.test"), 60).stage_output("job", 0, png)
    put = sent[-1]
    assert (put.method, put.url) == ("PUT", "https://s3.example.test/chriek-images/job/0.png")
    assert put.headers["Content-Type"] == b"image/png"
    assert "Content-Encoding" not in put.headers
    assert "X-Amz-Trailer" not in put.headers
    assert put.headers["X-Amz-Content-SHA256"] == hashlib.sha256(png).hexdigest().encode()


# Against the real service


def _query(url: str) -> dict[str, str]:
    return {key: values[0] for key, values in parse_qs(urlsplit(url).query).items()}


@needs_s3
def test_s3_walkthrough_json_and_multipart_jobs_download_directly(tmp_path, storage, s3_client):
    endpoint = s3_env(storage)["CHRIEK_S3_ENDPOINT_URL"]
    with running(tmp_path, CHRIEK_URL_TTL_SECONDS="120") as client:
        plain = wait_for(client, submit(client, prompt="lighthouse", seed=3, **SMALL)["id"])
        reference = make_image_bytes((200, 10, 10))
        response = client.post(
            "/jobs",
            headers=ALICE,
            data={"prompt": "lighthouse", "seed": "3", "width": "16", "height": "16"},
            files=[("references", ("ref.png", reference, "image/png"))],
        )
        assert response.status_code == 202, response.text
        edited = wait_for(client, response.json()["id"])
        # Owner isolation is unchanged: bob sees neither job.
        assert client.get(f"/jobs/{edited['id']}", headers=BOB).status_code == 404

    pngs = []
    for job in (plain, edited):
        assert job["status"] == "succeeded", job
        (image,) = job["images"]
        url = image["url"]
        assert url.startswith(f"{endpoint}/{S3_BUCKET}/{storage}/{job['id']}/0.png?")
        query = _query(url)
        assert query["X-Amz-Algorithm"] == "AWS4-HMAC-SHA256"
        assert query["X-Amz-Expires"] == "120"
        signed = datetime.strptime(query["X-Amz-Date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        assert image["expires_at"] == (signed + timedelta(seconds=120)).isoformat()
        status, body = fetch(url)
        assert status == 200
        with Image.open(io.BytesIO(body)) as png:
            assert png.format == "PNG"
            assert png.info["AI-Generated"] == "true"
            assert png.info["Seed"] == "3"
            assert DIGITAL_SOURCE_TYPE in png.info[XMP_KEY]
        pngs.append(body)
    assert pngs[0] != pngs[1], "the reference must change the fake backend's output"
    assert s3_keys(s3_client, storage) == sorted(
        [f"{plain['id']}/0.png", f"{edited['id']}/0.png", f"{edited['id']}/references/0.png"]
    )


@needs_s3
def test_s3_links_reject_tampering_unsigned_access_and_references(tmp_path, storage):
    with running(tmp_path) as client:
        first = submit(client, image_count=2, **SMALL)
        response = client.post(
            "/jobs",
            headers=ALICE,
            data={"prompt": "edit", "width": "16", "height": "16"},
            files=[("references", ("ref.png", make_image_bytes(), "image/png"))],
        )
        edit = response.json()
        first = wait_for(client, first["id"])
        assert wait_for(client, edit["id"])["status"] == "succeeded"
    url = first["images"][0]["url"]
    assert fetch(url)[0] == 200
    signature = _query(url)["X-Amz-Signature"]
    forged = "0" * len(signature) if signature != "0" * len(signature) else "1" * len(signature)
    tampered = {
        "other image": url.replace(f"{first['id']}/0.png", f"{first['id']}/1.png"),
        "reference": url.replace(f"{first['id']}/0.png", f"{edit['id']}/references/0.png"),
        "signature": url.replace(signature, forged),
        "longer expiry": url.replace("X-Amz-Expires=900", "X-Amz-Expires=604800"),
        "unsigned": url.split("?")[0],
    }
    for name, bad in tampered.items():
        assert bad != url, name
        assert fetch(bad)[0] == 403, name


@needs_s3
def test_s3_links_work_until_the_configured_ttl_then_expire(tmp_path):
    ttl = 3  # the signature date is truncated to whole seconds: at least 2 s to fetch in time
    with running(tmp_path, CHRIEK_URL_TTL_SECONDS=str(ttl)) as client:
        job = wait_for(client, submit(client, **SMALL)["id"])
    (image,) = job["images"]
    url = image["url"]
    assert _query(url)["X-Amz-Expires"] == str(ttl)
    expires = datetime.fromisoformat(image["expires_at"]).timestamp()
    status, _ = fetch(url)
    assert time.time() < expires, "fetch finished after expiry; rerun on a less loaded host"
    assert status == 200
    # The same link, once its expiry has passed (1 s slack for the service's second rounding).
    time.sleep(expires - time.time() + 1)
    assert fetch(url)[0] == 403


@needs_s3
def test_s3_mode_serves_no_local_download_route(tmp_path):
    with running(tmp_path) as client:
        job = wait_for(client, submit(client, **SMALL)["id"])
        expires = 4_000_000_000
        sig = UrlSigner(SIGNING_SECRET).sign(uuid.UUID(job["id"]), 0, expires)
        response = client.get(f"/images/{job['id']}/0.png?expires={expires}&sig={sig}")
        assert response.status_code == 404
        assert "/images/{job_id}/{index}.png" not in client.get("/openapi.json").json()["paths"]


@needs_s3
def test_s3_failed_and_partial_outputs_are_removed_and_never_linked(tmp_path, storage, s3_client):
    with running(tmp_path, backend=FailingBackend(fail_index=1)) as client:
        job = wait_for(client, submit(client, image_count=3, **SMALL)["id"])
    assert (job["status"], job["error"], job["images"]) == ("failed", "image 2 of 3 failed", [])
    assert s3_keys(s3_client, storage) == []


@needs_s3
def test_s3_output_write_failure_fails_the_job_and_keeps_no_outputs(
    tmp_path, storage, s3_client, monkeypatch
):
    puts: list[str] = []
    persisted: list[int] = []
    original_init = S3Images.__init__

    def refuse_second(params, **_kwargs):
        puts.append(params["Key"])
        if len(puts) == 2:
            # The first output really reached the bucket before the write failure.
            persisted.append(s3_client.head_object(Bucket=S3_BUCKET, Key=puts[0])["ContentLength"])
            raise ClientError({"Error": {"Code": "SlowDown", "Message": SECRET}}, "PutObject")

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._client.meta.events.register("before-parameter-build.s3.PutObject", refuse_second)

    monkeypatch.setattr(S3Images, "__init__", init)
    with running(tmp_path) as client:
        job = wait_for(client, submit(client, image_count=3, **SMALL)["id"])
    assert (job["status"], job["error"], job["images"]) == ("failed", "image 2 of 3 failed", [])
    assert puts == [f"{storage}/{job['id']}/{i}.png" for i in range(2)]
    assert len(persisted) == 1 and persisted[0] > 0
    assert s3_keys(s3_client, storage) == []


@needs_s3
def test_s3_missing_reference_fails_job_and_keeps_no_outputs(tmp_path, storage, s3_client):
    gated = GatedBackend()
    with running(tmp_path, backend=gated) as client:
        blocker = submit(client, prompt="hold worker", **SMALL)
        assert gated.rendering.wait(5)
        response = client.post(
            "/jobs",
            headers=ALICE,
            data={"prompt": "edit", "width": "16", "height": "16"},
            files=[("references", ("ref.png", make_image_bytes(), "image/png"))],
        )
        edit = response.json()
        s3_client.delete_object(Bucket=S3_BUCKET, Key=f"{storage}/{edit['id']}/references/0.png")
        gated.release()
        failed = wait_for(client, edit["id"])
        assert wait_for(client, blocker["id"])["status"] == "succeeded"
    assert (failed["status"], failed["error"]) == ("failed", "reference 1 of 1 could not be loaded")
    assert failed["images"] == []
    assert s3_keys(s3_client, storage) == [f"{blocker['id']}/0.png"]


@needs_s3
@pytest.mark.parametrize(
    ("changes", "files", "status"),
    [
        pytest.param(
            {"CHRIEK_MAX_QUEUED_JOBS": "1"},
            [("references", ("r.png", make_image_bytes(), "image/png"))],
            429,
            id="queue full",
        ),
        pytest.param(
            {}, [("references", ("r.png", b"not an image", "image/png"))], 422, id="invalid"
        ),
    ],
)
def test_s3_rejected_submissions_leave_no_objects_or_jobs(
    tmp_path, storage, s3_client, changes, files, status
):
    gated = GatedBackend()
    with running(tmp_path, backend=gated, **changes) as client:
        # One running and one queued job fill a one-job queue.
        running_job = submit(client, prompt="running", **SMALL)
        assert gated.rendering.wait(5)
        queued = submit(client, prompt="queued", **SMALL)
        response = client.post(
            "/jobs",
            headers=ALICE,
            data={"prompt": "rejected", "width": "16", "height": "16"},
            files=files,
        )
        assert response.status_code == status, response.text
        assert s3_keys(s3_client, storage) == []
        gated.release()
        for job in (running_job, queued):
            assert wait_for(client, job["id"])["status"] == "succeeded"
    assert len(stored_owners(tmp_path)) == 2


@pytest.fixture
def failing_deletes(monkeypatch, caplog):
    """Every DeleteObject of an app's S3 client fails; returns the keys it tried. The test
    fixtures' own client still deletes."""
    tried: list[str] = []
    original_init = S3Images.__init__

    def refuse(params, **_kwargs):
        tried.append(params["Key"])
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": SECRET}}, "DeleteObject")

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._client.meta.events.register("before-parameter-build.s3.DeleteObject", refuse)

    monkeypatch.setattr(S3Images, "__init__", init)
    caplog.set_level(logging.ERROR, "chriek.images")
    return tried


def _cleanup_failures_logged(caplog) -> bool:
    messages = [r.getMessage() for r in caplog.records if r.name == "chriek.images"]
    return any("could not remove" in m for m in messages) and not any(SECRET in m for m in messages)


@needs_s3
def test_s3_output_delete_failure_still_fails_the_job_and_keeps_the_worker(
    tmp_path, storage, s3_client, failing_deletes, caplog
):
    with running(tmp_path, backend=FailingBackend(fail_index=1)) as client:
        failed = wait_for(client, submit(client, image_count=3, **SMALL)["id"])
        assert (failed["status"], failed["error"], failed["images"]) == (
            ("failed", "image 2 of 3 failed", [])
        )
        assert failing_deletes == [f"{storage}/{failed['id']}/{i}.png" for i in range(3)]
        assert _cleanup_failures_logged(caplog)
        later = wait_for(client, submit(client, **SMALL)["id"])
        assert (later["status"], len(later["images"])) == ("succeeded", 1)
        assert client.get("/health").status_code == 200
    # The undeleted output stays as an unlinked orphan.
    assert f"{failed['id']}/0.png" in s3_keys(s3_client, storage)


@needs_s3
def test_s3_reference_delete_failure_keeps_the_queue_full_rejection(
    tmp_path, storage, s3_client, failing_deletes, caplog
):
    gated = GatedBackend()
    with running(tmp_path, backend=gated, CHRIEK_MAX_QUEUED_JOBS="1") as client:
        running_job = submit(client, prompt="running", **SMALL)
        assert gated.rendering.wait(5)
        queued = submit(client, prompt="queued", **SMALL)
        response = client.post(
            "/jobs",
            headers=ALICE,
            data={"prompt": "rejected", "width": "16", "height": "16"},
            files=[("references", ("r.png", make_image_bytes(), "image/png"))],
        )
        assert response.status_code == 429, response.text
        (key,) = failing_deletes
        assert key.startswith(f"{storage}/") and key.endswith("/references/0.png")
        assert _cleanup_failures_logged(caplog)
        gated.release()
        for job in (running_job, queued):
            assert wait_for(client, job["id"])["status"] == "succeeded"
    assert len(stored_owners(tmp_path)) == 2


@needs_s3
def test_s3_store_failure_after_references_removes_them(tmp_path, storage, s3_client, monkeypatch):
    def failing_create(self, *args, **kwargs):
        raise OperationalError("INSERT", {}, RuntimeError("database is full"))

    monkeypatch.setattr(JobStore, "create", failing_create)
    app = create_app(Settings.from_env(base_env(tmp_path)))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/jobs",
            headers=ALICE,
            data={"prompt": "lost", "width": "16", "height": "16"},
            files=[
                ("references", ("a.png", make_image_bytes((1, 2, 3)), "image/png")),
                ("references", ("b.png", make_image_bytes((3, 2, 1)), "image/png")),
            ],
        )
    assert response.status_code == 500
    assert s3_keys(s3_client, storage) == []


@needs_s3
def test_s3_write_failure_fails_submission_without_leaking(
    tmp_path, storage, s3_client, monkeypatch
):
    original_put = S3Images._put

    def fail_second_reference(self, key, data):
        if key.endswith("references/1.png"):
            raise ClientError({"Error": {"Code": "SlowDown", "Message": SECRET}}, "PutObject")
        original_put(self, key, data)

    monkeypatch.setattr(S3Images, "_put", fail_second_reference)
    app = create_app(Settings.from_env(base_env(tmp_path)))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/jobs",
            headers=ALICE,
            data={"prompt": "lost", "width": "16", "height": "16"},
            files=[
                ("references", ("a.png", make_image_bytes((1, 2, 3)), "image/png")),
                ("references", ("b.png", make_image_bytes((3, 2, 1)), "image/png")),
            ],
        )
    assert response.status_code == 500
    assert SECRET not in response.text
    assert s3_keys(s3_client, storage) == []
    assert stored_owners(tmp_path) == []


@needs_s3
@pytest.mark.parametrize(
    ("changes", "code"),
    [
        pytest.param({"CHRIEK_S3_BUCKET": "chriek-missing-bucket"}, "404", id="missing bucket"),
        pytest.param({"CHRIEK_S3_SECRET_ACCESS_KEY": SECRET}, "403", id="wrong secret"),
    ],
)
def test_s3_unusable_bucket_stops_startup_without_leaking(tmp_path, storage, changes, code):
    settings = Settings.from_env(base_env(tmp_path) | changes)
    with pytest.raises(StorageError, match=f"cannot be used \\(S3 error {code}\\)") as error:
        create_app(settings)
    assert SECRET not in str(error.value)
    assert s3_env(storage)["CHRIEK_S3_SECRET_ACCESS_KEY"] not in str(error.value)


@needs_s3
def test_s3_late_result_of_a_timed_out_job_is_removed(tmp_path, storage, s3_client):
    clock = FakeClock()
    gated = GatedBackend()
    with running(tmp_path, backend=gated, clock=clock, CHRIEK_JOB_TIMEOUT_SECONDS="60") as client:
        response = client.post(
            "/jobs",
            headers=ALICE,
            data={"prompt": "stuck", "image_count": "2", "width": "16", "height": "16"},
            files=[("references", ("ref.png", make_image_bytes(), "image/png"))],
        )
        stuck = response.json()
        assert gated.rendering.wait(5)
        successor = submit(client, prompt="successor", **SMALL)
        clock.advance(61)
        timed_out = client.get(f"/jobs/{stuck['id']}", headers=ALICE).json()
        assert (timed_out["status"], timed_out["images"]) == ("failed", [])
        gated.release()  # the stuck job finishes late, then the successor runs
        assert wait_for(client, successor["id"])["status"] == "succeeded"
        assert client.get(f"/jobs/{stuck['id']}", headers=ALICE).json() == timed_out
    assert s3_keys(s3_client, storage) == sorted(
        [f"{stuck['id']}/references/0.png", f"{successor['id']}/0.png"]
    )
