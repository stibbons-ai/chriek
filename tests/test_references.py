"""Issue #5: multipart reference images for edit jobs."""

import hashlib
import io
import json
import logging
import os
import sqlite3
import struct
import subprocess
import sys
import threading
import tracemalloc
import uuid
import zlib
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from conftest import (
    ALICE,
    BOB,
    SIGNING_SECRET,
    CountingBackend,
    FailingBackend,
    GatedBackend,
    base_env,
    make_image_bytes,
    running,
    submit,
    wait_for,
)
from fastapi.testclient import TestClient
from PIL import Image, ImageOps
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

import chriek.api
from chriek.api import create_app
from chriek.backend import FakeBackend
from chriek.config import Settings
from chriek.signing import UrlSigner
from chriek.store import JobStore
from chriek.worker import reference_path


def encode_multipart(
    data: dict[str, str],
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    boundary: str = "chriek-multipart-boundary",
) -> tuple[dict[str, str], bytes]:
    parts: list[bytes] = []
    for name, value in data.items():
        header = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
        parts.append(f"{header}{value}\r\n".encode())
    for field_name, (filename, content, content_type) in files or []:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field_name}"; '
            f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode()
            + content
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return {"Content-Type": f"multipart/form-data; boundary={boundary}"}, b"".join(parts)


def post_multipart(client, headers=ALICE, data=None, files=None):
    ct_header, body = encode_multipart(
        {"prompt": "a lighthouse"} if data is None else data, files=files
    )
    return client.post("/jobs", headers=headers | ct_header, content=body)


def submit_multipart(client, headers=ALICE, data=None, files=None):
    response = post_multipart(client, headers=headers, data=data, files=files)
    assert response.status_code == 202, response.text
    return response.json()


def download_pixels(client, job_body: dict, index: int = 0) -> bytes:
    response = client.get(job_body["images"][index]["url"])
    assert response.status_code == 200
    return Image.open(io.BytesIO(response.content)).tobytes()


def asgi_post_jobs(
    client: TestClient, headers: dict[str, str], chunks: list[bytes]
) -> tuple[int, dict, int]:
    """Drive the live ASGI app with explicit multi-message `http.request` chunks and raw headers."""
    consumed = 0
    iterator = iter(chunks)

    async def receive():
        nonlocal consumed
        try:
            chunk = next(iterator)
            consumed += 1
            return {"type": "http.request", "body": chunk, "more_body": True}
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/jobs",
        "raw_path": b"/jobs",
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    async def call() -> None:
        await client.app(scope, receive, send)

    assert client.portal is not None
    client.portal.call(call)
    status_code = next(m["status"] for m in sent if m["type"] == "http.response.start")
    raw_body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status_code, json.loads(raw_body), consumed


class RecordingBackend(FakeBackend):
    """Fake backend recording the (size, top-left pixel) of every reference per render."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[tuple[int, int], tuple[int, ...]]]] = []

    def render(self, prompt, width, height, seed, references=()):
        self.calls.append([(ref.size, ref.getpixel((0, 0))) for ref in references])
        return super().render(prompt, width, height, seed, references)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _palette_png_without_plte() -> bytes:
    """Invalid PNG: colour type 3 (palette) with a one-byte tRNS but no PLTE chunk."""
    ihdr = struct.pack(">IIBBBBB", 16, 16, 8, 3, 0, 0, 0)
    idat = zlib.compress(b"\x00" * 17 * 16)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"tRNS", b"\x00")
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _png_bytes(image: Image.Image, **save_args) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", **save_args)
    return buf.getvalue()


def test_multipart_without_references_matches_json_text_to_image(client):
    json_job = wait_for(
        client,
        submit(client, prompt="harbor", image_count=1, seed=77, width=32, height=16)["id"],
    )
    mp_job = wait_for(
        client,
        submit_multipart(
            client,
            data={
                "prompt": "harbor",
                "image_count": "1",
                "seed": "77",
                "width": "32",
                "height": "16",
            },
        )["id"],
    )

    assert json_job["status"] == mp_job["status"] == "succeeded"
    assert (
        json_job["params"]
        == mp_job["params"]
        == {
            "prompt": "harbor",
            "image_count": 1,
            "seed": 77,
            "width": 32,
            "height": 16,
            "reference_count": 0,
        }
    )
    assert download_pixels(client, json_job) == download_pixels(client, mp_job)


def test_multipart_defaults_and_blank_seed_choose_random_seed(client):
    submitted = submit_multipart(client, data={"prompt": "harbor", "seed": ""})
    assert submitted["params"]["image_count"] == 1
    assert submitted["params"]["width"] == 1024
    assert submitted["params"]["height"] == 1024
    assert submitted["params"]["reference_count"] == 0
    assert isinstance(submitted["params"]["seed"], int)


def test_references_change_fake_backend_output_deterministically_and_preserve_order(client):
    red = make_image_bytes((220, 20, 20), (24, 16), fmt="PNG")
    blue = make_image_bytes((20, 40, 220), (16, 24), fmt="JPEG")
    green = make_image_bytes((20, 200, 40), (16, 16), fmt="WEBP")
    # Same size, different colour: only the reference pixels tell these apart.
    red_square = make_image_bytes((220, 20, 20), (16, 16))
    blue_square = make_image_bytes((20, 40, 220), (16, 16))
    form = {"prompt": "character panel", "seed": "12", "width": "32", "height": "32"}

    text_only = wait_for(client, submit_multipart(client, data=form)["id"])
    one_ref = wait_for(
        client,
        submit_multipart(client, data=form, files=[("references", ("red.png", red, "image/png"))])[
            "id"
        ],
    )
    one_ref_again = wait_for(
        client,
        submit_multipart(
            client, data=form, files=[("references", ("other-name.png", red, "image/png"))]
        )["id"],
    )
    other_pixel_ref = wait_for(
        client,
        submit_multipart(
            client, data=form, files=[("references", ("green.webp", green, "image/webp"))]
        )["id"],
    )
    red_then_blue = wait_for(
        client,
        submit_multipart(
            client,
            data=form,
            files=[
                ("references", ("a.png", red, "image/png")),
                ("references", ("b.jpg", blue, "image/jpeg")),
            ],
        )["id"],
    )
    blue_then_red = wait_for(
        client,
        submit_multipart(
            client,
            data=form,
            files=[
                ("references", ("b.jpg", blue, "image/jpeg")),
                ("references", ("a.png", red, "image/png")),
            ],
        )["id"],
    )
    red_square_job, blue_square_job = (
        wait_for(
            client,
            submit_multipart(
                client, data=form, files=[("references", ("square.png", square, "image/png"))]
            )["id"],
        )
        for square in (red_square, blue_square)
    )

    pixels_text = download_pixels(client, text_only)
    pixels_one = download_pixels(client, one_ref)
    pixels_one_again = download_pixels(client, one_ref_again)
    pixels_other = download_pixels(client, other_pixel_ref)
    pixels_rb = download_pixels(client, red_then_blue)
    pixels_br = download_pixels(client, blue_then_red)
    pixels_red_square = download_pixels(client, red_square_job)
    pixels_blue_square = download_pixels(client, blue_square_job)

    assert pixels_one == pixels_one_again, (
        "same references must render identically regardless of filename"
    )
    assert len({pixels_text, pixels_one, pixels_other, pixels_rb, pixels_br}) == 5
    assert pixels_red_square != pixels_blue_square


def test_every_render_receives_references_in_upload_order(tmp_path):
    uploads = [((220, 20, 20), (16, 16)), ((20, 40, 220), (24, 16)), ((20, 200, 40), (16, 24))]
    backend = RecordingBackend()
    with running(tmp_path, backend=backend) as client:
        job = submit_multipart(
            client,
            data={"prompt": "ordered", "image_count": "3", "width": "16", "height": "16"},
            files=[
                ("references", (f"{i}.png", make_image_bytes(color, size), "image/png"))
                for i, (color, size) in enumerate(uploads)
            ],
        )
        assert wait_for(client, job["id"])["status"] == "succeeded"

    expected = [(size, color) for color, size in uploads]
    assert backend.calls == [expected, expected, expected]


def test_reference_count_reported_across_all_job_statuses(tmp_path):
    gated = GatedBackend()
    ref1 = make_image_bytes((10, 20, 30))
    ref2 = make_image_bytes((40, 50, 60))
    files = [
        ("references", ("r1.png", ref1, "image/png")),
        ("references", ("r2.png", ref2, "image/png")),
    ]

    with running(tmp_path, backend=gated) as client:
        first = submit_multipart(
            client, data={"prompt": "first", "width": "16", "height": "16"}, files=files
        )
        assert first["status"] == "queued"
        assert first["params"]["reference_count"] == 2
        assert gated.rendering.wait(5)

        second = submit_multipart(
            client,
            data={"prompt": "second", "width": "16", "height": "16"},
            files=[("references", ("r1.png", ref1, "image/png"))],
        )
        assert second["status"] == "queued"
        assert second["params"]["reference_count"] == 1

        running_body = client.get(f"/jobs/{first['id']}", headers=ALICE).json()
        queued_body = client.get(f"/jobs/{second['id']}", headers=ALICE).json()
        assert running_body["status"] == "running"
        assert running_body["params"]["reference_count"] == 2
        assert queued_body["status"] == "queued"
        assert queued_body["params"]["reference_count"] == 1

        gated.release()
        done1 = wait_for(client, first["id"])
        done2 = wait_for(client, second["id"])
        assert done1["status"] == "succeeded"
        assert done1["params"]["reference_count"] == 2
        assert done2["status"] == "succeeded"
        assert done2["params"]["reference_count"] == 1

    failing = FailingBackend(fail_index=0)
    with running(tmp_path, backend=failing) as client:
        failed_submit = submit_multipart(
            client,
            data={"prompt": "fails", "width": "16", "height": "16"},
            files=files + [("references", ("r3.png", ref1, "image/png"))],
        )
        assert failed_submit["params"]["reference_count"] == 3
        failed_poll = wait_for(client, failed_submit["id"])
        assert failed_poll["status"] == "failed"
        assert failed_poll["params"]["reference_count"] == 3


def test_old_sqlite_jobs_without_references_report_zero_reference_count(tmp_path):
    db_path = tmp_path / "db" / "chriek.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    old_id = str(uuid.uuid7())
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE jobs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id VARCHAR(36) UNIQUE NOT NULL,
                owner VARCHAR(64) NOT NULL,
                status VARCHAR(16) NOT NULL,
                prompt TEXT NOT NULL,
                image_count INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                created_at FLOAT NOT NULL,
                started_at FLOAT,
                finished_at FLOAT,
                error TEXT,
                run_record JSON
            )"""
        )
        conn.execute(
            """INSERT INTO jobs (
                id, owner, status, prompt, image_count, width, height, seed, created_at
            ) VALUES (?, 'alice', 'succeeded', 'legacy prompt', 1, 16, 16, 7, 1700000000.0)""",
            (old_id,),
        )

    with running(tmp_path) as client:
        legacy = client.get(f"/jobs/{old_id}", headers=ALICE).json()
        assert legacy["status"] == "succeeded"
        assert legacy["params"]["reference_count"] == 0

        edit = wait_for(
            client,
            submit_multipart(
                client,
                data={"prompt": "new edit", "width": "16", "height": "16"},
                files=[("references", ("ref.png", make_image_bytes(), "image/png"))],
            )["id"],
        )
        assert edit["status"] == "succeeded"
        assert edit["params"]["reference_count"] == 1


def test_references_stored_with_safe_names_and_persist_across_restart_for_queued_job(tmp_path):
    crashed = GatedBackend()
    ref_a = make_image_bytes((111, 22, 33))
    ref_b = make_image_bytes((44, 155, 66))
    malicious_files = [
        ("references", ("../../0.png", ref_a, "image/png")),
        ("references", ("../db/chriek.db", ref_b, "image/png")),
    ]

    with running(tmp_path, backend=crashed) as before:
        first = submit_multipart(
            before,
            data={"prompt": "blocker-edit", "image_count": "2", "width": "16", "height": "16"},
            files=[("references", ("blocker.png", ref_a, "image/png"))],
        )
        assert crashed.rendering.wait(5)
        queued = submit_multipart(
            before,
            data={"prompt": "queued-edit", "seed": "55", "width": "16", "height": "16"},
            files=malicious_files,
        )
        job_dir = tmp_path / "images" / queued["id"]
        assert sorted(p.name for p in (job_dir / "references").iterdir()) == ["0.png", "1.png"]
        assert not (job_dir / "0.png").exists()
        assert not (tmp_path / "images" / "0.png").exists()
        assert not (tmp_path / "0.png").exists()

        with running(tmp_path) as after:
            interrupted = after.get(f"/jobs/{first['id']}", headers=ALICE).json()
            assert interrupted["status"] == "failed"
            assert interrupted["error"].startswith("interrupted:")
            assert interrupted["params"]["reference_count"] == 1
            survivor = wait_for(after, queued["id"])
            assert survivor["status"] == "succeeded"
            assert survivor["params"]["reference_count"] == 2

            direct = wait_for(
                after,
                submit_multipart(
                    after,
                    data={"prompt": "queued-edit", "seed": "55", "width": "16", "height": "16"},
                    files=[
                        ("references", ("clean0.png", ref_a, "image/png")),
                        ("references", ("clean1.png", ref_b, "image/png")),
                    ],
                )["id"],
            )
            assert download_pixels(after, survivor) == download_pixels(after, direct)
            crashed.release()
            assert wait_for(after, first["id"])["status"] == "failed"

    # Stale worker finished `first` after restart recovery failed it: outputs are cleaned up,
    # but uploaded reference files remain intact on disk.
    first_dir = tmp_path / "images" / first["id"]
    assert sorted(p.name for p in first_dir.iterdir()) == ["references"]
    assert sorted(p.name for p in (first_dir / "references").iterdir()) == ["0.png"]


def test_reference_bytes_cannot_be_downloaded_and_failed_edit_job_publishes_no_outputs(tmp_path):
    ref_bytes = make_image_bytes((200, 100, 50))
    # Fail on the second image (index 1) after image 0 has already been rendered.
    failing = FailingBackend(fail_index=1)
    signer = UrlSigner(SIGNING_SECRET)

    with running(tmp_path, backend=failing) as client:
        failed_job = submit_multipart(
            client,
            data={"prompt": "fail edit", "image_count": "2", "width": "16", "height": "16"},
            files=[("references", ("0.png", ref_bytes, "image/png"))],
        )
        failed_body = wait_for(client, failed_job["id"])
        assert failed_body["status"] == "failed"
        assert failed_body["images"] == []
        assert failed_body["error"] == "image 2 of 2 failed: injected backend failure"
        job_dir = tmp_path / "images" / failed_job["id"]
        assert not (tmp_path / "images" / f"{failed_job['id']}.partial").exists()
        assert sorted(p.name for p in job_dir.iterdir()) == ["references"]
        assert sorted(p.name for p in (job_dir / "references").iterdir()) == ["0.png"]

        ok_job = wait_for(
            client,
            submit_multipart(
                client,
                data={"prompt": "ok edit", "image_count": "1", "width": "16", "height": "16"},
                files=[
                    ("references", ("0.png", ref_bytes, "image/png")),
                    ("references", ("1.png", ref_bytes, "image/png")),
                ],
            )["id"],
        )
        assert ok_job["status"] == "succeeded"
        url0 = ok_job["images"][0]["url"]
        query = parse_qs(urlsplit(url0).query)
        expires = int(query["expires"][0])
        sig_ok_0 = query["sig"][0]
        sig_ok_1 = signer.sign(uuid.UUID(ok_job["id"]), 1, expires)
        sig_failed_0 = signer.sign(uuid.UUID(failed_job["id"]), 0, expires)

        for path in (
            f"/images/{ok_job['id']}/references/0.png?expires={expires}&sig={sig_ok_0}",
            f"/images/{ok_job['id']}/references%2F0.png?expires={expires}&sig={sig_ok_0}",
            f"/images/{ok_job['id']}/1.png?expires={expires}&sig={sig_ok_1}",
            f"/images/{failed_job['id']}/0.png?expires={expires}&sig={sig_failed_0}",
        ):
            assert client.get(path).status_code == 404, path


def test_url_and_text_reference_fields_are_rejected_without_network_fetch(tmp_path):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        for bad_json in (
            {"prompt": "hi", "references": ["http://127.0.0.1:9/ref.png"]},
            {"prompt": "hi", "reference_urls": ["http://127.0.0.1:9/ref.png"]},
            {"prompt": "hi", "image_url": "http://127.0.0.1:9/ref.png"},
        ):
            resp = client.post("/jobs", headers=ALICE, json=bad_json)
            assert resp.status_code == 422
            assert resp.json()["detail"]

        for bad_form in (
            {"prompt": "hi", "references": "http://127.0.0.1:9/ref.png"},
            {"prompt": "hi", "reference_url": "http://127.0.0.1:9/ref.png"},
        ):
            resp = post_multipart(client, headers=ALICE, data=bad_form)
            assert resp.status_code == 422
            assert resp.json()["detail"]
        assert backend.renders == 0


def test_reference_count_and_upload_byte_limits_are_configurable_and_enforced(tmp_path):
    small_png = make_image_bytes((10, 20, 30), (16, 16))
    boundary = "chriek-test-boundary"
    ref_files = [("references", ("r.png", small_png, "image/png"))]
    _, base_body = encode_multipart(
        {"prompt": "short", "width": "16", "height": "16"}, files=ref_files, boundary=boundary
    )
    limit_bytes = max(1024, len(base_body))
    padded_prompt = "short" + "x" * (limit_bytes - len(base_body))
    ct_header, exact_sized_body = encode_multipart(
        {"prompt": padded_prompt, "width": "16", "height": "16"},
        files=ref_files,
        boundary=boundary,
    )
    assert len(exact_sized_body) == limit_bytes
    _, over_body = encode_multipart(
        {"prompt": padded_prompt + "y", "width": "16", "height": "16"},
        files=ref_files,
        boundary=boundary,
    )

    json_prefix = b'{"prompt": "'
    json_suffix = b'", "width": 16, "height": 16}'
    json_exact = (
        json_prefix + b"j" * (limit_bytes - len(json_prefix) - len(json_suffix)) + json_suffix
    )
    assert len(json_exact) == limit_bytes
    json_over = (
        json_prefix + b"j" * (limit_bytes + 1 - len(json_prefix) - len(json_suffix)) + json_suffix
    )

    backend = CountingBackend()
    with running(
        tmp_path,
        backend=backend,
        CHRIEK_MAX_REFERENCES="2",
        CHRIEK_MAX_UPLOAD_BYTES=str(limit_bytes),
    ) as client:
        # Misleadingly small Content-Length (10 bytes): actual received bytes govern.
        ok = client.post(
            "/jobs",
            headers=ALICE | ct_header | {"Content-Length": "10"},
            content=exact_sized_body,
        )
        assert ok.status_code == 202, ok.text
        assert wait_for(client, ok.json()["id"])["status"] == "succeeded"

        over = client.post(
            "/jobs",
            headers=ALICE | ct_header | {"Content-Length": "10"},
            content=over_body,
        )
        assert over.status_code == 422
        assert over.json()["detail"][0]["loc"] == ["body"]
        assert "maximum size" in over.json()["detail"][0]["msg"]

        json_headers = ALICE | {"Content-Type": "application/json", "Content-Length": "10"}
        json_ok = client.post("/jobs", headers=json_headers, content=json_exact)
        assert json_ok.status_code == 202, json_ok.text
        json_over_cl = client.post("/jobs", headers=json_headers, content=json_over)
        assert json_over_cl.status_code == 422
        assert json_over_cl.json()["detail"][0]["loc"] == ["body"]
        assert "maximum size" in json_over_cl.json()["detail"][0]["msg"]

        # Missing Content-Length header over chunked ASGI stream (multipart and JSON).
        mp_no_cl_status, mp_no_cl_body, _ = asgi_post_jobs(
            client,
            ALICE | ct_header,
            [exact_sized_body[:300], exact_sized_body[300:]],
        )
        assert mp_no_cl_status == 202, mp_no_cl_body
        assert wait_for(client, mp_no_cl_body["id"])["status"] == "succeeded"

        mp_over_status, mp_over_body, _ = asgi_post_jobs(
            client,
            ALICE | ct_header,
            [over_body[:300], over_body[300:]],
        )
        assert mp_over_status == 422
        assert mp_over_body["detail"][0]["loc"] == ["body"]
        assert "maximum size" in mp_over_body["detail"][0]["msg"]

        json_ok_status, json_ok_body, _ = asgi_post_jobs(
            client,
            ALICE | {"Content-Type": "application/json"},
            [json_exact[:300], json_exact[300:]],
        )
        assert json_ok_status == 202, json_ok_body
        assert wait_for(client, json_ok_body["id"])["status"] == "succeeded"

        json_over_status, json_over_body, _ = asgi_post_jobs(
            client,
            ALICE | {"Content-Type": "application/json"},
            [json_over[:300], json_over[300:]],
        )
        assert json_over_status == 422
        assert json_over_body["detail"][0]["loc"] == ["body"]
        assert "maximum size" in json_over_body["detail"][0]["msg"]

    with running(tmp_path, backend=backend, CHRIEK_MAX_REFERENCES="2") as client:
        two_refs = post_multipart(
            client,
            data={"prompt": "ok", "width": "16", "height": "16"},
            files=[
                ("references", ("1.png", small_png, "image/png")),
                ("references", ("2.png", small_png, "image/png")),
            ],
        )
        assert two_refs.status_code == 202

        three_refs = post_multipart(
            client,
            data={"prompt": "too many", "width": "16", "height": "16"},
            files=[
                ("references", ("1.png", small_png, "image/png")),
                ("references", ("2.png", small_png, "image/png")),
                ("references", ("3.png", small_png, "image/png")),
            ],
        )
        assert three_refs.status_code == 422
        assert any("at most 2 reference images" in e["msg"] for e in three_refs.json()["detail"])

    with running(tmp_path, backend=backend, CHRIEK_MAX_REFERENCES="0") as client:
        zero_refs_ok = submit_multipart(
            client, data={"prompt": "text only", "width": "16", "height": "16"}
        )
        assert zero_refs_ok["params"]["reference_count"] == 0

        one_ref_rejected = post_multipart(
            client,
            data={"prompt": "no refs allowed", "width": "16", "height": "16"},
            files=[("references", ("1.png", small_png, "image/png"))],
        )
        assert one_ref_rejected.status_code == 422
        assert any(
            "at most 0 reference images" in e["msg"] for e in one_ref_rejected.json()["detail"]
        )


def test_streaming_upload_bound_aborts_early_and_valid_multipart_streams_across_small_chunks(
    tmp_path,
):
    ref_png = make_image_bytes((50, 100, 150), (16, 16))
    ct_header, valid_body = encode_multipart(
        {"prompt": "chunked edit", "seed": "9", "width": "16", "height": "16"},
        files=[("references", ("ref.png", ref_png, "image/png"))],
    )
    small_chunks = [valid_body[i : i + 7] for i in range(0, len(valid_body), 7)]
    assert len(small_chunks) > 20

    with running(tmp_path, CHRIEK_MAX_UPLOAD_BYTES="2048") as client:
        # Valid multipart split across 7-byte chunks with no Content-Length succeeds.
        status_code, created, consumed = asgi_post_jobs(client, ALICE | ct_header, small_chunks)
        assert status_code == 202, created
        assert consumed == len(small_chunks)
        finished = wait_for(client, created["id"])
        assert finished["status"] == "succeeded"
        assert finished["params"]["reference_count"] == 1

        # Over-limit multipart (with no files) aborts at chunk 6 (58-byte part header +
        # 5 * 400 = 2058 > 2048), leaving 16 of 22 chunks unread; whole-body loc ["body"].
        mp_header = b'--bound\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n'
        mp_chunks = [mp_header] + [b"a" * 400 for _ in range(20)] + [b"\r\n--bound--\r\n"]
        mp_status, mp_err, mp_consumed = asgi_post_jobs(
            client,
            ALICE | {"Content-Type": "multipart/form-data; boundary=bound"},
            mp_chunks,
        )
        assert mp_status == 422
        assert mp_err["detail"] == [
            {
                "type": "value_error",
                "loc": ["body"],
                "msg": "upload exceeds maximum size of 2048 bytes",
                "input": None,
            }
        ]
        assert mp_consumed == 6

        # Over-limit JSON aborts at chunk 7 (12 + 6 * 400 = 2412 > 2048) without consuming
        # the remaining 15 chunks.
        json_chunks = [b'{"prompt": "'] + [b"b" * 400 for _ in range(20)] + [b'"}']
        json_status, json_err, json_consumed = asgi_post_jobs(
            client,
            ALICE | {"Content-Type": "application/json"},
            json_chunks,
        )
        assert json_status == 422
        assert json_err["detail"] == [
            {
                "type": "value_error",
                "loc": ["body"],
                "msg": "upload exceeds maximum size of 2048 bytes",
                "input": None,
            }
        ]
        assert json_consumed == 7


@pytest.mark.parametrize(
    ("data", "files", "expected_loc"),
    [
        ({}, None, ["body", "prompt"]),
        ({"prompt": "   "}, None, ["body", "prompt"]),
        ({"prompt": "hi", "width": "1024.0"}, None, ["body", "width"]),
        ({"prompt": "hi", "width": "1000"}, None, ["body", "width"]),
        ({"prompt": "hi", "image_count": "0"}, None, ["body", "image_count"]),
        ({"prompt": "hi", "seed": "not-an-int"}, None, ["body", "seed"]),
        ({"prompt": "hi", "unknown": "1"}, None, ["body", "unknown"]),
        (
            {"prompt": "hi"},
            [("references", ("empty.png", b"", "image/png"))],
            ["body", "references", 0],
        ),
        (
            {"prompt": "hi"},
            [("references", ("corrupt.png", b"not-an-image", "image/png"))],
            ["body", "references", 0],
        ),
        (
            {"prompt": "hi"},
            [("other_file", ("ref.png", make_image_bytes(), "image/png"))],
            ["body", "other_file"],
        ),
        (
            {"prompt": "hi"},
            [("references", ("huge.png", make_image_bytes(size=(64, 64)), "image/png"))],
            ["body", "references", 0],
        ),
    ],
    ids=[
        "missing-prompt",
        "blank-prompt",
        "float-width",
        "non-multiple-width",
        "zero-images",
        "non-int-seed",
        "unknown-field",
        "empty-reference",
        "corrupt-reference",
        "unexpected-file-field",
        "reference-exceeds-max-pixels",
    ],
)
def test_invalid_multipart_requests_return_422_and_leave_no_files(
    tmp_path, data, files, expected_loc
):
    backend = CountingBackend()
    with running(tmp_path, backend=backend, CHRIEK_MAX_PIXELS="1024") as client:
        response = post_multipart(client, headers=ALICE, data=data, files=files)
        assert response.status_code == 422, response.text
        assert expected_loc in [err["loc"] for err in response.json()["detail"]]
        assert not any((tmp_path / "images").iterdir())
        assert backend.renders == 0


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("multipart/form-data", b"--bound--\r\n"),
        ("multipart/form-data; boundary=bound", b"not multipart at all"),
        (
            "multipart/form-data; boundary=bound",
            b'--bound\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nunterminated',
        ),
        (
            "multipart/form-data; boundary=bound",
            b'--bound\r\nContent-Disposition: form-data; name="prompt"\r\n\r\na\r\n'
            b'--bound\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nb\r\n'
            b"--bound--\r\n",
        ),
        (
            "multipart/form-data; boundary=bound",
            b'--bound\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n'
            b"\xff\r\n--bound--\r\n",
        ),
        (
            "multipart/form-data; boundary=bound",
            b"--bound\r\nContent-Disposition: form-data\r\n\r\na\r\n--bound--\r\n",
        ),
        (
            "multipart/form-data; boundary=bound",
            b'--bound\r\nContent-Disposition: form-data; name="\xff"\r\n\r\na\r\n--bound--\r\n',
        ),
    ],
    ids=[
        "missing-boundary",
        "malformed-multipart",
        "truncated-multipart",
        "duplicate-scalar-field",
        "invalid-utf8-scalar",
        "part-without-name",
        "invalid-utf8-field-name",
    ],
)
def test_malformed_raw_multipart_returns_422(tmp_path, content_type, body):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        response = client.post(
            "/jobs", headers=ALICE | {"Content-Type": content_type}, content=body
        )
        assert response.status_code == 422, response.text
        assert backend.renders == 0
        assert not any((tmp_path / "images").iterdir())
    with sqlite3.connect(tmp_path / "db" / "chriek.db") as conn:
        assert conn.execute("SELECT count(*) FROM jobs").fetchone() == (0,)


def test_form_field_count_and_scalar_size_are_rejected_during_receipt(tmp_path):
    headers = ALICE | {"Content-Type": "multipart/form-data; boundary=bound"}
    # One empty field per chunk; the 33rd part's headers exceed the 32-field bound.
    field_chunks = [
        f'--bound\r\nContent-Disposition: form-data; name="f{i}"\r\n\r\n\r\n'.encode()
        for i in range(40)
    ] + [b"--bound--\r\n"]
    # A 3 MiB scalar in 128 KiB chunks crosses the 1 MiB field bound at its 9th data chunk.
    scalar_chunks = (
        [b'--bound\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n']
        + [b"a" * (128 * 1024) for _ in range(24)]
        + [b"\r\n--bound--\r\n"]
    )
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        for chunks, expected_consumed in ((field_chunks, 33), (scalar_chunks, 10)):
            status_code, error, consumed = asgi_post_jobs(client, headers, chunks)
            assert status_code == 422, error
            assert consumed == expected_consumed
            assert len(error["detail"]) == 1
            assert len(json.dumps(error)) < 1024
        assert backend.renders == 0
        assert not any((tmp_path / "images").iterdir())


TRUNCATION_REFS = [
    make_image_bytes((200, 30, 30), (16, 16)),
    make_image_bytes((30, 30, 200), (24, 16)),
]


@pytest.mark.parametrize(
    "cut", ["after-boundary-with-crlf", "after-boundary-without-crlf", "mid-second-reference"]
)
def test_multipart_truncated_after_complete_parts_returns_422(tmp_path, cut):
    ct_header, body = encode_multipart(
        {"prompt": "truncated", "width": "16", "height": "16"},
        files=[
            ("references", (f"{i}.png", ref, "image/png")) for i, ref in enumerate(TRUNCATION_REFS)
        ],
        boundary="bound",
    )
    first_ref_end = body.index(TRUNCATION_REFS[0]) + len(TRUNCATION_REFS[0])
    cut_at = {
        "after-boundary-with-crlf": first_ref_end + len(b"\r\n--bound\r\n"),
        "after-boundary-without-crlf": first_ref_end + len(b"\r\n--bound"),
        "mid-second-reference": body.index(TRUNCATION_REFS[1]) + len(TRUNCATION_REFS[1]) // 2,
    }[cut]

    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        truncated = client.post("/jobs", headers=ALICE | ct_header, content=body[:cut_at])
        assert truncated.status_code == 422, truncated.text
        assert not any((tmp_path / "images").iterdir())

        complete = client.post("/jobs", headers=ALICE | ct_header, content=body)
        assert complete.status_code == 202, complete.text
        finished = wait_for(client, complete.json()["id"])
        assert finished["status"] == "succeeded"
        assert finished["params"]["reference_count"] == 2
        assert backend.renders == 1


def test_unauthenticated_multipart_upload_returns_401_before_parsing_or_reading_stream(tmp_path):
    chunks_read = 0

    def unauth_stream():
        nonlocal chunks_read
        for _ in range(10):
            chunks_read += 1
            yield b"malformed multipart garbage" * 100

    with running(tmp_path) as client:
        for headers in ({}, {"Authorization": "Bearer wrong-key"}):
            chunks_read = 0
            response = client.post(
                "/jobs",
                headers=headers | {"Content-Type": "multipart/form-data; boundary=bound"},
                content=unauth_stream(),
            )
            assert response.status_code == 401
            assert response.json() == {"detail": "missing or invalid API key"}
            assert chunks_read == 0, "unauthenticated request must not consume the upload stream"


def test_owner_isolation_holds_for_multipart_edit_jobs(client):
    alice_edit = submit_multipart(
        client,
        headers=ALICE,
        data={"prompt": "alice edit", "width": "16", "height": "16"},
        files=[("references", ("ref.png", make_image_bytes(), "image/png"))],
    )
    finished = wait_for(client, alice_edit["id"], headers=ALICE)
    assert finished["status"] == "succeeded"
    assert finished["params"]["reference_count"] == 1

    bob_poll = client.get(f"/jobs/{alice_edit['id']}", headers=BOB)
    assert bob_poll.status_code == 404
    assert bob_poll.json() == {"detail": "job not found"}


def _corrupt_exif_image(fmt: str) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = 6
    exif[0x010F] = "maker"
    # Tag 0x0100 (ImageWidth) with an ASCII string payload cannot be decoded as ImageWidth;
    # only the orientation tag is read, so this metadata is dropped rather than parsed.
    corrupt_exif = exif.tobytes().replace(b"\x01\x0f\x00\x02", b"\x01\x00\x00\x02")
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (30, 60, 90)).save(buf, format=fmt, exif=corrupt_exif)
    return buf.getvalue()


def _bomb_header_png(width: int = 20000, height: int = 20000) -> bytes:
    raw = make_image_bytes((10, 20, 30), (16, 16), fmt="PNG")
    ihdr_data = struct.pack(">II", width, height) + raw[24:29]
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    return raw[:16] + ihdr_data + ihdr_crc + raw[33:]


def test_reference_format_allowlist_corrupt_metadata_and_bomb_headers(tmp_path, monkeypatch):
    gs_marker = tmp_path / "gs_invoked"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gs = fake_bin / "gs"
    fake_gs.write_text(f"#!/bin/sh\n: > {gs_marker}\nexit 1\n")
    fake_gs.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")

    eps_bytes = make_image_bytes((10, 20, 30), (16, 16), fmt="EPS")

    gif_bytes = make_image_bytes((10, 20, 30), (16, 16), fmt="GIF")
    bmp_bytes = make_image_bytes((10, 20, 30), (16, 16), fmt="BMP")
    tiff_bytes = make_image_bytes((10, 20, 30), (16, 16), fmt="TIFF")
    blp_buf = io.BytesIO()
    Image.new("P", (16, 16)).save(blp_buf, format="BLP")
    dds_bytes = make_image_bytes((10, 20, 30), (16, 16), fmt="DDS")
    # Decodable by Pillow, but outside the allowlist (encoders installed with Pillow).
    jpeg2000_bytes = make_image_bytes((10, 20, 30), (16, 16), fmt="JPEG2000")
    ico_bytes = make_image_bytes((10, 20, 30), (16, 16), fmt="ICO")
    ppm_bytes = make_image_bytes((10, 20, 30), (16, 16), fmt="PPM")

    with running(tmp_path) as client:
        for name, payload in (
            ("vector.eps", eps_bytes),
            ("anim.gif", gif_bytes),
            ("bitmap.bmp", bmp_bytes),
            ("scan.tiff", tiff_bytes),
            ("texture.blp", blp_buf.getvalue()),
            ("surface.dds", dds_bytes),
            ("image.jp2", jpeg2000_bytes),
            ("favicon.ico", ico_bytes),
            ("pixmap.ppm", ppm_bytes),
            ("trunc.png", make_image_bytes(fmt="PNG")[:45]),
        ):
            resp = post_multipart(
                client,
                data={"prompt": "check format", "width": "16", "height": "16"},
                files=[("references", (name, payload, "application/octet-stream"))],
            )
            assert resp.status_code == 422, f"{name}: {resp.text}"
            detail = resp.json()["detail"]
            assert detail[0]["loc"] == ["body", "references", 0]
            assert "not a valid image" in detail[0]["msg"]

        assert not gs_marker.exists(), "EPS upload must never invoke external Ghostscript"

        bomb_resp = post_multipart(
            client,
            data={"prompt": "bomb check", "width": "16", "height": "16"},
            files=[("references", ("bomb.png", _bomb_header_png(), "image/png"))],
        )
        assert bomb_resp.status_code == 422
        bomb_detail = bomb_resp.json()["detail"]
        assert bomb_detail[0]["loc"] == ["body", "references", 0]
        assert "exceeds maximum of" in bomb_detail[0]["msg"]

        # Valid EXIF orientation on an allowed format (JPEG) is transposed and normalized;
        # an undecodable unrelated EXIF tag is dropped with the rest of the metadata.
        for name, payload, size in (
            ("oriented.jpg", _oriented_jpeg(), (16, 24)),
            ("bad-exif.jpg", _corrupt_exif_image("JPEG"), (16, 16)),
            ("bad-exif.webp", _corrupt_exif_image("WEBP"), (16, 16)),
        ):
            ok = submit_multipart(
                client,
                data={"prompt": "oriented", "width": "16", "height": "16"},
                files=[("references", (name, payload, "image/jpeg"))],
            )
            assert wait_for(client, ok["id"])["status"] == "succeeded", name
            with Image.open(reference_path(tmp_path / "images", ok["id"], 0)) as stored_ref:
                assert stored_ref.format == "PNG", name
                assert stored_ref.mode == "RGB", name
                assert stored_ref.size == size, name
                assert dict(stored_ref.getexif()) == {}, name


def _oriented_jpeg() -> bytes:
    exif = Image.Exif()
    exif[0x0112] = 6  # Rotate 90 degrees clockwise: 24x16 becomes 16x24
    buf = io.BytesIO()
    Image.new("RGB", (24, 16), (120, 40, 80)).save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def test_non_rgb_and_transparent_references_are_stored_as_plain_rgb(tmp_path):
    backend = CountingBackend()
    form = {"prompt": "normalize", "width": "16", "height": "16"}
    with running(tmp_path, backend=backend) as client:
        invalid = post_multipart(
            client,
            data=form,
            files=[("references", ("no-plte.png", _palette_png_without_plte(), "image/png"))],
        )
        assert invalid.status_code == 422, invalid.text
        assert invalid.json()["detail"][0]["loc"] == ["body", "references", 0]
        assert "not a valid image" in invalid.json()["detail"][0]["msg"]
        assert not any((tmp_path / "images").iterdir())
        with sqlite3.connect(tmp_path / "db" / "chriek.db") as conn:
            assert conn.execute("SELECT count(*) FROM jobs").fetchone() == (0,)

        for name, upload in (
            ("1-bit-trns", _png_bytes(Image.new("1", (16, 16), 1), transparency=0)),
            ("16-bit-gray-trns", _png_bytes(Image.new("I;16", (16, 16), 1), transparency=300)),
            (
                "rgb-colour-key",
                _png_bytes(Image.new("RGB", (16, 16), (1, 2, 3)), transparency=(1, 2, 3)),
            ),
            ("palette-trns", _png_bytes(Image.new("P", (16, 16), 0), transparency=0)),
            ("rgba-alpha", _png_bytes(Image.new("RGBA", (16, 16), (10, 20, 30, 0)))),
        ):
            job = submit_multipart(
                client, data=form, files=[("references", (f"{name}.png", upload, "image/png"))]
            )
            assert wait_for(client, job["id"])["status"] == "succeeded", name
            with Image.open(reference_path(tmp_path / "images", job["id"], 0)) as stored:
                assert stored.format == "PNG", name
                assert stored.mode == "RGB", name
                assert "transparency" not in stored.info, name
        assert backend.renders == 5


def test_reference_pixel_limit_includes_exact_boundary(tmp_path):
    form = {"prompt": "boundary", "width": "16", "height": "16"}
    with running(tmp_path, CHRIEK_MAX_PIXELS="1024") as client:
        exact = submit_multipart(
            client,
            data=form,
            files=[("references", ("exact.png", make_image_bytes(size=(32, 32)), "image/png"))],
        )
        assert wait_for(client, exact["id"])["status"] == "succeeded"

        over = post_multipart(
            client,
            data=form,
            files=[("references", ("over.png", make_image_bytes(size=(32, 33)), "image/png"))],
        )
        assert over.status_code == 422
        assert over.json()["detail"][0]["msg"] == "reference image 0 exceeds maximum of 1024 pixels"


def test_16_bit_greyscale_references_are_scaled_not_clipped(client, tmp_path):
    form = {"prompt": "deep grey", "seed": "5", "width": "16", "height": "16"}

    def grey16(value: int, **save_args) -> bytes:
        return _png_bytes(Image.new("I;16", (16, 16), value), **save_args)

    for value, expected, save_args in (
        (0x0100, 1, {}),
        (0x8000, 128, {"transparency": 300}),
        (0xFFFF, 255, {}),
    ):
        upload = grey16(value, **save_args)
        job = submit_multipart(
            client, data=form, files=[("references", ("grey.png", upload, "image/png"))]
        )
        assert wait_for(client, job["id"])["status"] == "succeeded"
        with Image.open(reference_path(tmp_path / "images", job["id"], 0)) as stored:
            assert stored.mode == "RGB"
            assert stored.getextrema() == ((expected, expected),) * 3, hex(value)

    dark, light = (
        wait_for(
            client,
            submit_multipart(
                client, data=form, files=[("references", ("grey.png", grey16(value), "image/png"))]
            )["id"],
        )
        for value in (0x4000, 0xC000)
    )
    assert download_pixels(client, dark) != download_pixels(client, light)


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP"])
def test_reference_icc_profile_is_dropped_so_large_profiles_stay_loadable(client, tmp_path, fmt):
    # 1,048,832 bytes: above Pillow's 1 MiB limit on a decompressed PNG iCCP profile.
    icc = bytes(range(256)) * 4097
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, format=fmt, icc_profile=icc)
    with Image.open(io.BytesIO(buf.getvalue())) as upload:
        assert upload.info["icc_profile"] == icc

    job = submit_multipart(
        client,
        data={"prompt": "print profile", "width": "16", "height": "16"},
        files=[("references", (f"ref.{fmt.lower()}", buf.getvalue(), f"image/{fmt.lower()}"))],
    )
    assert wait_for(client, job["id"])["status"] == "succeeded"
    with Image.open(reference_path(tmp_path / "images", job["id"], 0)) as stored:
        assert stored.format == "PNG"
        assert stored.mode == "RGB"
        assert "icc_profile" not in stored.info


_SUBMIT_FROM_STDIN = """
import json, sys
from fastapi.testclient import TestClient
from chriek.api import create_app
from chriek.config import Settings

env, headers = json.loads(sys.argv[1]), json.loads(sys.argv[2])
body = sys.stdin.buffer.read()
with TestClient(create_app(Settings.from_env(env)), raise_server_exceptions=False) as client:
    print(client.post("/jobs", headers=headers, content=body).status_code)
"""


@pytest.mark.parametrize(
    ("reference", "python_args", "env"),
    [
        # Between Pillow's warning and error thresholds Image.open only warns; as an error
        # the warning must still map to the pixel-limit 422.
        (_bomb_header_png(10000, 10000), ["-W", "error::RuntimeWarning"], {}),
        # `python -O` strips Pillow's palette assert; AttributeError follows instead.
        (_palette_png_without_plte(), [], {"PYTHONOPTIMIZE": "1"}),
    ],
    ids=["bomb-warning-as-error", "palette-without-plte-optimized"],
)
def test_invalid_reference_returns_422_in_non_default_interpreter_modes(
    tmp_path, reference, python_args, env
):
    """Runs the server in a subprocess so interpreter flags never leak into this suite."""
    ct_header, body = encode_multipart(
        {"prompt": "interpreter mode", "width": "16", "height": "16"},
        files=[("references", ("ref.png", reference, "image/png"))],
    )
    result = subprocess.run(
        [
            sys.executable,
            *python_args,
            "-c",
            _SUBMIT_FROM_STDIN,
            json.dumps(base_env(tmp_path)),
            json.dumps(ALICE | ct_header),
        ],
        input=body,
        capture_output=True,
        check=True,
        env=os.environ | env,
    )
    assert result.stdout.split()[-1] == b"422", result.stderr.decode()
    assert not any((tmp_path / "images").iterdir())
    with sqlite3.connect(tmp_path / "db" / "chriek.db") as conn:
        assert conn.execute("SELECT count(*) FROM jobs").fetchone() == (0,)


def test_mid_publish_failure_cleans_already_moved_outputs_and_keeps_references(
    tmp_path, monkeypatch
):
    ref_bytes = make_image_bytes((80, 120, 160))
    original_replace = Path.replace
    signer = UrlSigner(SIGNING_SECRET)

    def fail_on_second_output(self: Path, target: Path) -> Path:
        if self.parent.suffix == ".partial" and target.name == "1.png":
            raise OSError("disk full while publishing second image")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_on_second_output)

    with running(tmp_path) as client:
        job = submit_multipart(
            client,
            data={"prompt": "mid publish fail", "image_count": "2", "width": "16", "height": "16"},
            files=[("references", ("ref.png", ref_bytes, "image/png"))],
        )
        body = wait_for(client, job["id"])
        assert body["status"] == "failed"
        assert body["images"] == []
        job_dir = tmp_path / "images" / job["id"]
        assert not (tmp_path / "images" / f"{job['id']}.partial").exists()
        assert sorted(p.name for p in job_dir.iterdir()) == ["references"]
        assert sorted(p.name for p in (job_dir / "references").iterdir()) == ["0.png"]

        expires = 1_900_000_000
        sig0 = signer.sign(uuid.UUID(job["id"]), 0, expires)
        assert (
            client.get(f"/images/{job['id']}/0.png?expires={expires}&sig={sig0}").status_code == 404
        )


def test_references_exist_when_job_becomes_claimable_and_create_failure_cleans_up(
    tmp_path, monkeypatch, caplog
):
    ref_a = make_image_bytes((15, 85, 195), (16, 16))
    ref_b = make_image_bytes((195, 85, 15), (24, 16))
    original_create = JobStore.create
    storage_dir = tmp_path / "images"
    creating_thread: int | None = None
    claimed_at_first_commit: list[tuple[str, list[tuple[int, tuple[int, int] | None]]]] = []

    def claim_on_first_commit(_session: Session) -> None:
        # The worker's own claim, through an independent connection, at the instant the
        # first transaction inside `store.create` commits.
        nonlocal creating_thread
        if threading.get_ident() != creating_thread:
            return
        creating_thread = None  # only the first commit; also stops recursion via claim_next
        claimed = observer.claim_next(now=0.0)
        assert claimed is not None
        seen: list[tuple[int, tuple[int, int] | None]] = []
        for ref in claimed.references:
            path = reference_path(storage_dir, claimed.id, ref.index)
            if path.is_file():
                with Image.open(path) as img:
                    seen.append((ref.index, img.size))
            else:
                seen.append((ref.index, None))
        claimed_at_first_commit.append((claimed.id, seen))

    def observed_create(self: JobStore, *args, **kwargs):
        nonlocal creating_thread
        creating_thread = threading.get_ident()
        try:
            return original_create(self, *args, **kwargs)
        finally:
            creating_thread = None

    gated = GatedBackend()
    with running(tmp_path, backend=gated) as client:
        # Keep the worker busy so only the observer can claim the next queued job.
        blocker = submit(client, prompt="hold worker", width=16, height=16)
        assert gated.rendering.wait(5)
        observer = JobStore(tmp_path / "db" / "chriek.db")
        monkeypatch.setattr(JobStore, "create", observed_create)
        event.listen(Session, "after_commit", claim_on_first_commit)
        try:
            submitted = submit_multipart(
                client,
                data={"prompt": "claim-seam", "width": "16", "height": "16"},
                files=[
                    ("references", ("a.png", ref_a, "image/png")),
                    ("references", ("b.png", ref_b, "image/png")),
                ],
            )
        finally:
            event.remove(Session, "after_commit", claim_on_first_commit)
            monkeypatch.undo()
            observer.close()
            gated.release()
        assert claimed_at_first_commit == [(submitted["id"], [(0, (16, 16)), (1, (24, 16))])]
        assert wait_for(client, blocker["id"])["status"] == "succeeded"

    # Now force `store.create` to fail after `write_references` has written the files.
    def failing_create(self: JobStore, *args, **kwargs):
        raise OperationalError("INSERT", {}, RuntimeError("database is full"))

    monkeypatch.setattr(JobStore, "create", failing_create)
    fail_dir = tmp_path / "fail_case"
    app = create_app(Settings.from_env(base_env(fail_dir)))
    with TestClient(app, raise_server_exceptions=False) as fail_client:
        resp = post_multipart(
            fail_client,
            data={"prompt": "db failure", "width": "16", "height": "16"},
            files=[("references", ("a.png", ref_a, "image/png"))],
        )
        assert resp.status_code == 500
        assert list((fail_dir / "images").iterdir()) == []

    # Double fault: cleanup cannot remove the written reference. The client still sees the
    # original store failure, and the leftover file is logged.
    stuck_dir = tmp_path / "stuck_case"

    def failing_create_with_stuck_files(self: JobStore, *args, job_id: str, **kwargs):
        (stuck_dir / "images" / job_id / "references").chmod(0o500)
        raise OperationalError("INSERT", {}, RuntimeError("database is full"))

    monkeypatch.setattr(JobStore, "create", failing_create_with_stuck_files)
    app = create_app(Settings.from_env(base_env(stuck_dir)))
    try:
        with (
            TestClient(app) as stuck_client,
            caplog.at_level(logging.ERROR, "chriek.api"),
            pytest.raises(OperationalError, match="database is full"),
        ):
            post_multipart(
                stuck_client,
                data={"prompt": "db failure", "width": "16", "height": "16"},
                files=[("references", ("a.png", ref_a, "image/png"))],
            )
        (job_dir,) = (stuck_dir / "images").iterdir()
        assert (job_dir / "references" / "0.png").is_file()
        assert any(
            "could not remove" in record.getMessage() and record.name == "chriek.api"
            for record in caplog.records
        )
    finally:
        for path in (stuck_dir / "images").glob("*/references"):
            path.chmod(0o700)


def test_submit_decodes_and_persists_off_the_event_loop(tmp_path, monkeypatch):
    threads: dict[str, int] = {}
    original_decode = chriek.api._decode_reference_png
    original_write = chriek.api.write_references
    original_create = JobStore.create

    def recording_decode(*args):
        threads["decode"] = threading.get_ident()
        return original_decode(*args)

    def recording_write(*args):
        threads["write"] = threading.get_ident()
        return original_write(*args)

    def recording_create(self: JobStore, *args, **kwargs):
        threads["create"] = threading.get_ident()
        return original_create(self, *args, **kwargs)

    monkeypatch.setattr(chriek.api, "_decode_reference_png", recording_decode)
    monkeypatch.setattr(chriek.api, "write_references", recording_write)
    monkeypatch.setattr(JobStore, "create", recording_create)

    with running(tmp_path) as client:
        loop_thread = client.portal.call(threading.get_ident)
        job = submit_multipart(
            client,
            data={"prompt": "offloaded submit", "width": "16", "height": "16"},
            files=[("references", ("ref.png", make_image_bytes(), "image/png"))],
        )
        assert wait_for(client, job["id"])["status"] == "succeeded"

    assert set(threads) == {"decode", "write", "create"}
    assert loop_thread not in threads.values()


def test_cleanup_fault_and_missing_reference_fail_job_cleanly_without_stopping_worker(
    tmp_path, monkeypatch, caplog
):
    ref_bytes = make_image_bytes((70, 90, 110))
    original_replace = Path.replace
    original_unlink = Path.unlink
    signer = UrlSigner(SIGNING_SECRET)

    def fail_second_publish(self: Path, target: Path) -> Path:
        if self.parent.suffix == ".partial" and target.name == "1.png":
            raise OSError("publish failed after first output moved")
        return original_replace(self, target)

    def fault_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == "0.png" and self.parent.name != "references":
            raise PermissionError("simulated storage unlink fault")
        original_unlink(self, missing_ok=missing_ok)

    with running(tmp_path) as client:
        monkeypatch.setattr(Path, "replace", fail_second_publish)
        monkeypatch.setattr(Path, "unlink", fault_unlink)

        cleanup_job = submit_multipart(
            client,
            data={"prompt": "cleanup fault", "image_count": "2", "width": "16", "height": "16"},
            files=[("references", ("ref.png", ref_bytes, "image/png"))],
        )
        failed_cleanup = wait_for(client, cleanup_job["id"])
        assert failed_cleanup["status"] == "failed"
        assert (
            "image 2 of 2 failed: publish failed after first output moved"
            in failed_cleanup["error"]
        )
        assert client.get("/health").status_code == 200

        # Even though 0.png could not be unlinked from disk, the download guard rejects it.
        expires = 1_900_000_000
        sig0 = signer.sign(uuid.UUID(cleanup_job["id"]), 0, expires)
        assert (
            client.get(
                f"/images/{cleanup_job['id']}/0.png?expires={expires}&sig={sig0}"
            ).status_code
            == 404
        )
        assert any(
            "could not remove unpublished output" in record.getMessage()
            for record in caplog.records
            if record.levelno == logging.ERROR
        )

        monkeypatch.undo()

    # Next, a missing reference file (OSError) and a corrupt one (ValueError: a PNG whose
    # iCCP profile decompresses beyond the reader's limit) each fail only their job: the
    # error does not leak the absolute storage path, and subsequent jobs still succeed.
    gated_next = GatedBackend()
    with running(tmp_path, backend=gated_next) as client:
        blocker = submit(client, prompt="hold worker", width=16, height=16)
        assert gated_next.rendering.wait(5)

        missing_ref_job = submit_multipart(
            client,
            data={"prompt": "missing ref", "image_count": "2", "width": "16", "height": "16"},
            files=[
                ("references", ("r0.png", ref_bytes, "image/png")),
                ("references", ("r1.png", ref_bytes, "image/png")),
            ],
        )
        reference_path(tmp_path / "images", missing_ref_job["id"], 0).unlink()
        corrupt_ref_job = submit_multipart(
            client,
            data={"prompt": "corrupt ref", "width": "16", "height": "16"},
            files=[("references", ("r0.png", ref_bytes, "image/png"))],
        )
        Image.new("RGB", (16, 16)).save(
            reference_path(tmp_path / "images", corrupt_ref_job["id"], 0),
            format="PNG",
            icc_profile=b"\x00" * (1024 * 1024 + 1),
        )
        follow_up = submit(client, prompt="after missing ref", width=16, height=16)

        gated_next.release()
        assert wait_for(client, blocker["id"])["status"] == "succeeded"
        failed_ref = wait_for(client, missing_ref_job["id"])
        assert failed_ref["status"] == "failed"
        assert failed_ref["error"] == (
            "reference 1 of 2 could not be loaded: No such file or directory"
        )
        assert str(tmp_path) not in failed_ref["error"]
        assert wait_for(client, corrupt_ref_job["id"])["status"] == "failed"
        assert wait_for(client, follow_up["id"])["status"] == "succeeded"
        assert client.get("/health").status_code == 200


FORM_16 = {"prompt": "reference checks", "width": "16", "height": "16"}


def _noise_image(size: tuple[int, int] = (64, 64), mode: str = "RGB") -> Image.Image:
    """Deterministic non-uniform pixels: rotations, flips and lossy decodes all show."""
    noise = hashlib.shake_256(b"chriek").digest(size[0] * size[1] * 3)
    image = Image.frombytes("RGB", size, noise)
    return image.convert(mode)


def _jpeg(image: Image.Image | None = None, **save_args) -> bytes:
    buf = io.BytesIO()
    (_noise_image() if image is None else image).save(buf, format="JPEG", **save_args)
    return buf.getvalue()


def _after_first_segment(data: bytes, extra: bytes) -> bytes:
    """Insert `extra` right after the JPEG's first marker segment (Pillow writes APP0)."""
    end = 4 + int.from_bytes(data[4:6])
    return data[:end] + extra + data[end:]


def _with_segments(baseline: bytes, total: int) -> bytes:
    """Pad a single-scan JPEG with empty APP1 segments to `total` segments (SOS included)."""
    pos, before_scan = 2, 0
    while baseline[pos + 1] != 0xDA:
        pos += 2 + int.from_bytes(baseline[pos + 2 : pos + 4])
        before_scan += 1
    return baseline[:2] + b"\xff\xe1\x00\x02" * (total - before_scan - 1) + baseline[2:]


def _with_scans(progressive: bytes, total: int) -> bytes:
    """Repeat a progressive JPEG's last scan until the file has `total` scans."""
    last_scan = progressive[progressive.rindex(b"\xff\xda") : -2]
    extra = total - progressive.count(b"\xff\xda")
    return progressive[:-2] + last_scan * extra + b"\xff\xd9"


def _with_fill(data: bytes, fill: int) -> bytes:
    """Add `fill` 0xFF fill bytes before the marker after SOI."""
    return data[:2] + b"\xff" * fill + data[2:]


def _assert_nothing_persisted(tmp_path: Path, backend: CountingBackend) -> None:
    assert not any((tmp_path / "images").iterdir())
    with sqlite3.connect(tmp_path / "db" / "chriek.db") as conn:
        assert conn.execute("SELECT count(*) FROM jobs").fetchone() == (0,)
    assert backend.renders == 0


def _stored_reference(tmp_path: Path, job_id: str) -> Image.Image:
    with Image.open(reference_path(tmp_path / "images", job_id, 0)) as stored:
        stored.load()
        return stored


@pytest.mark.parametrize(
    "second",
    [
        pytest.param(b"not-an-image", id="corrupt"),
        pytest.param(b"", id="empty"),
        pytest.param(make_image_bytes(fmt="GIF"), id="unsupported-format"),
        pytest.param(make_image_bytes(size=(64, 64)), id="over-pixel-limit"),
        pytest.param(_with_segments(_jpeg(_noise_image((16, 16))), 257), id="complex-jpeg"),
    ],
)
def test_invalid_second_reference_rejects_the_whole_request(tmp_path, second):
    backend = CountingBackend()
    with running(tmp_path, backend=backend, CHRIEK_MAX_PIXELS="1024") as client:
        response = post_multipart(
            client,
            data=FORM_16,
            files=[
                ("references", ("first.png", make_image_bytes(), "image/png")),
                ("references", ("second.bin", second, "application/octet-stream")),
            ],
        )
        assert response.status_code == 422, response.text
        assert [err["loc"] for err in response.json()["detail"]] == [["body", "references", 1]]
        _assert_nothing_persisted(tmp_path, backend)


@pytest.mark.parametrize(
    "upload",
    [
        pytest.param(_jpeg(), id="baseline"),
        pytest.param(_jpeg(progressive=True), id="progressive"),
        pytest.param(_jpeg(progressive=True, restart_marker_blocks=1), id="restart-markers"),
        pytest.param(_with_fill(_jpeg(), 3), id="fill-bytes"),
    ],
)
def test_jpeg_references_keep_their_decoded_pixels(client, tmp_path, upload):
    assert b"\xff\x00" in upload  # the walk must skip stuffed bytes in scan data
    job = submit_multipart(
        client, data=FORM_16, files=[("references", ("r.jpg", upload, "image/jpeg"))]
    )
    assert wait_for(client, job["id"])["status"] == "succeeded"
    with Image.open(io.BytesIO(upload)) as source:
        expected = source.convert("RGB").tobytes()
    assert _stored_reference(tmp_path, job["id"]).tobytes() == expected


def test_jpeg_structure_bounds_hold_at_their_limits(tmp_path):
    baseline = _jpeg(_noise_image((16, 16)))
    progressive = _jpeg(_noise_image((16, 16)), progressive=True)
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        for name, upload in (
            ("256 segments", _with_segments(baseline, 256)),
            ("64 scans", _with_scans(progressive, 64)),
            ("256 fill bytes", _with_fill(baseline, 256)),
        ):
            job = submit_multipart(
                client, data=FORM_16, files=[("references", ("r.jpg", upload, "image/jpeg"))]
            )
            assert wait_for(client, job["id"])["status"] == "succeeded", name
        accepted = sorted(p.name for p in (tmp_path / "images").iterdir())

        for name, upload in (
            ("257 segments", _with_segments(baseline, 257)),
            ("65 scans", _with_scans(progressive, 65)),
            ("257 fill bytes", _with_fill(baseline, 257)),
            ("APP1 flood", _with_segments(baseline, 200_000)),
            ("scan flood", _with_scans(progressive, 5_000)),
        ):
            response = post_multipart(
                client, data=FORM_16, files=[("references", ("r.jpg", upload, "image/jpeg"))]
            )
            assert response.status_code == 422, f"{name}: {response.text}"
            [error] = response.json()["detail"]
            assert error["loc"] == ["body", "references", 0], name
            assert "JPEG structure" in error["msg"], name
        assert sorted(p.name for p in (tmp_path / "images").iterdir()) == accepted
        assert backend.renders == 3


@pytest.mark.parametrize(
    "upload",
    [
        pytest.param(_jpeg()[:20], id="truncated-segment"),
        pytest.param(_jpeg()[:-2], id="scan-without-eoi"),
        pytest.param(_after_first_segment(_jpeg(), b"\x00\x00"), id="junk-between-segments"),
        pytest.param(_after_first_segment(_jpeg(), b"\xff\x00"), id="stuffing-between-segments"),
        pytest.param(_after_first_segment(_jpeg(), b"\xff\xe1\x00\x00"), id="length-0"),
        pytest.param(_after_first_segment(_jpeg(), b"\xff\xe1\x00\x01"), id="length-1"),
        pytest.param(_after_first_segment(_jpeg(), b"\xff\xd8"), id="nested-soi"),
    ],
)
def test_malformed_jpeg_structure_is_rejected_before_decoding(tmp_path, upload):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        response = post_multipart(
            client, data=FORM_16, files=[("references", ("r.jpg", upload, "image/jpeg"))]
        )
        assert response.status_code == 422, response.text
        [error] = response.json()["detail"]
        assert error["loc"] == ["body", "references", 0]
        assert "not a valid image" in error["msg"]
        _assert_nothing_persisted(tmp_path, backend)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_is_applied_once_and_not_stored(client, tmp_path, orientation):
    exif = Image.Exif()
    exif[0x0112] = orientation
    upload = _png_bytes(_noise_image((4, 2)), exif=exif.tobytes())
    job = submit_multipart(
        client, data=FORM_16, files=[("references", ("r.png", upload, "image/png"))]
    )
    assert wait_for(client, job["id"])["status"] == "succeeded"
    with Image.open(io.BytesIO(upload)) as source:
        expected = ImageOps.exif_transpose(source)
    stored = _stored_reference(tmp_path, job["id"])
    assert (stored.size, stored.tobytes()) == (expected.size, expected.tobytes())
    assert dict(stored.getexif()) == {}


def _png_with_short_array_exif(payload_bytes: int) -> bytes:
    """24x16 PNG whose eXIf holds Orientation=6 plus one SHORT array of `payload_bytes`."""
    count = payload_bytes // 2
    entries = struct.pack(">HHIHH", 0x0112, 3, 1, 6, 0)
    entries += struct.pack(">HHII", 0x9999, 3, count, 38)  # data after the 30-byte IFD
    exif = b"MM\x00*" + struct.pack(">IH", 8, 2) + entries + b"\x00\x00\x00\x00"
    exif += b"\x00\x01" * count
    png = _png_bytes(_noise_image((24, 16)))
    return png[:33] + _png_chunk(b"eXIf", exif) + png[33:]


def test_large_exif_with_orientation_is_not_reserialized(client, tmp_path):
    upload = _png_with_short_array_exif(1024 * 1024)
    tracemalloc.start()
    try:
        response = post_multipart(
            client, data=FORM_16, files=[("references", ("r.png", upload, "image/png"))]
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert response.status_code == 202, response.text
    # Python-heap peak of the whole request, body copies included. Re-serialising the EXIF
    # (ImageOps.exif_transpose) turned every SHORT into Python objects: ~75x the upload.
    assert peak < 16 * len(upload)
    assert wait_for(client, response.json()["id"])["status"] == "succeeded"
    stored = _stored_reference(tmp_path, response.json()["id"])
    assert stored.size == (16, 24)
    assert dict(stored.getexif()) == {}


def test_cmyk_jpeg_reference_uses_plain_rgb_conversion(client, tmp_path):
    upload = _jpeg(_noise_image((16, 16), mode="CMYK"))
    job = submit_multipart(
        client, data=FORM_16, files=[("references", ("r.jpg", upload, "image/jpeg"))]
    )
    assert wait_for(client, job["id"])["status"] == "succeeded"
    with Image.open(io.BytesIO(upload)) as source:
        assert source.mode == "CMYK"
        expected = source.convert("RGB").tobytes()
    assert _stored_reference(tmp_path, job["id"]).tobytes() == expected


@pytest.mark.parametrize(
    ("fmt", "save_args"),
    [
        pytest.param("PNG", {}, id="apng-first-frame"),
        pytest.param("PNG", {"default_image": True}, id="apng-default-image"),
        pytest.param("WEBP", {"lossless": True}, id="animated-webp"),
        pytest.param("MPO", {}, id="mpo-first-picture"),
    ],
)
def test_multi_image_uploads_keep_only_the_first_picture(client, tmp_path, fmt, save_args):
    first, *later = (
        Image.new("RGB", (16, 16), c) for c in ((200, 30, 30), (30, 30, 200), (30, 200, 30))
    )
    buf = io.BytesIO()
    first.save(buf, format=fmt, save_all=True, append_images=later, **save_args)
    job = submit_multipart(
        client, data=FORM_16, files=[("references", ("r", buf.getvalue(), "image/octet-stream"))]
    )
    assert wait_for(client, job["id"])["status"] == "succeeded"
    red, green, blue = _stored_reference(tmp_path, job["id"]).getpixel((8, 8))
    assert red > 190 and green < 40 and blue < 40  # MPO is lossy JPEG
