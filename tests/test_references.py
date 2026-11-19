"""Issue #5: multipart reference images for edit jobs."""

import hashlib
import http.client
import io
import json
import logging
import os
import socket
import sqlite3
import struct
import subprocess
import sys
import threading
import urllib.request
import uuid
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from conftest import (
    ALICE,
    BOB,
    POSTGRES_URL,
    SIGNING_SECRET,
    SMALL,
    CountingBackend,
    FailingBackend,
    GatedBackend,
    StepBackend,
    base_env,
    connect_db,
    db_dump,
    make_image_bytes,
    post_job,
    proxy_env,
    running,
    stored_owners,
    submit,
    wait_for,
)
from fastapi.testclient import TestClient
from PIL import Image, ImageCms, ImageFile, ImageOps, PngImagePlugin
from sqlalchemy import Select, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

import chriek.api
from chriek.api import create_app
from chriek.auth import PROXY_OWNER
from chriek.backend import FakeBackend
from chriek.config import Settings
from chriek.signing import UrlSigner
from chriek.storage import StorageError
from chriek.store import JobStore
from chriek.worker import reference_path, write_references

MAX_SEED = 2**32 - 1
SMALL_FORM = {"width": "16", "height": "16"}


def encode_multipart(
    data: dict[str, str],
    files: list[tuple[str, tuple[str | None, bytes, str]]] | None = None,
    boundary: str = "chriek-multipart-boundary",
) -> tuple[dict[str, str], bytes]:
    parts: list[bytes] = []
    for name, value in data.items():
        header = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
        parts.append(f"{header}{value}\r\n".encode())
    for field_name, (filename, content, content_type) in files or []:
        # None omits the filename parameter, which RFC 7578 makes optional.
        filename_param = "" if filename is None else f'; filename="{filename}"'
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field_name}"'
            f"{filename_param}\r\nContent-Type: {content_type}\r\n\r\n".encode()
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


def _png_16x16(*middle: bytes, ihdr_size: int = 13) -> bytes:
    """Real 16x16 RGB PNG; `middle` chunks go between two halves of its IDAT data."""
    ihdr = struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)[:ihdr_size]
    idat = zlib.compress(b"\x00" * (16 * 3 + 1) * 16)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat[:10])
        + b"".join(middle)
        + _png_chunk(b"IDAT", idat[10:])
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


def test_multipart_accepts_the_highest_valid_seed(client):
    form = {"prompt": "harbor", "image_count": "2", "seed": str(MAX_SEED - 1)}
    body = wait_for(client, submit_multipart(client, data=form | SMALL_FORM)["id"])
    assert body["status"] == "succeeded"
    assert [image["seed"] for image in body["images"]] == [MAX_SEED - 1, MAX_SEED]


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


def test_every_reference_position_changes_the_fake_backend_output(client):
    """Only the middle reference differs, so first-only and last-only backends render alike."""
    first, middle, other_middle, last = (
        make_image_bytes(color, (16, 16))
        for color in ((220, 20, 20), (20, 40, 220), (20, 200, 40), (200, 200, 20))
    )
    form = {"prompt": "character panel", "seed": "12", "width": "32", "height": "32"}
    pixels = []
    for refs in ((first, middle, last), (first, other_middle, last)):
        job = wait_for(
            client,
            submit_multipart(
                client,
                data=form,
                files=[
                    ("references", (f"{i}.png", ref, "image/png")) for i, ref in enumerate(refs)
                ],
            )["id"],
        )
        assert job["status"] == "succeeded"
        assert job["params"]["reference_count"] == 3
        pixels.append(download_pixels(client, job))
    assert pixels[0] != pixels[1]


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


@pytest.mark.sqlite  # a database file created before reference support
def test_old_sqlite_jobs_without_references_report_zero_reference_count(tmp_path):
    db_path = tmp_path / "db" / "chriek.db"
    db_path.parent.mkdir(mode=0o700)
    db_path.touch(mode=0o600)
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


def _draft_db_with_queued_edit_job(tmp_path: Path, refs: list[bytes]) -> str:
    """A database as earlier drafts of this branch left it: associations in `"references"`."""
    store = JobStore(Path(base_env(tmp_path)["CHRIEK_DB_PATH"]))
    (tmp_path / "images").mkdir(mode=0o700)
    job_id = str(uuid.uuid7())
    write_references(tmp_path / "images", job_id, refs)
    store.create(
        job_id=job_id,
        owner="alice",
        prompt="draft edit",
        image_count=1,
        width=16,
        height=16,
        seed=55,
        reference_count=len(refs),
        now=1_800_000_000.0,
        max_queued=1,
    )
    store.close()
    with connect_db(tmp_path) as db:
        db.execute(text('ALTER TABLE job_references RENAME TO "references"'))
    return job_id


@pytest.mark.sqlite  # only SQLite drafts had the old table name
def test_draft_database_reference_table_is_renamed_keeping_associations_and_order(tmp_path):
    ref_a = make_image_bytes((111, 22, 33))
    ref_b = make_image_bytes((44, 155, 66))
    job_id = _draft_db_with_queued_edit_job(tmp_path, [ref_a, ref_b])

    with running(tmp_path) as client:
        survivor = wait_for(client, job_id)
        assert survivor["status"] == "succeeded"
        assert survivor["params"]["reference_count"] == 2
        direct = wait_for(
            client,
            submit_multipart(
                client,
                data={"prompt": "draft edit", "seed": "55", "width": "16", "height": "16"},
                files=[
                    ("references", ("a", ref_a, "image/png")),
                    ("references", ("b", ref_b, "image/png")),
                ],
            )["id"],
        )
        assert download_pixels(client, survivor) == download_pixels(client, direct)

    with running(tmp_path) as client:  # a second startup finds nothing left to rename
        assert client.get(f"/jobs/{job_id}", headers=ALICE).json()["params"]["reference_count"] == 2
    with connect_db(tmp_path) as db:
        tables = set(db.scalars(text("SELECT name FROM sqlite_master WHERE type = 'table'")))
        assert "references" not in tables
        rows = db.execute(
            text('SELECT job_id, "index" FROM job_references WHERE job_id = :id'), {"id": job_id}
        )
        assert sorted(map(tuple, rows)) == [(job_id, 0), (job_id, 1)]


@pytest.mark.sqlite
def test_startup_refuses_a_database_with_both_reference_tables(tmp_path):
    job_id = _draft_db_with_queued_edit_job(tmp_path, [make_image_bytes()])
    with connect_db(tmp_path) as db:
        db.execute(
            text(
                "CREATE TABLE job_references (job_id VARCHAR(36) NOT NULL, "
                '"index" INTEGER NOT NULL, PRIMARY KEY (job_id, "index"))'
            )
        )
        db.execute(
            text('INSERT INTO job_references (job_id, "index") VALUES (:id, 0)'), {"id": job_id}
        )
    before = db_dump(tmp_path)

    with pytest.raises(StorageError, match="job_references"):
        create_app(Settings.from_env(base_env(tmp_path)), backend=FakeBackend())
    assert db_dump(tmp_path) == before


@pytest.mark.skipif(POSTGRES_URL is None, reason="SQLite drafts' `references` table is renamed")
def test_postgres_never_renames_an_unrelated_references_table(tmp_path):
    with connect_db(tmp_path) as db:
        db.execute(text('CREATE TABLE "references" (note TEXT)'))
        db.execute(text("INSERT INTO \"references\" VALUES ('keep me')"))

    with running(tmp_path) as client:
        job = submit_multipart(
            client,
            data={"prompt": "edit", "width": "16", "height": "16"},
            files=[("references", ("r.png", make_image_bytes(), "image/png"))],
        )
        assert wait_for(client, job["id"])["status"] == "succeeded"
    with connect_db(tmp_path) as db:
        assert list(db.scalars(text('SELECT note FROM "references"'))) == ["keep me"]
        assert db.scalar(text("SELECT count(*) FROM job_references")) == 1


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
        assert failed_body["error"] == "image 2 of 2 failed"
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


@pytest.fixture
def network_attempts(monkeypatch) -> list:
    """Records, then refuses, every outbound connection: urllib, socket.create_connection and
    raw socket connects (which httpx, requests and http.client all end in).

    Recorded before raising, so code that swallows the network error is still seen.
    """
    attempts = []

    def refuse(entry):
        def guard(*args, **kwargs):
            attempts.append(entry)
            raise OSError(f"network access blocked in tests: {entry}")

        return guard

    monkeypatch.setattr(urllib.request, "urlopen", refuse("urlopen"))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket.socket, "connect", refuse("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", refuse("socket.connect_ex"))
    return attempts


def test_url_and_text_reference_fields_are_rejected_without_network_fetch(
    tmp_path, network_attempts
):
    url = "http://127.0.0.1:9/ref.png"
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        for bad_json in (
            {"prompt": "hi", "references": [url]},
            {"prompt": "hi", "reference_urls": [url]},
            {"prompt": "hi", "image_url": url},
        ):
            resp = client.post("/jobs", headers=ALICE, json=bad_json)
            assert resp.status_code == 422
            assert resp.json()["detail"]

        # A filename-less `references` part is an upload like any other: a URL in it is
        # bytes that fail image validation, never an address chriek fetches.
        text_reference = post_multipart(
            client, headers=ALICE, data={"prompt": "hi", "references": url}
        )
        assert text_reference.status_code == 422
        [error] = text_reference.json()["detail"]
        assert error["loc"] == ["body", "references", 0]
        assert error["msg"] == (
            "reference image 0 is not a valid image (supported formats: PNG, JPEG, WebP)"
        )
        assert url not in text_reference.text

        resp = post_multipart(client, headers=ALICE, data={"prompt": "hi", "reference_url": url})
        assert resp.status_code == 422
        assert resp.json()["detail"]
        assert backend.renders == 0
    assert network_attempts == []

    # Control: the guard sees the usual client paths, so an empty list above means no fetch.
    with pytest.raises(OSError):
        urllib.request.urlopen(url, timeout=1)
    with pytest.raises(OSError):
        http.client.HTTPConnection("127.0.0.1", 9, timeout=1).request("GET", "/ref.png")
    with socket.socket() as raw, pytest.raises(OSError):
        raw.connect(("127.0.0.1", 9))
    assert network_attempts == ["urlopen", "create_connection", "socket.connect"]


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

        # Missing Content-Length header over chunked ASGI stream.
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

    # Upper ends of the configurable ranges are usable: 10 references, a 512 MiB body cap.
    refs = [("references", (f"{i}.png", small_png, "image/png")) for i in range(11)]
    backend = CountingBackend()
    with running(
        tmp_path,
        backend=backend,
        CHRIEK_MAX_REFERENCES="10",
        CHRIEK_MAX_UPLOAD_BYTES=str(512 * 1024 * 1024),
    ) as client:
        before = _admission_state(tmp_path)
        eleven = post_multipart(client, data=FORM_16, files=refs)
        assert eleven.status_code == 422
        assert any("at most 10 reference images" in e["msg"] for e in eleven.json()["detail"])
        assert _admission_state(tmp_path) == before
        assert backend.renders == 0

        job = wait_for(client, submit_multipart(client, data=FORM_16, files=refs[:10])["id"])
        assert job["status"] == "succeeded"
        assert job["params"]["reference_count"] == 10
        download_pixels(client, job)


# Media types are case-insensitive: a mixed-case multipart body above 2 MiB is still exempt.
# A filename-less reference above the 1 MiB scalar-field cap is an upload too.
@pytest.mark.parametrize("filename", ["big.png", None])
@pytest.mark.parametrize("media_type", ["multipart/form-data", "Multipart/Form-Data"])
def test_json_and_multipart_body_caps_are_independent(tmp_path, media_type, filename):
    """JSON keeps the fixed 2 MiB cap whatever CHRIEK_MAX_UPLOAD_BYTES says, and vice versa."""
    json_body = json.dumps({"prompt": "small cap", "width": 16, "height": 16}).encode()
    json_body = json_body[:-1] + b" " * (4096 - len(json_body)) + b"}"
    with running(tmp_path, CHRIEK_MAX_UPLOAD_BYTES="1024") as client:
        response = client.post(
            "/jobs", headers=ALICE | {"Content-Type": "application/json"}, content=json_body
        )
        assert response.status_code == 202, response.text
        job = wait_for(client, response.json()["id"])
        assert job["status"] == "succeeded"
        assert job["params"]["prompt"] == "small cap"
        download_pixels(client, job)

    with running(tmp_path) as client:
        _, body = encode_multipart(
            FORM_16, files=[("references", (filename, _BIG_PNG, "image/png"))], boundary="big"
        )
        assert len(body) > 2 * 1024 * 1024
        response = client.post(
            "/jobs", headers=ALICE | {"Content-Type": f"{media_type}; boundary=big"}, content=body
        )
        assert response.status_code == 202, response.text
        job = wait_for(client, response.json()["id"])
        assert job["status"] == "succeeded"
        assert job["params"]["reference_count"] == 1
        download_pixels(client, job)


# Media types are case-insensitive: a mixed-case multipart body must still stream under the
# upload cap, not fall through to the (buffered) JSON path.
@pytest.mark.parametrize("media_type", ["multipart/form-data", "Multipart/Form-Data"])
def test_streaming_upload_bound_aborts_early_and_valid_multipart_streams_across_small_chunks(
    tmp_path, media_type
):
    ref_png = make_image_bytes((50, 100, 150), (16, 16))
    _, valid_body = encode_multipart(
        {"prompt": "chunked edit", "seed": "9", "width": "16", "height": "16"},
        files=[("references", ("ref.png", ref_png, "image/png"))],
        boundary="bound",
    )
    ct_header = {"Content-Type": f"{media_type}; boundary=bound"}
    small_chunks = [valid_body[i : i + 7] for i in range(0, len(valid_body), 7)]
    assert len(small_chunks) > 20

    with running(tmp_path, CHRIEK_MAX_UPLOAD_BYTES="2048") as client:
        # Valid multipart split across 7-byte chunks with no Content-Length succeeds.
        status_code, created, consumed = asgi_post_jobs(client, ALICE | ct_header, small_chunks)
        assert status_code == 202, created
        assert consumed == len(small_chunks)
        assert created["params"]["reference_count"] == 1
        finished = wait_for(client, created["id"])
        assert finished["status"] == "succeeded"
        assert finished["params"]["reference_count"] == 1
        download_pixels(client, finished)

        # Over-limit multipart (with no files) aborts at chunk 6 (58-byte part header +
        # 5 * 400 = 2058 > 2048), leaving 16 of 22 chunks unread; whole-body loc ["body"].
        mp_header = b'--bound\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n'
        mp_chunks = [mp_header] + [b"a" * 400 for _ in range(20)] + [b"\r\n--bound--\r\n"]
        mp_status, mp_err, mp_consumed = asgi_post_jobs(
            client,
            ALICE | ct_header,
            mp_chunks,
        )
        assert mp_status == 422
        assert mp_err["detail"] == [
            {
                "type": "value_error",
                "loc": ["body"],
                "msg": "upload exceeds maximum size of 2048 bytes",
            }
        ]
        assert mp_consumed == 6

        # A fifth reference part (default limit 4) stops receipt at its own chunk, although
        # the whole body is far below the byte cap: the count limit is enforced on its own.
        ref_part = (
            b"--bound\r\nContent-Disposition: form-data; "
            b'name="references"; filename="r"\r\n\r\nx\r\n'
        )
        count_chunks = [ref_part] * 5 + [b"z" * 10 for _ in range(10)] + [b"--bound--\r\n"]
        assert len(b"".join(count_chunks)) < 2048
        count_status, count_err, count_consumed = asgi_post_jobs(
            client,
            ALICE | ct_header,
            count_chunks,
        )
        assert count_status == 422
        assert "at most 4 reference images" in count_err["detail"][0]["msg"]
        assert count_consumed == 5


@pytest.mark.parametrize(
    ("data", "files", "expected_loc"),
    [
        ({}, None, ["body", "prompt"]),
        ({"prompt": "   "}, None, ["body", "prompt"]),
        (
            {"prompt": "a\x00b"},
            [("references", ("ref.png", make_image_bytes(), "image/png"))],
            ["body", "prompt"],
        ),
        ({"prompt": "hi", "width": "1024.0"}, None, ["body", "width"]),
        ({"prompt": "hi", "width": "1000"}, None, ["body", "width"]),
        ({"prompt": "hi", "image_count": "0"}, None, ["body", "image_count"]),
        ({"prompt": "hi", "seed": "not-an-int"}, None, ["body", "seed"]),
        ({"prompt": "hi", "seed": str(MAX_SEED + 1)} | SMALL_FORM, None, ["body", "seed"]),
        (
            {"prompt": "hi", "seed": str(MAX_SEED), "image_count": "2"} | SMALL_FORM,
            None,
            ["body", "seed"],
        ),
        ({"prompt": "hi", "seed": "9" * 5000} | SMALL_FORM, None, ["body", "seed"]),
        ({"prompt": "hi", "width": "9" * 5000}, None, ["body", "width"]),
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
        # A file part under a valid scalar name is rejected, not read as that scalar's value.
        (
            SMALL_FORM,
            [("prompt", ("p.txt", b"a lighthouse", "text/plain"))],
            ["body", "prompt"],
        ),
        (
            {"prompt": "hi"} | SMALL_FORM,
            [("seed", ("s.txt", b"7", "text/plain"))],
            ["body", "seed"],
        ),
        (
            {"prompt": "hi"},
            [("references", ("huge.png", make_image_bytes(size=(64, 64)), "image/png"))],
            ["body", "references", 0],
        ),
        # Real corrupt PNGs: Pillow raises SyntaxError (bad chunk type between IDATs, during
        # load) and ValueError (IHDR shorter than 13 bytes, during open).
        (
            {"prompt": "hi"},
            [("references", ("x.png", _png_16x16(_png_chunk(b"-\xf4\0\0", b"")), "image/png"))],
            ["body", "references", 0],
        ),
        (
            {"prompt": "hi"},
            [("references", ("x.png", _png_16x16(ihdr_size=12), "image/png"))],
            ["body", "references", 0],
        ),
    ],
    ids=[
        "missing-prompt",
        "blank-prompt",
        "nul-in-prompt",
        "float-width",
        "non-multiple-width",
        "zero-images",
        "non-int-seed",
        "seed-above-max",
        "seed-overflows-with-count",
        "5000-digit-seed",
        "5000-digit-width",
        "unknown-field",
        "empty-reference",
        "corrupt-reference",
        "unexpected-file-field",
        "prompt-file-part",
        "seed-file-part",
        "reference-exceeds-max-pixels",
        "png-bad-chunk-type-syntax-error",
        "png-truncated-ihdr-value-error",
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
        _assert_nothing_persisted(tmp_path, backend)


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
    assert stored_owners(tmp_path) == []


def test_multipart_part_header_names_are_case_insensitive(client):
    form = {"prompt": "lowercase headers", "seed": "7"} | SMALL_FORM
    files = [("references", ("ref.png", make_image_bytes((200, 30, 90)), "image/png"))]
    content_type, body = encode_multipart(form, files)
    lowered = body.replace(b"Content-Disposition:", b"content-disposition:").replace(
        b"Content-Type:", b"content-type:"
    )
    assert b"Content-" not in lowered

    response = client.post("/jobs", headers=ALICE | content_type, content=lowered)
    assert response.status_code == 202, response.text
    assert response.json()["params"]["reference_count"] == 1
    done = wait_for(client, response.json()["id"])
    assert done["status"] == "succeeded"
    assert done["params"]["reference_count"] == 1
    # Same output as canonical header names: the reference reached the backend unchanged.
    canonical = wait_for(client, submit_multipart(client, data=form, files=files)["id"])
    text_only = wait_for(client, submit_multipart(client, data=form)["id"])
    assert download_pixels(client, done) == download_pixels(client, canonical)
    assert download_pixels(client, done) != download_pixels(client, text_only)


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

    # Decodable by Pillow, but outside the allowlist (encoders installed with Pillow).
    unsupported = {
        "EPS": make_image_bytes((10, 20, 30), (16, 16), fmt="EPS"),
        "GIF": make_image_bytes((10, 20, 30), (16, 16), fmt="GIF"),
        "BMP": make_image_bytes((10, 20, 30), (16, 16), fmt="BMP"),
        "TIFF": make_image_bytes((10, 20, 30), (16, 16), fmt="TIFF"),
        "DDS": make_image_bytes((10, 20, 30), (16, 16), fmt="DDS"),
        "JPEG2000": make_image_bytes((10, 20, 30), (16, 16), fmt="JPEG2000"),
        "ICO": make_image_bytes((10, 20, 30), (16, 16), fmt="ICO"),
        "PPM": make_image_bytes((10, 20, 30), (16, 16), fmt="PPM"),
    }
    blp_buf = io.BytesIO()
    Image.new("P", (16, 16)).save(blp_buf, format="BLP")
    unsupported["BLP"] = blp_buf.getvalue()

    # The allowlist must keep these plugins' header parsers off untrusted bytes, not merely
    # reject what they parsed: record every entry into their `_open`.
    opened = []

    def recording_open(fmt, original):
        def _open(self):
            opened.append(fmt)
            return original(self)

        return _open

    Image.init()
    for fmt in unsupported:
        plugin = Image.OPEN[fmt][0]
        monkeypatch.setattr(plugin, "_open", recording_open(fmt, plugin._open))

    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        for name, payload in (
            *unsupported.items(),
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

        assert opened == []
        _assert_nothing_persisted(tmp_path, backend)
        assert not gs_marker.exists(), "EPS upload must never invoke external Ghostscript"
        # Control: an unrestricted open of each fixture enters its plugin's spied `_open`.
        for fmt, payload in unsupported.items():
            with Image.open(io.BytesIO(payload)):
                assert fmt in opened, fmt

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
        assert stored_owners(tmp_path) == []

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


def test_reference_pixel_limit_includes_exact_boundary(tmp_path, monkeypatch):
    form = {"prompt": "boundary", "width": "16", "height": "16"}
    exact_png = make_image_bytes(size=(32, 32))
    over_png = make_image_bytes(size=(25, 41))  # 1025 pixels, one over the limit
    loaded = []
    original_load = ImageFile.ImageFile.load

    def recording_load(self):
        loaded.append(self.size)
        return original_load(self)

    monkeypatch.setattr(ImageFile.ImageFile, "load", recording_load)
    backend = CountingBackend()
    with running(tmp_path, backend=backend, CHRIEK_MAX_PIXELS="1024") as client:
        # Far below Pillow's own bomb threshold: only chriek's size check, before decoding.
        over = post_multipart(
            client, data=form, files=[("references", ("over.png", over_png, "image/png"))]
        )
        assert over.status_code == 422
        assert over.json()["detail"][0]["msg"] == "reference image 0 exceeds maximum of 1024 pixels"
        assert loaded == []
        _assert_nothing_persisted(tmp_path, backend)

        exact = submit_multipart(
            client, data=form, files=[("references", ("exact.png", exact_png, "image/png"))]
        )
        assert loaded[0] == (32, 32)  # the spy sees the decode of a permitted reference
        assert wait_for(client, exact["id"])["status"] == "succeeded"


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
    assert stored_owners(tmp_path) == []


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
        observer = JobStore(Settings.from_env(base_env(tmp_path)).database)
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

    # A failure inside the admitting transaction, after `write_references` has written the
    # files but before COMMIT, rolls everything back: the references are removed too.
    fail_dir = tmp_path / "fail_case"
    app = create_app(Settings.from_env(base_env(fail_dir)))
    with (
        _fault_in_create(monkeypatch, "after_flush") as fired,
        TestClient(app, raise_server_exceptions=False) as fail_client,
    ):
        owners_before = stored_owners(fail_dir)
        resp = post_multipart(
            fail_client,
            data={"prompt": "db failure", "width": "16", "height": "16"},
            files=[("references", ("a.png", ref_a, "image/png"))],
        )
        assert fired == ["after_flush"]
        assert resp.status_code == 500
        assert list((fail_dir / "images").iterdir()) == []
    assert stored_owners(fail_dir) == owners_before

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


def test_second_reference_write_failure_removes_the_first_and_stores_no_job(tmp_path, monkeypatch):
    original_write = chriek.worker.write_private
    failed_at: list[Path] = []

    def fail_second_reference(path: Path, data: bytes) -> None:
        if path.name == "1.png" and path.parent.name == "references":
            # Only now, with reference 0 really on disk, does the write fail.
            assert path.with_name("0.png").is_file()
            failed_at.append(path)
            raise OSError(28, "No space left on device")
        original_write(path, data)

    monkeypatch.setattr(chriek.worker, "write_private", fail_second_reference)
    backend = CountingBackend()
    app = create_app(Settings.from_env(base_env(tmp_path)), backend=backend)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = post_multipart(
            client,
            data=FORM_16,
            files=[
                _reference_file("a.png", (15, 85, 195)),
                _reference_file("b.png", (195, 85, 15)),
            ],
        )
    assert len(failed_at) == 1
    assert resp.status_code == 500
    _assert_nothing_persisted(tmp_path, backend)


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
        assert failed_cleanup["error"] == "image 2 of 2 failed"
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

    # Next, a missing reference file (OSError), a corrupt one (ValueError: a PNG whose
    # iCCP profile decompresses beyond the reader's limit) and a valid non-PNG one (the
    # worker reopens stored references as PNG only) each fail only their job: the error
    # does not leak the absolute storage path, and subsequent jobs still succeed.
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
        jpeg_ref_job = submit_multipart(
            client,
            data={"prompt": "jpeg ref", "width": "16", "height": "16"},
            files=[("references", ("r0.png", ref_bytes, "image/png"))],
        )
        Image.new("RGB", (16, 16)).save(
            reference_path(tmp_path / "images", jpeg_ref_job["id"], 0), format="JPEG"
        )
        follow_up = submit(client, prompt="after missing ref", width=16, height=16)

        gated_next.release()
        assert wait_for(client, blocker["id"])["status"] == "succeeded"
        failed_ref = wait_for(client, missing_ref_job["id"])
        assert failed_ref["status"] == "failed"
        assert failed_ref["error"] == "reference 1 of 2 could not be loaded"
        assert str(tmp_path) not in failed_ref["error"]
        assert wait_for(client, corrupt_ref_job["id"])["status"] == "failed"
        failed_jpeg = wait_for(client, jpeg_ref_job["id"])
        assert failed_jpeg["status"] == "failed"
        assert failed_jpeg["error"] == "reference 1 of 1 could not be loaded"
        assert failed_jpeg["images"] == []
        assert wait_for(client, follow_up["id"])["status"] == "succeeded"
        # A job whose reference fails to load never reaches the backend.
        assert gated_next.prompts == ["hold worker", "after missing ref"]
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
    assert stored_owners(tmp_path) == []
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


def _ordinary_exif(orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x010F] = "Camera maker"
    exif[0x0110] = "Camera model"
    exif[0x0132] = "2026:11:11 12:00:00"
    exif[0x0112] = orientation
    return exif.tobytes()


def _encoded(fmt: str, image: Image.Image, **save_args) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt, **save_args)
    return buf.getvalue()


def _assert_stored_like_pillow_orients(tmp_path: Path, client, upload: bytes) -> Image.Image:
    """Submit `upload` and compare the stored pixels with Pillow's own orientation of it."""
    job = submit_multipart(
        client, data=FORM_16, files=[("references", ("r", upload, "application/octet-stream"))]
    )
    assert wait_for(client, job["id"])["status"] == "succeeded"
    with Image.open(io.BytesIO(upload)) as source:
        expected = ImageOps.exif_transpose(source).convert("RGB")
    stored = _stored_reference(tmp_path, job["id"])
    assert (stored.size, stored.tobytes()) == (expected.size, expected.tobytes())
    assert dict(stored.getexif()) == {}
    return stored


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_exif_orientation_is_applied_once_and_not_stored(client, tmp_path, fmt, orientation):
    # Non-uniform, non-square pixels: every rotation and flip gives different bytes. The
    # expected pixels are decoded from the upload itself, so lossy encoders compare exactly.
    upload = _encoded(fmt, _noise_image((24, 16)), exif=_ordinary_exif(orientation))
    stored = _assert_stored_like_pillow_orients(tmp_path, client, upload)
    assert stored.size == ((16, 24) if orientation >= 5 else (24, 16))


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP"])
def test_xmp_orientation_applies_when_exif_has_none(client, tmp_path, fmt):
    xmp = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
        b'tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>'
    )
    upload = _encoded(fmt, _noise_image((24, 16)), xmp=xmp)
    assert _assert_stored_like_pillow_orients(tmp_path, client, upload).size == (16, 24)


def _oriented_exif(size: int) -> bytes:
    """Big-endian TIFF of exactly `size` bytes: Orientation=6, then zero padding."""
    tiff = b"MM\x00*" + struct.pack(">IH", 8, 1) + struct.pack(">HHIHH", 0x0112, 3, 1, 6, 0)
    tiff += b"\x00\x00\x00\x00"  # no next IFD
    return tiff + b"\x00" * (size - len(tiff))


def _exif_array(tag: int, field_type: int, item: bytes, count: int) -> bytes:
    """Big-endian TIFF whose only tag holds `count` copies of `item`, stored after the IFD."""
    entry = struct.pack(">HHII", tag, field_type, count, 26)
    return b"MM\x00*" + struct.pack(">IH", 8, 1) + entry + b"\x00\x00\x00\x00" + item * count


def _jpeg_with_exif_segments(image: Image.Image, exif: bytes, parts: int) -> bytes:
    """JPEG whose EXIF is split over `parts` APP1 segments, which Pillow concatenates."""
    step = -(-len(exif) // parts)
    segments = b"".join(
        _app1_exif(piece) for piece in (exif[i : i + step] for i in range(0, len(exif), step))
    )
    return _after_first_segment(_jpeg(image), segments)


def _png_with_chunks(*chunks: bytes, at: int = 33) -> bytes:
    """Noise PNG with `chunks` inserted at byte `at`: after IHDR by default, -12 before IEND."""
    png = _png_bytes(_noise_image((24, 16)))
    return png[:at] + b"".join(chunks) + png[at:]


def _webp_chunk(kind: bytes, data: bytes) -> bytes:
    return kind + len(data).to_bytes(4, "little") + data + b"\x00" * (len(data) & 1)


def _webp_with_chunks(
    *chunks: bytes, in_first_frame: bool = False, top_level: bytes = b""
) -> bytes:
    """Extended-format WebP with `chunks` after VP8X, or after the first ANMF's frame data.

    `top_level` is inserted after VP8X as well, so one file can carry both.
    """
    frames = [_noise_image((24, 16)), Image.new("RGB", (24, 16), (30, 30, 200))]
    # A small EXIF makes Pillow write the extended format, with its VP8X header first.
    webp = bytearray(
        _encoded(
            "WEBP",
            frames[0],
            save_all=in_first_frame,
            append_images=frames[1:],
            exif=_ordinary_exif(1),
        )
    )
    extra = b"".join(chunks)
    if in_first_frame:
        anmf = webp.index(b"ANMF", 30)
        size = int.from_bytes(webp[anmf + 4 : anmf + 8], "little")
        webp[anmf + 4 : anmf + 8] = (size + len(extra)).to_bytes(4, "little")
        at = anmf + 8 + size
    else:
        at = 30  # RIFF header (12) + VP8X chunk (18)
    webp[at:at] = extra
    webp[30:30] = top_level
    webp[4:8] = (len(webp) - 8).to_bytes(4, "little")
    return bytes(webp)


def _webp_chunk_count(webp: bytes) -> int:
    """Chunks in a WebP: top-level, each ANMF frame and the chunks inside the frames."""
    pos, count = 12, 0
    while pos < len(webp):
        count += 1
        size = (
            16
            if webp.startswith(b"ANMF", pos)
            else int.from_bytes(webp[pos + 4 : pos + 8], "little")
        )
        pos += 8 + size + (size & 1)
    return count


def _reserved_jpeg_marker(code: int, payload: bytes) -> bytes:
    """A reserved JPG marker that Pillow reads without a length, so its payload is parsed."""
    return bytes((0xFF, code)) + (2 + len(payload)).to_bytes(2) + payload


def _app1(payload: bytes) -> bytes:
    return b"\xff\xe1" + (2 + len(payload)).to_bytes(2) + payload


def _app1_exif(exif: bytes) -> bytes:
    return _app1(b"Exif\x00\x00" + exif)


def _raw_exif_profile_itxt(flag: int, text: bytes) -> bytes:
    data = zlib.compress(text) if flag else text
    return _png_chunk(
        b"iTXt", b"Raw profile type exif\x00" + bytes((flag, 0)) + b"en\x00\x00" + data
    )


def _png_around_idat(before: bytes, after: bytes) -> bytes:
    """Noise PNG with `before` inserted after IHDR and `after` between IDAT and IEND."""
    png = _png_bytes(_noise_image((24, 16)))
    return png[:33] + before + png[33:-12] + after + png[-12:]


def _png_with_idat_chunks(png: bytes, size: int) -> bytes:
    """`png` (IHDR, IDAT..., IEND only) re-split into `size`-byte IDAT chunks, as libpng does."""
    pos, data = 33, b""
    while png[pos + 4 : pos + 8] == b"IDAT":
        length = int.from_bytes(png[pos : pos + 4])
        data += png[pos + 8 : pos + 8 + length]
        pos += 12 + length
    assert png[pos:] == _png_chunk(b"IEND", b"")
    chunks = (_png_chunk(b"IDAT", data[i : i + size]) for i in range(0, len(data), size))
    return png[:33] + b"".join(chunks) + png[pos:]


# 1024x1024 noise: over 2 MiB, at the default pixel limit, in more 8 KiB IDAT chunks than
# MAX_PNG_CHUNKS.
_BIG_PNG = _png_with_idat_chunks(_png_bytes(_noise_image((1024, 1024))), 8192)
_PNG_PRIVATE = _png_chunk(b"prVt", b"")
_PNG_ZTXT = _png_chunk(b"zTXt", b"k\x00\x00" + zlib.compress(b"text"))
_PNG_ICCP = _png_chunk(b"iCCP", b"icc\x00\x00" + zlib.compress(b"\x00" * 128))
# Nonzero compression flag, invalid UTF-8 text: Pillow inflates it, then drops it.
_PNG_ITXT_ZIP = _png_chunk(b"iTXt", b"k\x00\x01\x00\x00\x00" + zlib.compress(b"\xff\xfe"))
# Nonstandard nonzero flag: Pillow inflates any nonzero flag, so it must count as compressed.
_PNG_ITXT_FLAG_2 = _png_chunk(b"iTXt", b"k\x00\x02\x00\x00\x00" + zlib.compress(b"\xff\xfe"))
_PNG_ITXT_PLAIN = _png_chunk(b"iTXt", b"k\x00\x00\x00\x00\x00text")
_PNG_EMPTY_IDAT = _png_chunk(b"IDAT", b"")
# IHDR and IEND count towards MAX_PNG_CHUNKS too.
_PNG_SPARE_CHUNKS = chriek.api.MAX_PNG_CHUNKS - 2
_PNG_MAX_COMPRESSED = chriek.api.MAX_PNG_COMPRESSED_CHUNKS
# Fixed, not derived from MAX_PNG_CHUNKS, so a loosened budget fails the tests.
_PNG_FLOOD = 1000
_WEBP_UNKNOWN = _webp_chunk(b"XTRA", b"")
# A frame header with no image data: counted as a chunk although the walk steps into it.
_WEBP_EMPTY_FRAME = _webp_chunk(b"ANMF", b"\x00" * 16)
# The documented bound; fixed, not read from chriek.api, so a loosened budget fails the tests.
_WEBP_MAX_CHUNKS = 256
_WEBP_FLOOD = 1000


def _webp_shared_budget(extra: int) -> bytes:
    """Animated WebP with `_WEBP_MAX_CHUNKS + extra` chunks, the unknown ones split between
    the top level and the first frame so that neither half alone comes near the budget."""

    def animated(nested: int, top: int) -> bytes:
        return _webp_with_chunks(
            _WEBP_UNKNOWN * nested, in_first_frame=True, top_level=_WEBP_UNKNOWN * top
        )

    spare = _WEBP_MAX_CHUNKS - _webp_chunk_count(animated(0, 0))
    webp = animated(spare // 2 + extra, spare - spare // 2)
    assert _webp_chunk_count(webp) == _WEBP_MAX_CHUNKS + extra
    return webp


def _ordinary_png_with_profile_and_text() -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text("Title", "lighthouse")
    info.add_text("Comment", "harbor at dusk", zip=True)
    info.add_itxt("Description", "Leuchtturm", lang="de", zip=True)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return _png_bytes(_noise_image((24, 16)), icc_profile=profile, pnginfo=info)


def _with_exif(fmt: str, exif: bytes) -> bytes:
    if fmt == "JPEG":
        return _jpeg_with_exif_segments(_noise_image((24, 16)), exif, parts=2)
    return _encoded(fmt, _noise_image((24, 16)), exif=exif)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_exif_cap_holds_at_its_limit(tmp_path, fmt):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        at_cap = _with_exif(fmt, _oriented_exif(chriek.api.MAX_EXIF_BYTES))
        assert _assert_stored_like_pillow_orients(tmp_path, client, at_cap).size == (16, 24)
        stored = sorted(p.name for p in (tmp_path / "images").iterdir())

        over_cap = _with_exif(fmt, _oriented_exif(chriek.api.MAX_EXIF_BYTES + 1))
        response = post_multipart(
            client, data=FORM_16, files=[("references", ("r", over_cap, "image/x"))]
        )
        assert response.status_code == 422, response.text
        [error] = response.json()["detail"]
        assert error["loc"] == ["body", "references", 0]
        assert "too much EXIF metadata (at most 65536 bytes)" in error["msg"]
        assert sorted(p.name for p in (tmp_path / "images").iterdir()) == stored
        assert backend.renders == 1


@pytest.fixture
def decoder_entries(monkeypatch) -> list:
    """Records every `Image.open`: the boundary where Pillow's parsers start to run."""
    entries = []
    original_open = Image.open

    def recording_open(fp, *args, **kwargs):
        entries.append(fp)
        return original_open(fp, *args, **kwargs)

    monkeypatch.setattr(Image, "open", recording_open)
    return entries


_BASELINE_16 = _jpeg(_noise_image((16, 16)))


@pytest.mark.parametrize(
    ("upload", "reason"),
    [
        # Structure Pillow or libjpeg would have to skip or guess at.
        pytest.param(_jpeg()[:20], "not a valid image", id="truncated-segment"),
        pytest.param(
            _after_first_segment(_jpeg(), b"\x00\x00"),
            "not a valid image",
            id="junk-between-segments",
        ),
        pytest.param(
            _after_first_segment(_jpeg(), b"\xff\x00"),
            "not a valid image",
            id="stuffing-between-segments",
        ),
        pytest.param(
            _after_first_segment(_jpeg(), b"\xff\xe1\x00\x00"), "not a valid image", id="length-0"
        ),
        pytest.param(
            _after_first_segment(_jpeg(), b"\xff\xe1\x00\x01"), "not a valid image", id="length-1"
        ),
        pytest.param(
            _after_first_segment(_jpeg(), b"\xff\xd8"), "not a valid image", id="nested-soi"
        ),
        # Reserved JPG markers carry no length for Pillow, which parses their payload as
        # further segments: hidden EXIF and segment floods, at both ends of each family.
        *(
            pytest.param(
                _after_first_segment(
                    _BASELINE_16,
                    _reserved_jpeg_marker(code, _app1_exif(_oriented_exif(40_000)))
                    + _reserved_jpeg_marker(code, _app1_exif(_oriented_exif(40_000))),
                ),
                "not a valid image",
                id=f"exif-wrapped-in-{code:02X}",
            )
            for code in (0xC8, 0xF0, 0xFD)
        ),
        pytest.param(
            _after_first_segment(
                _BASELINE_16, _reserved_jpeg_marker(0xF8, b"\xff\xe0\x00\x02" * 16_000) * 4
            ),
            "not a valid image",
            id="app0-flood-wrapped-in-F8",
        ),
        # Pillow's header parser reads on past an EOI until the first SOS, keeping every
        # segment in Python objects.
        pytest.param(
            b"\xff\xd8\xff\xd9" + b"\xff\xe1\x00\x02" * 200_000 + _BASELINE_16[2:],
            "not a valid image",
            id="early-eoi-app1-flood",
        ),
        pytest.param(
            b"\xff\xd8\xff\xd9" + b"\xff\xfe\x00\x02" * 200_000 + _BASELINE_16[2:],
            "not a valid image",
            id="early-eoi-com-flood",
        ),
        # Only the walk's bounds reject these; Pillow alone accepts the RST and fill cases
        # (and fails a TEM marker only once it has opened the file).
        pytest.param(
            _after_first_segment(_BASELINE_16, b"\xff\xd0" * 300),
            "JPEG structure",
            id="rst-flood",
        ),
        pytest.param(
            _after_first_segment(_BASELINE_16, b"\xff\x01" * 300),
            "JPEG structure",
            id="tem-flood",
        ),
        pytest.param(
            _with_fill(_after_first_segment(_BASELINE_16, b"\xff" * 57), 200),
            "JPEG structure",
            id="fill-200-plus-57",
        ),
        pytest.param(
            _BASELINE_16[:-2] + b"\x00" * 4096, "not a valid image", id="scan-without-eoi"
        ),
        # EXIF that Pillow would decode into Python objects many times its size.
        pytest.param(
            _png_with_chunks(_png_chunk(b"eXIf", _exif_array(0x0112, 3, b"\x00\x06", 40_000))),
            "too much EXIF",
            id="png-orientation-short-array",
        ),
        pytest.param(
            _with_exif("JPEG", _exif_array(0x0112, 3, b"\x00\x06", 40_000)),
            "too much EXIF",
            id="jpeg-orientation-short-array",
        ),
        pytest.param(
            # Read by Pillow's JPEG open itself: no JFIF density, so it looks up the DPI.
            _jpeg_with_exif_segments(
                _noise_image((16, 16)), _exif_array(0x011A, 5, b"\x00\x00\x00\x48" * 2, 9_000), 2
            ),
            "too much EXIF",
            id="jpeg-xresolution-split-over-two-segments",
        ),
        pytest.param(
            # Without the nested-SOI check each FFD8 reads FFE1 as its length and skips the
            # whole 0xFFE1-byte APP1 (10 header bytes + TIFF) uncounted, while Pillow skips
            # the SOI and loads both payloads: 2x the cap.
            _BASELINE_16[:2]
            + (b"\xff\xd8" + _app1_exif(_oriented_exif(0xFFE1 - 10))) * 2
            + _BASELINE_16[2:],
            "not a valid image",
            id="nested-soi-exif-flood",
        ),
        pytest.param(
            _png_with_chunks(
                _png_chunk(b"eXIf", _oriented_exif(40_000)),
                _png_chunk(b"eXIf", _oriented_exif(40_000)),
            ),
            "too much EXIF",
            id="png-repeated-exif-chunks",
        ),
        pytest.param(
            _png_with_chunks(_png_chunk(b"tEXt", b"exif\x00" + _oriented_exif(70_000))),
            "too much EXIF",
            id="png-text-exif",
        ),
        pytest.param(
            _png_with_chunks(
                _png_chunk(b"zTXt", b"Raw profile type exif\x00\x00" + zlib.compress(b"0" * 70_000))
            ),
            "too much EXIF",
            id="png-compressed-raw-profile",
        ),
        pytest.param(
            _png_with_chunks(
                _png_chunk(b"iTXt", b"exif\x00\x01\x00en\x00\x00" + zlib.compress(b"0" * 70_000))
            ),
            "too much EXIF",
            id="png-compressed-international-text",
        ),
        pytest.param(_with_exif("WEBP", _oriented_exif(70_000)), "too much EXIF", id="webp-exif"),
        *(
            pytest.param(
                _png_with_chunks(_raw_exif_profile_itxt(flag, b"0" * 70_000)),
                "too much EXIF",
                id=f"png-international-text-flag-{flag}",
            )
            for flag in (0, 2, 255)  # Pillow inflates for every nonzero flag
        ),
        pytest.param(
            _png_with_chunks(_png_chunk(b"eXIf", _oriented_exif(70_000)), at=-12),
            "too much EXIF",
            id="png-exif-after-idat",
        ),
        pytest.param(
            _webp_with_chunks(
                _webp_chunk(b"ICCP", b"\x00" * 333), _webp_chunk(b"EXIF", _oriented_exif(70_000))
            ),
            "too much EXIF",
            id="webp-exif-after-odd-chunk",
        ),
        pytest.param(
            _webp_with_chunks(
                _webp_chunk(b"EXIF", _oriented_exif(40_000)),
                _webp_chunk(b"EXIF", _oriented_exif(40_000)),
            ),
            "too much EXIF",
            id="webp-repeated-exif-chunks",
        ),
        pytest.param(
            _webp_with_chunks(_webp_chunk(b"EXIF", _oriented_exif(70_000)), in_first_frame=True),
            "too much EXIF",
            id="webp-exif-inside-animation-frame",
        ),
        pytest.param(
            _webp_with_chunks(_WEBP_UNKNOWN * _WEBP_FLOOD),
            "too many WebP chunks",
            id="webp-unknown-chunk-flood",
        ),
        pytest.param(
            _webp_with_chunks(_WEBP_UNKNOWN * _WEBP_FLOOD, in_first_frame=True),
            "too many WebP chunks",
            id="webp-unknown-chunk-flood-inside-animation-frame",
        ),
        pytest.param(
            _webp_with_chunks(_WEBP_EMPTY_FRAME * _WEBP_FLOOD),
            "too many WebP chunks",
            id="webp-empty-frame-flood",
        ),
        pytest.param(
            _webp_shared_budget(1),
            f"too many WebP chunks (at most {_WEBP_MAX_CHUNKS},",
            id="webp-top-level-and-frame-chunks-one-over-budget",
        ),
        *(
            pytest.param(
                _png_around_idat(chunk * _PNG_MAX_COMPRESSED, chunk),
                "too many PNG chunks",
                id=f"png-{chunk[4:8].decode()}-flood-ending-after-idat",
            )
            for chunk in (_PNG_ZTXT, _PNG_ICCP, _PNG_ITXT_ZIP)
        ),
        pytest.param(
            _png_around_idat(_PNG_ITXT_FLAG_2 * _PNG_MAX_COMPRESSED, _PNG_ITXT_FLAG_2),
            "too many PNG chunks",
            id="png-iTXt-flag-2-flood-ending-after-idat",
        ),
        pytest.param(
            _png_around_idat(_PNG_ZTXT * 3 + _PNG_ICCP * 3, _PNG_ITXT_ZIP * 3),
            "too many PNG chunks",
            id="png-mixed-compressed-flood",
        ),
        pytest.param(
            _png_around_idat(_PNG_PRIVATE * 200, _PNG_PRIVATE * (_PNG_SPARE_CHUNKS - 199)),
            "too many PNG chunks",
            id="png-empty-chunk-flood-ending-after-idat",
        ),
        pytest.param(
            _png_around_idat(_PNG_EMPTY_IDAT * _PNG_FLOOD, b""),
            "too many PNG chunks",
            id="png-empty-idat-flood-before-data",
        ),
        pytest.param(
            _png_around_idat(b"", _PNG_EMPTY_IDAT * _PNG_FLOOD),
            "too many PNG chunks",
            id="png-empty-idat-flood-after-data",
        ),
        pytest.param(
            _png_with_idat_chunks(_png_bytes(_noise_image((24, 16))), 1),
            "too many PNG chunks",
            id="png-one-byte-idat-chunks",
        ),
        pytest.param(
            _png_around_idat(_PNG_PRIVATE * 100, _PNG_EMPTY_IDAT * (_PNG_SPARE_CHUNKS - 99)),
            "too many PNG chunks",
            id="png-short-idat-shares-chunk-budget",
        ),
        pytest.param(
            _png_with_idat_chunks(_BIG_PNG, chriek.api.MIN_UNCOUNTED_IDAT_BYTES - 1),
            "too many PNG chunks",
            id="png-idat-chunks-just-below-floor",
        ),
    ],
)
def test_costly_uploads_are_rejected_before_pillow_opens_them(
    tmp_path, decoder_entries, upload, reason
):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        response = post_multipart(
            client, data=FORM_16, files=[("references", ("r", upload, "image/x"))]
        )
        assert response.status_code == 422, response.text
        [error] = response.json()["detail"]
        assert error["loc"] == ["body", "references", 0]
        assert reason in error["msg"]
        assert decoder_entries == []
        _assert_nothing_persisted(tmp_path, backend)


# FastAPI's JSON rule, which chriek's own body decoding keeps: main type `application`,
# subtype `json` or `*+json`, case-insensitive.
@pytest.mark.parametrize(
    ("content_type", "accepted"),
    [
        ("application/vnd.api+json", True),
        ("Application/VND.API+JSON", True),
        ("text/x+json", False),
        ("text/plain", False),
    ],
)
def test_json_body_media_types_follow_fastapi_rule(tmp_path, content_type, accepted):
    body = json.dumps({"prompt": "media type", **SMALL}).encode()
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        response = client.post(
            "/jobs", headers=ALICE | {"Content-Type": content_type}, content=body
        )
        if not accepted:
            assert response.status_code == 422, response.text
            _assert_nothing_persisted(tmp_path, backend)
            return
        assert response.status_code == 202, response.text
        done = wait_for(client, response.json()["id"])
        assert done["status"] == "succeeded"
        assert done["params"]["reference_count"] == 0
        assert done["params"]["prompt"] == "media type"
        download_pixels(client, done)


def test_bug_in_the_structure_walk_is_logged_and_not_reported_as_an_invalid_image(
    tmp_path, monkeypatch, caplog
):
    secret = "walker-bug-detail-SENTINEL"

    def broken_walk(data: bytes) -> None:
        raise KeyError(secret)

    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        monkeypatch.setattr(chriek.api, "_check_png_structure", broken_walk)
        response = post_multipart(
            client, data=FORM_16, files=[("references", ("r.png", make_image_bytes(), "image/png"))]
        )
        assert response.status_code == 500
        assert secret not in response.text
        assert "not a valid image" not in response.text
        _assert_nothing_persisted(tmp_path, backend)
        [record] = [r for r in caplog.records if r.name == "chriek.api" and r.exc_info]
        assert record.levelno == logging.ERROR
        assert record.exc_info[0] is KeyError

        monkeypatch.undo()
        healthy = wait_for(
            client,
            submit_multipart(
                client,
                data=FORM_16,
                files=[("references", ("r.png", make_image_bytes(), "image/png"))],
            )["id"],
        )
        assert healthy["status"] == "succeeded"


_INFLATE_BOMB = zlib.compress(b"0" * 1_000_000)


@pytest.mark.parametrize(
    "exif_text",
    [
        pytest.param(
            _png_chunk(b"zTXt", b"Raw profile type exif\x00\x00" + _INFLATE_BOMB), id="ztxt-raw"
        ),
        pytest.param(_png_chunk(b"zTXt", b"exif\x00\x00" + _INFLATE_BOMB), id="ztxt-exif"),
        pytest.param(
            _png_chunk(b"iTXt", b"Raw profile type exif\x00\x01\x00en\x00\x00" + _INFLATE_BOMB),
            id="itxt-raw",
        ),
        pytest.param(
            _png_chunk(b"iTXt", b"exif\x00\x01\x00en\x00\x00" + _INFLATE_BOMB), id="itxt-exif"
        ),
    ],
)
def test_compressed_exif_text_inflates_only_up_to_the_remaining_budget(
    tmp_path, monkeypatch, decoder_entries, exif_text
):
    preceding = 40_000  # plain EXIF text before the compressed chunk, 25,536 bytes remain
    upload = _png_with_chunks(_png_chunk(b"tEXt", b"exif\x00" + b"0" * preceding), exif_text)
    inflate_limits = []
    real_decompressobj = zlib.decompressobj

    class RecordingInflate:
        """Forwards bounded inflation; refuses an unbounded call instead of allocating it."""

        def __init__(self, *args, **kwargs):
            self._real = real_decompressobj(*args, **kwargs)

        def decompress(self, data, max_length=0):
            inflate_limits.append(max_length)
            if max_length <= 0:
                raise zlib.error("unbounded inflate refused in test")
            return self._real.decompress(data, max_length)

    monkeypatch.setattr(zlib, "decompressobj", RecordingInflate)
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        response = post_multipart(
            client, data=FORM_16, files=[("references", ("r", upload, "image/png"))]
        )
        assert response.status_code == 422, response.text
        [error] = response.json()["detail"]
        assert error["loc"] == ["body", "references", 0]
        assert "too much EXIF" in error["msg"]
        # Non-empty: the walker's inflation was observed. Each call may produce at most one
        # byte more than the budget left after the preceding EXIF text.
        assert inflate_limits
        assert all(0 < limit <= 64 * 1024 - preceding + 1 for limit in inflate_limits)
        assert decoder_entries == []
        _assert_nothing_persisted(tmp_path, backend)


@pytest.mark.parametrize(
    "upload",
    [
        pytest.param(_jpeg(_noise_image((16, 16)), progressive=True), id="progressive-jpeg"),
        # APP15 and COM border the reserved F0-FD markers and stay length-bearing.
        pytest.param(
            _after_first_segment(_BASELINE_16, b"\xff\xef\x00\x04ab\xff\xfe\x00\x04cd"),
            id="jpeg-app15-and-com",
        ),
        pytest.param(_png_with_chunks(_raw_exif_profile_itxt(0, b"0" * 60_000)), id="itxt-plain"),
        pytest.param(_png_with_chunks(_raw_exif_profile_itxt(1, b"0" * 60_000)), id="itxt-zlib"),
        pytest.param(
            _webp_with_chunks(_webp_chunk(b"EXIF", _ordinary_exif(1)), in_first_frame=True),
            id="webp-small-exif-inside-animation-frame",
        ),
        pytest.param(_webp_shared_budget(0), id="webp-top-level-and-frame-chunks-at-budget"),
        pytest.param(_ordinary_png_with_profile_and_text(), id="png-icc-profile-and-text"),
        pytest.param(_BIG_PNG, id="png-many-8k-idat-chunks"),
        pytest.param(
            _png_with_idat_chunks(_BIG_PNG, chriek.api.MIN_UNCOUNTED_IDAT_BYTES),
            id="png-many-idat-chunks-at-floor",
        ),
        pytest.param(
            _png_around_idat(_PNG_PRIVATE * 100, _PNG_EMPTY_IDAT * (_PNG_SPARE_CHUNKS - 100)),
            id="png-short-idat-at-chunk-budget",
        ),
        pytest.param(
            _png_around_idat(_PNG_PRIVATE * 200, _PNG_PRIVATE * (_PNG_SPARE_CHUNKS - 200)),
            id="png-chunk-count-at-limit",
        ),
        pytest.param(
            _png_around_idat(_PNG_ZTXT * 3 + _PNG_ICCP * 3, _PNG_ITXT_ZIP * 2),
            id="png-compressed-count-at-limit",
        ),
        pytest.param(
            _png_around_idat(_PNG_ITXT_PLAIN * _PNG_MAX_COMPRESSED, _PNG_ITXT_PLAIN),
            id="png-uncompressed-itxt-not-counted",
        ),
        # Large non-EXIF metadata stays outside the EXIF budget: XMP and extended-XMP APP1
        # segments (together over 64 KiB), text after IEND, a 70 KB text chunk whose keyword
        # only starts with "exif", and an EXIF text chunk whose zlib data is corrupt.
        pytest.param(
            _after_first_segment(
                _BASELINE_16,
                _app1(b"http://ns.adobe.com/xap/1.0/\x00" + b"x" * 60_000)
                + _app1(b"http://ns.adobe.com/xmp/extension/\x00" + b"y" * 60_000),
            ),
            id="jpeg-large-xmp-app1",
        ),
        pytest.param(
            _png_bytes(_noise_image((24, 16))) + _png_chunk(b"tEXt", b"k\x00v") * 300,
            id="png-text-chunks-after-iend",
        ),
        pytest.param(
            _png_with_chunks(_png_chunk(b"tEXt", b"exifnote\x00" + b"0" * 70_000)),
            id="png-large-text-with-exif-prefixed-keyword",
        ),
        pytest.param(
            _png_with_chunks(_png_chunk(b"zTXt", b"Raw profile type exif\x00\x00not zlib")),
            id="png-exif-ztxt-with-corrupt-zlib",
        ),
    ],
)
def test_ordinary_uploads_pass_the_walk_and_reach_the_decoder(tmp_path, decoder_entries, upload):
    with running(tmp_path) as client:
        job = submit_multipart(
            client, data=FORM_16, files=[("references", ("r", upload, "image/x"))]
        )
        assert job["params"]["reference_count"] == 1
        done = wait_for(client, job["id"])
        assert done["status"] == "succeeded"
        download_pixels(client, done)
    # Positive control for the recorder: it sees the submission's in-memory decode.
    assert any(isinstance(entry, io.BytesIO) for entry in decoder_entries)


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


class RecordingStepBackend(StepBackend):
    """StepBackend that also records each render's reference top-left pixels, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.references: dict[str, list[tuple[int, ...]]] = {}

    def render(self, prompt, width, height, seed, references=()):
        self.references[prompt] = [ref.getpixel((0, 0)) for ref in references]
        return super().render(prompt, width, height, seed, references)


def _reference_file(name: str, color: tuple[int, int, int]):
    return ("references", (name, make_image_bytes(color), "image/png"))


def test_edit_jobs_report_backend_metrics_only_when_the_backend_ran(tmp_path):
    class MeasuredBackend(RecordingStepBackend):
        """Counts `start_job` calls; reports them and the renders seen so far as run details."""

        jobs_started = 0

        def start_job(self):
            self.jobs_started += 1

        def run_details(self):
            return {"jobs_started": self.jobs_started, "renders": len(self.prompts)}

        def render(self, prompt, width, height, seed, references=()):
            image = super().render(prompt, width, height, seed, references)
            if prompt == "fail":
                raise RuntimeError("render failed")
            return image

    backend = MeasuredBackend()
    colors = [(200, 10, 10), (10, 10, 200)]
    with running(tmp_path, backend=backend) as client:
        submit(client, prompt="hold", width=16, height=16)
        backend.wait_started(1)
        edit = submit_multipart(
            client,
            data=FORM_16 | {"prompt": "edit", "image_count": "2"},
            files=[_reference_file(f"{i}.png", color) for i, color in enumerate(colors)],
        )
        failing = submit_multipart(
            client, data=FORM_16 | {"prompt": "fail"}, files=[_reference_file("r", colors[0])]
        )
        missing = submit_multipart(
            client, data=FORM_16 | {"prompt": "missing"}, files=[_reference_file("r", colors[1])]
        )
        reference_path(tmp_path / "images", missing["id"], 0).unlink()
        backend.step(4)

        edited = wait_for(client, edit["id"])
        assert (edited["status"], edited["params"]["reference_count"]) == ("succeeded", 2)
        assert (edited["run"]["jobs_started"], edited["run"]["renders"]) == (2, 3)
        assert backend.references["edit"] == colors
        assert [client.get(image["url"]).status_code for image in edited["images"]] == [200, 200]

        failed = wait_for(client, failing["id"])
        assert (failed["status"], failed["error"]) == ("failed", "image 1 of 1 failed")
        assert (failed["run"]["jobs_started"], failed["run"]["renders"]) == (3, 4)

        unloaded = wait_for(client, missing["id"])
        assert unloaded["error"] == "reference 1 of 1 could not be loaded"
        assert set(unloaded["run"]) == {"backend", "model", "revision", "render_seconds"}
        assert (backend.jobs_started, backend.prompts) == (3, ["hold", "edit", "edit", "fail"])

    # The failed render published no output and kept its private reference.
    stored = tmp_path / "images" / failing["id"]
    assert [p.relative_to(stored).as_posix() for p in stored.rglob("*.png")] == ["references/0.png"]
    assert _stored_reference(tmp_path, failing["id"]).getpixel((0, 0)) == colors[0]


def _admission_state(tmp_path: Path) -> tuple[str, list[Path]]:
    """Database dump and storage tree, to prove a rejected admission changed neither."""
    storage = Path(base_env(tmp_path)["CHRIEK_STORAGE_DIR"])
    return db_dump(tmp_path), sorted(p.relative_to(storage) for p in storage.rglob("*"))


def test_full_queue_rejects_every_transport_and_leaves_no_reference_artifacts(tmp_path):
    red, blue, green = (200, 0, 0), (0, 0, 200), (0, 200, 0)
    backend = RecordingStepBackend()
    with running(tmp_path, backend=backend, CHRIEK_MAX_QUEUED_JOBS="1") as client:
        first = submit(client, prompt="running", **SMALL)
        backend.wait_started(1)  # running jobs free their slot
        queued = submit_multipart(
            client,
            data=FORM_16 | {"prompt": "queued"},
            files=[_reference_file("a.png", red), _reference_file("b.png", blue)],
        )
        before = _admission_state(tmp_path)

        rejected = [
            post_job(client, "json", **SMALL),
            post_multipart(client, data=FORM_16 | {"prompt": "text-only"}),
            post_multipart(
                client, data=FORM_16 | {"prompt": "edit"}, files=[_reference_file("c.png", green)]
            ),
        ]

        for response in rejected:
            assert response.status_code == 429, response.text
            assert "location" not in response.headers
        assert _admission_state(tmp_path) == before, (
            "a rejected admission must leave no row or reference file"
        )
        other_owner = submit_multipart(
            client,
            headers=BOB,
            data=FORM_16 | {"prompt": "bob"},
            files=[_reference_file("d", green)],
        )

        backend.step(3)
        for job, owner in ((first, ALICE), (queued, ALICE), (other_owner, BOB)):
            assert wait_for(client, job["id"], headers=owner)["status"] == "succeeded"
        follow_up = submit_multipart(
            client, data=FORM_16 | {"prompt": "follow-up"}, files=[_reference_file("e", blue)]
        )
        backend.step()
        assert wait_for(client, follow_up["id"])["params"]["reference_count"] == 1
    assert backend.prompts == ["running", "queued", "bob", "follow-up"]
    assert backend.references == {
        "running": [],
        "queued": [red, blue],
        "bob": [green],
        "follow-up": [blue],
    }


def test_admission_lost_after_the_early_count_saw_room_is_429_without_artifacts(
    tmp_path, monkeypatch
):
    red, blue = (200, 0, 0), (0, 0, 200)
    original_scalar = Session.scalar
    early_counts: list[int] = []

    def early_count_sees_room(self, statement, *args, **kwargs):
        # Only the standalone pre-check is a plain SELECT count; the admitting
        # INSERT ... SELECT still counts for real, as a competing admission would leave it.
        if isinstance(statement, Select) and "count(" in str(statement):
            early_counts.append(original_scalar(self, statement, *args, **kwargs))
            return 0
        return original_scalar(self, statement, *args, **kwargs)

    backend = StepBackend()
    with running(tmp_path, backend=backend, CHRIEK_MAX_QUEUED_JOBS="1") as client:
        try:
            first = submit(client, prompt="running", **SMALL)
            backend.wait_started(1)
            queued = submit(client, prompt="queued", **SMALL)
            before = _admission_state(tmp_path)

            with monkeypatch.context() as patch:
                patch.setattr(Session, "scalar", early_count_sees_room)
                rejected = post_multipart(
                    client,
                    data=FORM_16 | {"prompt": "raced"},
                    files=[_reference_file("a.png", red), _reference_file("b.png", blue)],
                )

            assert early_counts == [1], "the early count saw the full queue and was overridden"
            assert rejected.status_code == 429, rejected.text
            assert "location" not in rejected.headers
            assert _admission_state(tmp_path) == before, (
                "a lost admission must leave no row or reference file"
            )
        finally:
            backend.step(2)
        for job in (first, queued):
            assert wait_for(client, job["id"])["status"] == "succeeded"
    assert backend.prompts == ["running", "queued"]


def test_proxy_mode_edit_jobs_use_the_shared_owner_and_the_host_check(tmp_path):
    storage = Path(base_env(tmp_path)["CHRIEK_STORAGE_DIR"])
    files = [_reference_file("a.png", (200, 0, 0))]
    with running(tmp_path, env_base=proxy_env) as client:
        job = submit_multipart(
            client, headers={"X-Forwarded-User": "carol"}, data=FORM_16, files=files
        )
        body = wait_for(client, job["id"], headers={})
        assert (body["status"], body["params"]["reference_count"]) == ("succeeded", 1)
        jobs_before = sorted(p.name for p in storage.iterdir())

        hostile = post_multipart(
            client, headers={"Host": "evil.example"}, data=FORM_16, files=files
        )

        assert (hostile.status_code, hostile.text) == (400, "Invalid host header")
        assert sorted(p.name for p in storage.iterdir()) == jobs_before
    assert stored_owners(tmp_path) == [PROXY_OWNER]


class _Interrupted(BaseException):
    """A non-Exception failure, as a KeyboardInterrupt or SystemExit would be."""


@contextmanager
def _fault_in_create(
    monkeypatch, event_name: str, error: BaseException | None = None
) -> Iterator[list[str]]:
    """Raise once from Session event `event_name` inside the next `JobStore.create` call.

    `after_flush` fails the admitting transaction before COMMIT (a definite rollback);
    `before_commit` fails the commit call before the database commits anything, and
    `after_commit` loses the acknowledgement of a COMMIT the database did run. chriek can't
    tell the last two apart. Other threads' sessions (the worker's) are left alone.
    """
    original = JobStore.create
    armed = threading.local()
    fired: list[str] = []

    def fault(*_args) -> None:
        if getattr(armed, "on", False):
            armed.on = False
            fired.append(event_name)
            raise error or OperationalError("COMMIT", {}, ConnectionError("connection lost"))

    def create(self: JobStore, *args, **kwargs):
        armed.on = True
        try:
            return original(self, *args, **kwargs)
        finally:
            armed.on = False

    with monkeypatch.context() as patch:
        patch.setattr(JobStore, "create", create)
        event.listen(Session, event_name, fault)
        try:
            yield fired
        finally:
            event.remove(Session, event_name, fault)


def _reference_sizes(storage: Path, job_id: str, count: int) -> list[tuple[int, int]]:
    sizes = []
    for index in range(count):
        with Image.open(reference_path(storage, job_id, index)) as stored:
            stored.load()
            sizes.append(stored.size)
    return sizes


@pytest.mark.parametrize("error", [None, _Interrupted()], ids=["lost-connection", "non-exception"])
def test_lost_commit_acknowledgement_keeps_the_committed_job_and_its_references(
    tmp_path, monkeypatch, error
):
    storage = Path(base_env(tmp_path)["CHRIEK_STORAGE_DIR"])
    form = FORM_16 | {"prompt": "lost ack", "seed": "5"}
    files = [
        ("references", ("a.png", make_image_bytes((15, 85, 195), (16, 16)), "image/png")),
        ("references", ("b.png", make_image_bytes((195, 85, 15), (24, 16)), "image/png")),
    ]
    with monkeypatch.context() as patch:
        # The job stays queued until the restart below runs it.
        patch.setattr(JobStore, "claim_next", lambda _self, now: None)
        app = create_app(Settings.from_env(base_env(tmp_path)))
        with (
            _fault_in_create(patch, "after_commit", error) as fired,
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = post_multipart(client, data=form, files=files)
    assert fired == ["after_commit"]
    assert response.status_code == 500
    assert "location" not in response.headers

    # The database committed: the job and its ordered association exist, so must the files.
    (job_dir,) = storage.iterdir()
    job_id = job_dir.name
    with connect_db(tmp_path) as db:
        rows = db.execute(
            text('SELECT "index" FROM job_references WHERE job_id = :id ORDER BY "index"'),
            {"id": job_id},
        )
        assert list(rows.scalars()) == [0, 1]
    assert _reference_sizes(storage, job_id, 2) == [(16, 16), (24, 16)]

    with running(tmp_path) as client:
        body = wait_for(client, job_id)
        assert (body["status"], body["params"]["reference_count"]) == ("succeeded", 2)
        control = wait_for(client, submit_multipart(client, data=form, files=files)["id"])
        assert download_pixels(client, body) == download_pixels(client, control)


def test_failed_commit_call_keeps_references_even_when_nothing_was_committed(tmp_path, monkeypatch):
    storage = Path(base_env(tmp_path)["CHRIEK_STORAGE_DIR"])
    backend = CountingBackend()
    app = create_app(Settings.from_env(base_env(tmp_path)), backend=backend)
    with (
        _fault_in_create(monkeypatch, "before_commit") as fired,
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        response = post_multipart(
            client, data=FORM_16, files=[_reference_file("a.png", (200, 0, 0))]
        )
        assert fired == ["before_commit"]
        assert response.status_code == 500
        # Unknown outcome: no job row, but the upload stays as an orphan rather than risk
        # deleting the files of a job the database might have committed.
        assert stored_owners(tmp_path) == []
        (job_dir,) = storage.iterdir()
        assert _reference_sizes(storage, job_dir.name, 1) == [(16, 16)]
        assert backend.renders == 0


def test_failure_building_the_returned_job_rolls_back_rows_and_files(tmp_path, monkeypatch):
    storage = Path(base_env(tmp_path)["CHRIEK_STORAGE_DIR"])
    built: list[str] = []

    def init(self, **kwargs):  # create() builds the only Job here; ORM loads skip __init__
        built.append(kwargs["id"])
        raise MemoryError

    with monkeypatch.context() as patch:
        patch.setattr(JobStore, "claim_next", lambda _self, now: None)  # keep rows inspectable
        patch.setattr(chriek.store.Job, "__init__", init)
        app = create_app(Settings.from_env(base_env(tmp_path)))
        with TestClient(app, raise_server_exceptions=False) as client:
            response = post_multipart(
                client,
                data=FORM_16,
                files=[
                    _reference_file("a.png", (200, 0, 0)),
                    _reference_file("b.png", (0, 0, 200)),
                ],
            )
    assert len(built) == 1
    assert response.status_code == 500
    assert "location" not in response.headers
    # Rows and files agree: a definite rollback leaves neither. Building the Job after
    # COMMIT left a queued job and its reference rows while the API deleted the files.
    assert stored_owners(tmp_path) == []
    with connect_db(tmp_path) as db:
        assert db.scalar(text("SELECT count(*) FROM job_references")) == 0
    assert not any(storage.iterdir())


def test_filename_less_reference_parts_upload_like_named_ones(tmp_path):
    storage = Path(base_env(tmp_path)["CHRIEK_STORAGE_DIR"])
    red = make_image_bytes((220, 20, 20), (16, 16))
    blue = make_image_bytes((20, 40, 220), (24, 16))
    form = FORM_16 | {"seed": "7"}
    with running(tmp_path) as client:
        named, unnamed = (
            wait_for(
                client,
                submit_multipart(
                    client, data=form, files=[("references", (name, red, "image/png"))]
                )["id"],
            )
            for name in ("red.png", None)
        )
        for body in (named, unnamed):
            assert (body["status"], body["params"]["reference_count"]) == ("succeeded", 1)
        assert download_pixels(client, named) == download_pixels(client, unnamed)

        mixed = submit_multipart(
            client,
            data=form,
            files=[
                ("references", (None, red, "image/png")),
                ("references", ("blue.png", blue, "image/png")),
                ("references", (None, blue, "application/octet-stream")),
                ("references", ("red.png", red, "image/png")),
            ],
        )
        body = wait_for(client, mixed["id"])
        assert (body["status"], body["params"]["reference_count"]) == ("succeeded", 4)
        assert _reference_sizes(storage, mixed["id"], 4) == [(16, 16), (24, 16), (24, 16), (16, 16)]


def test_invalid_filename_less_references_reject_the_whole_request(tmp_path):
    backend = CountingBackend()
    first = ("references", ("first.png", make_image_bytes(), "image/png"))
    with running(tmp_path, backend=backend) as client:
        too_many = post_multipart(
            client,
            data=FORM_16,
            files=[
                ("references", (None if i % 2 else f"{i}.png", make_image_bytes(), "image/png"))
                for i in range(5)
            ],
        )
        assert too_many.status_code == 422, too_many.text
        assert too_many.json()["detail"][0]["msg"] == "at most 4 reference images are allowed"

        for second in (b"https://example.invalid/ref.png", b"not-an-image", b""):
            response = post_multipart(
                client, data=FORM_16, files=[first, ("references", (None, second, "image/png"))]
            )
            assert response.status_code == 422, response.text
            assert [err["loc"] for err in response.json()["detail"]] == [["body", "references", 1]]
        _assert_nothing_persisted(tmp_path, backend)
