"""Issue #5: multipart reference images for edit jobs."""

import io
import sqlite3
import uuid
from urllib.parse import parse_qs, urlsplit

import pytest
from conftest import (
    ALICE,
    BOB,
    CountingBackend,
    FailingBackend,
    GatedBackend,
    make_image_bytes,
    running,
    submit,
    wait_for,
)
from PIL import Image


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

    pixels_text = download_pixels(client, text_only)
    pixels_one = download_pixels(client, one_ref)
    pixels_one_again = download_pixels(client, one_ref_again)
    pixels_other = download_pixels(client, other_pixel_ref)
    pixels_rb = download_pixels(client, red_then_blue)
    pixels_br = download_pixels(client, blue_then_red)

    assert pixels_one == pixels_one_again, (
        "same references must render identically regardless of filename"
    )
    assert len({pixels_text, pixels_one, pixels_other, pixels_rb, pixels_br}) == 5


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
        first = submit(before, prompt="blocker", width=16, height=16)
        assert crashed.rendering.wait(5)
        queued = submit_multipart(
            before,
            data={"prompt": "queued-edit", "seed": "55", "width": "16", "height": "16"},
            files=malicious_files,
        )
        job_dir = tmp_path / "images" / queued["id"]
        assert sorted(p.name for p in (job_dir / "references").iterdir()) == ["0.png", "1.png"]
        assert not (job_dir / "0.png").exists()
        assert not (tmp_path / "0.png").exists()

        with running(tmp_path) as after:
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


def test_reference_bytes_cannot_be_downloaded_and_failed_edit_job_publishes_no_outputs(tmp_path):
    ref_bytes = make_image_bytes((200, 100, 50))
    failing = FailingBackend(fail_index=0)

    with running(tmp_path, backend=failing) as client:
        failed_job = submit_multipart(
            client,
            data={"prompt": "fail edit", "image_count": "2", "width": "16", "height": "16"},
            files=[("references", ("0.png", ref_bytes, "image/png"))],
        )
        failed_body = wait_for(client, failed_job["id"])
        assert failed_body["status"] == "failed"
        assert failed_body["images"] == []
        job_dir = tmp_path / "images" / failed_job["id"]
        assert not (job_dir / "0.png").exists()
        assert not (job_dir / "1.png").exists()

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
        expires, sig = query["expires"][0], query["sig"][0]

        for path in (
            f"/images/{ok_job['id']}/references/0.png?expires={expires}&sig={sig}",
            f"/images/{ok_job['id']}/references%2F0.png?expires={expires}&sig={sig}",
            f"/images/{ok_job['id']}/1.png?expires={expires}&sig={sig}",
            f"/images/{failed_job['id']}/0.png?expires={expires}&sig={sig}",
        ):
            assert client.get(path).status_code in (403, 404), path


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

    backend = CountingBackend()
    with running(
        tmp_path,
        backend=backend,
        CHRIEK_MAX_REFERENCES="2",
        CHRIEK_MAX_UPLOAD_BYTES=str(limit_bytes),
    ) as client:
        # Exact byte limit succeeds, even if Content-Length is omitted or misleadingly small.
        ok = client.post(
            "/jobs",
            headers=ALICE | ct_header | {"Content-Length": "10"},
            content=exact_sized_body,
        )
        assert ok.status_code == 202, ok.text
        assert wait_for(client, ok.json()["id"])["status"] == "succeeded"

        # 1 byte over limit fails with 422, even when Content-Length claims 10 bytes.
        _, over_body = encode_multipart(
            {"prompt": padded_prompt + "y", "width": "16", "height": "16"},
            files=ref_files,
            boundary=boundary,
        )
        over = client.post(
            "/jobs",
            headers=ALICE | ct_header | {"Content-Length": "10"},
            content=over_body,
        )
        assert over.status_code == 422
        assert any("maximum size" in e["msg"] for e in over.json()["detail"])

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


def test_streaming_upload_bound_aborts_before_consuming_remaining_parts(tmp_path):
    # Second part is malformed UTF-8 that would trigger a field error if the parser kept reading
    # past the upload byte limit.
    boundary = "bound"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="prompt"\r\n\r\n'
            "hello\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="references"; filename="big.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        + (b"x" * 2500)
        + (
            f"\r\n--{boundary}\r\n"
            'Content-Disposition: form-data; name="width"\r\n\r\n'
            "not-an-int\r\n"
            f"--{boundary}--\r\n"
        ).encode()
    )

    with running(tmp_path, CHRIEK_MAX_UPLOAD_BYTES="2048") as client:
        response = client.post(
            "/jobs",
            headers=ALICE | {"Content-Type": f"multipart/form-data; boundary={boundary}"},
            content=body,
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert len(detail) == 1
        assert detail[0]["loc"] == ["body", "references"]
        assert "maximum size of 2048 bytes" in detail[0]["msg"]


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
    ],
    ids=[
        "missing-boundary",
        "malformed-multipart",
        "truncated-multipart",
        "duplicate-scalar-field",
        "invalid-utf8-scalar",
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
