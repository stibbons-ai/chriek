"""Job submission, polling, validation (property 5) and whole-job failure (property 4)."""

import uuid

import pytest
from conftest import ALICE, CountingBackend, FailingBackend, running, submit, wait_for

MAX_SEED = 2**32 - 1


def test_submit_returns_202_with_uuid7_and_effective_parameters(client):
    response = client.post(
        "/jobs",
        headers=ALICE,
        json={"prompt": "a lighthouse", "image_count": 2, "seed": 41, "width": 512, "height": 256},
    )

    assert response.status_code == 202
    body = response.json()
    assert uuid.UUID(body["id"]).version == 7
    assert response.headers["location"] == f"/jobs/{body['id']}"
    assert body["status"] == "queued"
    assert body["params"] == {
        "prompt": "a lighthouse",
        "image_count": 2,
        "seed": 41,
        "width": 512,
        "height": 256,
    }
    assert body["images"] == []
    assert body["error"] is None


def test_defaults_are_one_1024_square_image(client):
    params = submit(client)["params"]
    assert (params["image_count"], params["width"], params["height"]) == (1, 1024, 1024)


def test_succeeded_job_reports_run_record_and_seed_per_image(client):
    job = submit(client, image_count=3, seed=100, width=64, height=32)

    body = wait_for(client, job["id"])

    assert body["status"] == "succeeded"
    assert body["error"] is None
    assert [(image["index"], image["seed"]) for image in body["images"]] == [
        (0, 100),
        (1, 101),
        (2, 102),
    ]
    assert all(image["url"] and image["expires_at"] for image in body["images"])
    assert body["run"]["backend"] == "fake"
    assert body["run"]["model"] == "chriek-fake-backend"
    assert body["run"]["revision"] == "fake-v1"
    assert body["run"]["render_seconds"] >= 0
    assert body["created_at"] <= body["started_at"] <= body["finished_at"]


def test_omitted_seed_is_chosen_and_reported(client):
    seeds = set()
    for _ in range(3):
        body = wait_for(client, submit(client, image_count=2, width=16, height=16)["id"])
        seed = body["params"]["seed"]
        assert isinstance(seed, int)
        assert 0 <= seed <= MAX_SEED - 1
        assert [image["seed"] for image in body["images"]] == [seed, seed + 1]
        seeds.add(seed)
    assert len(seeds) > 1


def test_highest_valid_seed_is_accepted(client):
    body = wait_for(
        client, submit(client, image_count=2, seed=MAX_SEED - 1, width=16, height=16)["id"]
    )
    assert [image["seed"] for image in body["images"]] == [MAX_SEED - 1, MAX_SEED]


INVALID_REQUESTS = {
    "missing-prompt": {"prompt": None},
    "blank-prompt": {"prompt": "   "},
    "prompt-too-long": {"prompt": "x" * 2001},
    "prompt-not-string": {"prompt": 7},
    "width-not-multiple-of-16": {"width": 1000},
    "height-not-multiple-of-16": {"height": 17},
    "zero-width": {"width": 0},
    "negative-height": {"height": -16},
    "width-above-max-dimension": {"width": 2064, "height": 16},
    "too-many-pixels": {"width": 2048, "height": 1024},
    "zero-images": {"image_count": 0},
    "too-many-images": {"image_count": 5},
    "negative-seed": {"seed": -1},
    "seed-above-max": {"seed": MAX_SEED + 1},
    "seed-overflows-with-count": {"seed": MAX_SEED, "image_count": 2},
    "string-width": {"width": "1024"},
    "float-width": {"width": 1024.0},
    "bool-count": {"image_count": True},
    "unknown-field": {"steps": 50},
}


@pytest.mark.parametrize("overrides", INVALID_REQUESTS.values(), ids=INVALID_REQUESTS.keys())
def test_invalid_requests_are_422_and_never_execute(tmp_path, overrides):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        body = {"prompt": "a lighthouse", "width": 16, "height": 16} | overrides
        body = {key: value for key, value in body.items() if value is not None}

        response = client.post("/jobs", headers=ALICE, json=body)

        assert response.status_code == 422, response.text
        assert response.json()["detail"]
        # A valid job submitted afterwards is the first and only one the worker renders.
        valid = submit(client, width=16, height=16)
        assert wait_for(client, valid["id"])["status"] == "succeeded"
        assert backend.renders == 1


NUMERIC_FIELDS = ("image_count", "seed", "width", "height")
NONFINITE = ("NaN", "Infinity", "-Infinity", "1e999", "-1e999")
RAW_INVALID_BODIES = {
    **{
        f"{field}={constant}": (
            f'{{"prompt": "a lighthouse", "width": 16, "height": 16, "{field}": {constant}}}',
            ["body", field],
        )
        for field in NUMERIC_FIELDS
        for constant in NONFINITE
    },
    "finite-float-control": ('{"prompt": "a lighthouse", "width": 16.5}', ["body", "width"]),
    "nonfinite-unknown-field": ('{"prompt": "a lighthouse", "steps": NaN}', ["body", "steps"]),
    "lone-high-surrogate": (r'{"prompt": "\ud800"}', ["body", "prompt"]),
    "lone-low-surrogate": (r'{"prompt": "\udc00"}', ["body", "prompt"]),
    "embedded-lone-surrogate": (r'{"prompt": "a light\ud83dhouse"}', ["body", "prompt"]),
    "reversed-surrogate-pair": (r'{"prompt": "\ude00\ud83d"}', ["body", "prompt"]),
    # Pydantic rejects the whole body when an object key holds a lone surrogate.
    "surrogate-in-unknown-key": (r'{"prompt": "a lighthouse", "\ud800": 1}', ["body"]),
    "surrogate-key-with-infinity": (r'{"\udfff": -Infinity}', ["body"]),
}


@pytest.mark.parametrize(
    ("content", "loc"), RAW_INVALID_BODIES.values(), ids=RAW_INVALID_BODIES.keys()
)
def test_unencodable_input_is_422_and_never_executes(tmp_path, content, loc):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        response = client.post(
            "/jobs", headers=ALICE | {"Content-Type": "application/json"}, content=content
        )

        assert response.status_code == 422, response.text
        assert loc in [error["loc"] for error in response.json()["detail"]]
        valid = submit(client, width=16, height=16)
        assert wait_for(client, valid["id"])["status"] == "succeeded"
        assert backend.renders == 1


@pytest.mark.parametrize(
    ("prompt_json", "prompt"),
    [
        ('"Leuchtturm im Nebel, 灯台 ☀"', "Leuchtturm im Nebel, 灯台 ☀"),
        (r'"a lighthouse \ud83d\ude00"', "a lighthouse \N{GRINNING FACE}"),
    ],
    ids=["literal-unicode", "escaped-surrogate-pair"],
)
def test_valid_unicode_prompt_is_persisted_and_rendered(client, prompt_json, prompt):
    response = client.post(
        "/jobs",
        headers=ALICE | {"Content-Type": "application/json"},
        content=f'{{"prompt": {prompt_json}, "width": 16, "height": 16}}'.encode(),
    )

    assert response.status_code == 202, response.text
    body = wait_for(client, response.json()["id"])
    assert body["status"] == "succeeded"
    assert body["params"]["prompt"] == prompt
    assert len(body["images"]) == 1


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"not json", 422),
        # Python's JSON parser refuses integers this long; FastAPI answers 400.
        (b'{"prompt": "x", "seed": 1' + b"0" * 5000 + b"}", 400),
        # JSON bodies must be valid UTF-8; FastAPI answers 400 before validation.
        (b'{"prompt": "a light\xffhouse"}', 400),
    ],
    ids=["not-json", "5000-digit-seed", "invalid-utf8"],
)
def test_unparseable_body_is_rejected(client, content, expected):
    response = client.post(
        "/jobs", headers=ALICE | {"Content-Type": "application/json"}, content=content
    )
    assert response.status_code == expected


@pytest.mark.parametrize(
    ("content_type", "content"),
    [
        ("text/plain", b"\xff"),
        ("application/octet-stream", b"\xff"),
        ("application/octet-stream", b'{"prompt": "a light\xc3house"}'),
        ("text/plain", b"hello"),
    ],
    ids=["text-ff", "octet-ff", "octet-truncated-utf8", "text-ascii"],
)
def test_undecodable_body_bytes_are_422_and_never_execute(tmp_path, content_type, content):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        response = client.post(
            "/jobs", headers=ALICE | {"Content-Type": content_type}, content=content
        )

        assert response.status_code == 422, response.text
        assert isinstance(response.json()["detail"], list)
        valid = submit(client, width=16, height=16)
        assert wait_for(client, valid["id"])["status"] == "succeeded"
        assert backend.renders == 1


def test_limits_come_from_configuration(tmp_path):
    with running(
        tmp_path,
        CHRIEK_MAX_IMAGES="2",
        CHRIEK_MAX_DIMENSION="64",
        CHRIEK_MAX_PIXELS="2048",
        CHRIEK_MAX_PROMPT_CHARS="5",
    ) as client:
        accepted = {"prompt": "short", "image_count": 2, "width": 64, "height": 32}
        assert client.post("/jobs", headers=ALICE, json=accepted).status_code == 202
        for rejected in (
            {"image_count": 3},
            {"width": 80, "height": 16},
            {"width": 64, "height": 48},
            {"prompt": "longer"},
        ):
            response = client.post("/jobs", headers=ALICE, json=accepted | rejected)
            assert response.status_code == 422, rejected


def test_any_image_failure_fails_the_whole_job_without_images(tmp_path):
    backend = FailingBackend(fail_index=1)
    with running(tmp_path, backend=backend) as client:
        job = submit(client, image_count=3, seed=5, width=16, height=16)

        body = wait_for(client, job["id"])

        assert body["status"] == "failed"
        assert body["images"] == []
        assert body["error"] == "image 2 of 3 failed: injected backend failure"
        assert body["run"]["backend"] == "fake"
        assert backend.renders == 2  # rendering stops at the first failure
        assert not any((tmp_path / "images").iterdir()), "partial output must be discarded"


def test_failed_job_does_not_stop_the_queue(tmp_path):
    backend = FailingBackend(fail_index=0)
    with running(tmp_path, backend=backend) as client:
        failed = submit(client, width=16, height=16)
        assert wait_for(client, failed["id"])["status"] == "failed"
        succeeded = submit(client, width=16, height=16)
        assert wait_for(client, succeeded["id"])["status"] == "succeeded"
        assert client.get("/health").json() == {"status": "ok"}
