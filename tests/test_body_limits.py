"""Request bodies are capped at 2 MiB after authentication; 422 responses at 16 KiB."""

import asyncio
import itertools
import json

import pytest
from conftest import ALICE, ALICE_KEY, CountingBackend, running, submit, wait_for

MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_ERROR_BYTES = 16 * 1024
MIB = 1024 * 1024
VALID = b'{"prompt": "a lighthouse", "width": 16, "height": 16}'
JSON_ALICE = {"Authorization": f"Bearer {ALICE_KEY}", "Content-Type": "application/json"}


def padded(size: int) -> bytes:
    """A valid job body of exactly `size` bytes; JSON whitespace pads it."""
    return VALID[:-1] + b" " * (size - len(VALID)) + b"}"


def post_asgi(app, chunks: list[bytes], headers: dict[str, str]) -> tuple[int, int]:
    """POST /jobs straight through ASGI, one receive message per chunk.

    TestClient delivers the body in a single receive message; this harness can stream chunks,
    understate or omit the length, and count how many chunks the app pulled.
    Returns (status, receive calls).
    """
    messages = [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ]
    calls = 0
    statuses = []

    async def receive():
        nonlocal calls
        calls += 1
        return messages[calls - 1]

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/jobs",
        "raw_path": b"/jobs",
        "root_path": "",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    return statuses[0], calls


def assert_nothing_queued(client, backend: CountingBackend) -> None:
    """A valid job submitted afterwards is the first and only one the worker renders."""
    valid = submit(client, width=16, height=16)
    assert wait_for(client, valid["id"])["status"] == "succeeded"
    assert backend.renders == 1


def test_declared_oversize_is_413_without_reading_the_body(tmp_path):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        headers = JSON_ALICE | {"Content-Length": str(MAX_BODY_BYTES + 1)}

        assert post_asgi(client.app, [padded(MAX_BODY_BYTES + 1)], headers) == (413, 0)
        assert_nothing_queued(client, backend)


@pytest.mark.parametrize(
    "length", [None, "10", "not-a-number"], ids=["absent", "understated", "malformed"]
)
def test_streamed_overflow_stops_at_the_first_overflowing_chunk(tmp_path, length):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        headers = JSON_ALICE | ({} if length is None else {"Content-Length": length})
        # 1 MiB + 1 MiB is exactly the cap; the third chunk overflows; two more stay unread.
        chunks = [b" " * MIB] * 5

        assert post_asgi(client.app, chunks, headers) == (413, 3)
        assert_nothing_queued(client, backend)


def test_streamed_valid_job_overflow_is_413_and_never_queued(tmp_path):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        # A valid job followed by whitespace: every prefix up to the cap would be admitted.
        chunks = [VALID] + [b" " * MIB] * 4
        totals = itertools.accumulate(map(len, chunks))
        first_overflow = next(i for i, total in enumerate(totals, 1) if total > MAX_BODY_BYTES)
        assert first_overflow < len(chunks)

        assert post_asgi(client.app, chunks, JSON_ALICE) == (413, first_overflow)
        assert_nothing_queued(client, backend)


def test_exact_limit_is_admitted_and_one_more_byte_is_413(tmp_path):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        over = padded(MAX_BODY_BYTES + 1)
        response = client.post("/jobs", headers=JSON_ALICE, content=over)
        assert response.status_code == 413
        assert response.json() == {"detail": f"request body must be at most {MAX_BODY_BYTES} bytes"}
        # Streamed without Content-Length, split mid-body: received bytes decide.
        assert post_asgi(client.app, [over[:MIB], over[MIB:]], JSON_ALICE)[0] == 413
        assert_nothing_queued(client, backend)

        exact = padded(MAX_BODY_BYTES)
        response = client.post("/jobs", headers=JSON_ALICE, content=exact)
        assert response.status_code == 202, response.text
        assert wait_for(client, response.json()["id"])["status"] == "succeeded"
        assert post_asgi(client.app, [exact[:MIB], exact[MIB:]], JSON_ALICE)[0] == 202


@pytest.mark.parametrize(
    "auth", [{}, {"Authorization": "Bearer invalid"}], ids=["missing", "invalid"]
)
@pytest.mark.parametrize(
    "length", [None, str(MAX_BODY_BYTES + 1)], ids=["no-length", "oversized-length"]
)
def test_missing_or_invalid_key_is_401_before_reading_any_body(tmp_path, auth, length):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        headers = auth | {"Content-Type": "application/json"}
        if length is not None:
            headers["Content-Length"] = length

        assert post_asgi(client.app, [b"{", b" " * MIB], headers) == (401, 0)
        assert_nothing_queued(client, backend)


ADVERSARIAL_422 = {
    "huge-prompt": json.dumps({"prompt": "x" * (MAX_BODY_BYTES - 100)}),
    "huge-wrong-type": json.dumps({"prompt": "a", "width": "9" * (MAX_BODY_BYTES - 100)}),
    "long-unknown-key": json.dumps({"prompt": "a", "k" * (MAX_BODY_BYTES - 100): 1}),
    "control-char-key": json.dumps({"prompt": "a", "\x00" * 300_000: 1}),
    "surrogate-key": '{"prompt": "a", "' + "\\ud800" * 300_000 + '": 1}',
    "surrogate-prompt": '{"prompt": "' + "\\udc00" * 300_000 + '"}',
    "many-unknown-fields": json.dumps({"prompt": "a"} | {f"{i:0>200}": 1 for i in range(9000)}),
    "broad-list": json.dumps({"prompt": "a", "seed": [1.5] * 300_000}),
    "deep-nesting": '{"prompt": ' + '{"a": ' * 100_000 + "0" + "}" * 100_000 + "}",
    "nonfinite-many": "{" + ",".join(f'"f{i}": NaN' for i in range(100_000)) + "}",
    "malformed": "{" + " " * (MAX_BODY_BYTES - 10),
}


ADVERSARIAL_422_CASES = [
    pytest.param(content.encode(), "application/json", id=name)
    for name, content in ADVERSARIAL_422.items()
] + [pytest.param(b"\xff" * MAX_BODY_BYTES, "text/plain", id="invalid-bytes-text-plain")]


@pytest.mark.parametrize(("body", "content_type"), ADVERSARIAL_422_CASES)
def test_422_is_bounded_json_without_input_echo(tmp_path, body, content_type):
    backend = CountingBackend()
    with running(tmp_path, backend=backend) as client:
        assert len(body) <= MAX_BODY_BYTES

        response = client.post(
            "/jobs", headers=ALICE | {"Content-Type": content_type}, content=body
        )

        assert response.status_code == 422, response.text[:500]
        assert len(response.content) <= MAX_ERROR_BYTES
        detail = response.json()["detail"]
        assert detail, "at least one actionable error survives the bound"
        for error in detail:
            assert set(error) == {"type", "loc", "msg"}
        assert_nothing_queued(client, backend)


def test_422_fills_the_budget_exactly_and_drops_the_error_that_would_cross_it(client):
    keys = [f"{i:0>200}" for i in range(57)]

    def unknown_fields(last_key_length: int):
        body = {"prompt": "a"} | dict.fromkeys(keys, 1) | {"z" * last_key_length: 1}
        response = client.post("/jobs", headers=ALICE, json=body)
        assert response.status_code == 422
        return response

    # Calibrate on the live output: each extra ASCII key character adds one response byte.
    short = unknown_fields(1)
    assert len(short.json()["detail"]) == 58
    fitting = MAX_ERROR_BYTES - len(short.content) + 1

    exact = unknown_fields(fitting)
    assert len(exact.content) == MAX_ERROR_BYTES
    assert exact.json()["detail"][-1]["loc"] == ["body", "z" * fitting]

    over = unknown_fields(fitting + 1)
    assert len(over.content) <= MAX_ERROR_BYTES
    assert [error["loc"] for error in over.json()["detail"]] == [["body", key] for key in keys]


def test_ordinary_errors_stay_actionable(client):
    response = client.post(
        "/jobs", headers=ALICE, json={"prompt": "a lighthouse", "width": 1000, "steps": 50}
    )

    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert set(error) == {"type", "loc", "msg"}
    assert error["type"] == "extra_forbidden"
    assert error["loc"] == ["body", "steps"]
    assert error["msg"]
    response = client.post("/jobs", headers=ALICE, json={"prompt": "a lighthouse", "width": 1000})
    assert response.json() == {
        "detail": [
            {
                "type": "value_error",
                "loc": ["body", "width"],
                "msg": "width must be a multiple of 16 between 16 and 2048",
            }
        ]
    }
