"""The installed `chriek serve` command: safe startup errors and a live HTTP round trip."""

import io
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
from conftest import ALICE, ALICE_KEY, BOB_KEY, SIGNING_SECRET, base_env, submit, wait_for
from PIL import Image

CHRIEK = Path(sys.executable).parent / "chriek"


def serve(env: dict[str, str], **kwargs) -> subprocess.Popen:
    return subprocess.Popen(
        [str(CHRIEK), "serve"],
        env={"PATH": os.environ["PATH"]} | env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs,
    )


EMOJI_KEY = "\N{OTTER}" * 32
LATIN1_KEY = "\N{LATIN SMALL LETTER E WITH ACUTE}" * 32
CONTROL_KEY = ALICE_KEY + "\x07"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"CHRIEK_API_KEYS": ""}, "CHRIEK_API_KEYS must name at least one key"),
        ({"CHRIEK_API_KEYS": "alice:short-secret-key"}, "at least 32 printable ASCII"),
        ({"CHRIEK_API_KEYS": f"alice:{EMOJI_KEY}"}, "at least 32 printable ASCII"),
        ({"CHRIEK_API_KEYS": f"alice:{LATIN1_KEY}"}, "at least 32 printable ASCII"),
        ({"CHRIEK_API_KEYS": f"alice:{CONTROL_KEY}"}, "at least 32 printable ASCII"),
        ({"CHRIEK_API_KEYS": ALICE_KEY}, "must be name:key"),
        ({"CHRIEK_API_KEYS": f"bad name:{ALICE_KEY}"}, "must be name:key"),
        (
            {"CHRIEK_API_KEYS": f"alice:{ALICE_KEY},alice:{ALICE_KEY}z"},
            "repeats an earlier name",
        ),
        # Names may equal secrets; errors must not echo them. base_env signs with SIGNING_SECRET.
        (
            {"CHRIEK_API_KEYS": f"{SIGNING_SECRET}:{SIGNING_SECRET},{SIGNING_SECRET}:{ALICE_KEY}"},
            "repeats an earlier name",
        ),
        ({"CHRIEK_API_KEYS": f"{BOB_KEY}:short-secret-key"}, "at least 32 printable ASCII"),
        (
            {"CHRIEK_API_KEYS": f"alice:{ALICE_KEY},{BOB_KEY}:{ALICE_KEY}"},
            "shared with another name",
        ),
        (
            {"CHRIEK_API_KEYS": f"alice:{ALICE_KEY},bob:{ALICE_KEY}"},
            "shared with another name",
        ),
        ({"CHRIEK_SIGNING_SECRET": ""}, "CHRIEK_SIGNING_SECRET must be at least 32"),
        ({"CHRIEK_SIGNING_SECRET": "short-signing-secret"}, "CHRIEK_SIGNING_SECRET"),
        (
            {"CHRIEK_SIGNING_SECRET": ALICE_KEY},
            "CHRIEK_SIGNING_SECRET must differ from every CHRIEK_API_KEYS",
        ),
        (
            {"CHRIEK_SIGNING_SECRET": BOB_KEY},
            "CHRIEK_SIGNING_SECRET must differ from every CHRIEK_API_KEYS",
        ),
        (
            {"CHRIEK_API_KEYS": f"{SIGNING_SECRET}:{ALICE_KEY}"},
            "CHRIEK_SIGNING_SECRET must differ from every CHRIEK_API_KEYS",
        ),
        (
            {"CHRIEK_API_KEYS": f"alice:{ALICE_KEY},{SIGNING_SECRET}:{BOB_KEY}"},
            "CHRIEK_SIGNING_SECRET must differ from every CHRIEK_API_KEYS",
        ),
        ({"CHRIEK_URL_TTL_SECONDS": "0"}, "CHRIEK_URL_TTL_SECONDS must be between"),
        ({"CHRIEK_URL_TTL_SECONDS": "soon"}, "CHRIEK_URL_TTL_SECONDS must be an integer"),
        ({"CHRIEK_MAX_DIMENSION": "1000"}, "CHRIEK_MAX_DIMENSION must be a multiple of 16"),
        ({"CHRIEK_MAX_IMAGES": "0"}, "CHRIEK_MAX_IMAGES must be between"),
        ({"CHRIEK_PORT": "70000"}, "CHRIEK_PORT must be between"),
    ],
)
def test_invalid_configuration_exits_2_without_leaking_secrets(tmp_path, overrides, message):
    env = base_env(tmp_path) | overrides
    process = serve(env)
    output, _ = process.communicate(timeout=30)

    assert process.returncode == 2
    assert output.startswith("chriek: configuration error: ")
    assert message in output
    for secret in (ALICE_KEY, BOB_KEY, SIGNING_SECRET, "short-secret-key", "short-signing-secret"):
        assert secret not in output
    for secret in (EMOJI_KEY, LATIN1_KEY):
        assert secret[:2] not in output
    assert not (tmp_path / "db").exists(), "nothing may start on invalid configuration"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_healthy(process: subprocess.Popen, base: str) -> None:
    deadline = time.monotonic() + 30
    while True:
        try:
            if httpx2.get(f"{base}/health").json() == {"status": "ok"}:
                return
        except httpx2.TransportError:
            pass
        assert process.poll() is None, process.stdout.read()
        assert time.monotonic() < deadline, "chriek serve never became healthy"
        time.sleep(0.1)


@contextmanager
def served_process(
    env: dict[str, str], **kwargs
) -> Iterator[tuple[subprocess.Popen, httpx2.Client]]:
    port = free_port()
    process = serve(env | {"CHRIEK_PORT": str(port)}, **kwargs)
    base = f"http://127.0.0.1:{port}"
    try:
        wait_healthy(process, base)
        with httpx2.Client(base_url=base) as client:
            yield process, client
    finally:
        process.terminate()
        process.wait(timeout=30)


@contextmanager
def served(env: dict[str, str], **kwargs) -> Iterator[httpx2.Client]:
    with served_process(env, **kwargs) as (_process, client):
        yield client


# A non-ASCII signing secret stays valid: only API keys travel in headers.
@pytest.mark.parametrize("signing_secret", [SIGNING_SECRET, EMOJI_KEY])
def test_chriek_serve_round_trip(tmp_path, signing_secret):
    with served(base_env(tmp_path) | {"CHRIEK_SIGNING_SECRET": signing_secret}) as client:
        body = wait_for(client, submit(client, prompt="otter", image_count=2, seed=1)["id"])

        assert body["status"] == "succeeded", body
        for image in body["images"]:
            assert image["url"].startswith(f"{client.base_url}/images/{body['id']}/")
            png = Image.open(io.BytesIO(client.get(image["url"]).content))
            assert png.size == (1024, 1024)
            assert png.text["Seed"] == str(image["seed"])
        assert (tmp_path / "db" / "chriek.db").is_file()
        assert sorted(p.name for p in (tmp_path / "images" / body["id"]).iterdir()) == [
            "0.png",
            "1.png",
        ]


def test_chriek_serve_memory_database_runs_jobs_and_resets_on_restart(tmp_path):
    env = base_env(tmp_path) | {"CHRIEK_DB_PATH": ":memory:"}
    with served(env, cwd=tmp_path) as client:
        body = wait_for(client, submit(client, seed=3, width=16, height=16)["id"])
        assert body["status"] == "succeeded", body
        download = client.get(body["images"][0]["url"])
        assert download.status_code == 200
        assert Image.open(io.BytesIO(download.content)).text["Seed"] == "3"

    with served(env, cwd=tmp_path) as client:
        assert client.get(f"/jobs/{body['id']}", headers=ALICE).status_code == 404

    assert sorted(path.name for path in tmp_path.iterdir()) == ["images"]


def test_chriek_serve_output_never_contains_download_signatures(tmp_path):
    with served_process(base_env(tmp_path)) as (process, client):
        url = wait_for(client, submit(client, width=16, height=16)["id"])["images"][0]["url"]
        assert client.get(url).status_code == 200
    output = process.stdout.read()

    query = parse_qs(urlsplit(url).query)
    assert "Uvicorn running on" in output, output
    assert query["sig"][0] not in output
    assert f"expires={query['expires'][0]}" not in output
