"""The installed `chriek serve` command: safe startup errors and a live HTTP round trip."""

import io
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx2
import pytest
from conftest import ALICE, ALICE_KEY, SIGNING_SECRET, base_env
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"CHRIEK_API_KEYS": ""}, "CHRIEK_API_KEYS must name at least one key"),
        ({"CHRIEK_API_KEYS": "alice:short-secret-key"}, "at least 32 characters"),
        ({"CHRIEK_API_KEYS": ALICE_KEY}, "must be name:key"),
        ({"CHRIEK_API_KEYS": f"bad name:{ALICE_KEY}"}, "must be name:key"),
        (
            {"CHRIEK_API_KEYS": f"alice:{ALICE_KEY},alice:{ALICE_KEY}z"},
            "names 'alice' more than once",
        ),
        (
            {"CHRIEK_API_KEYS": f"alice:{ALICE_KEY},bob:{ALICE_KEY}"},
            "shared with another name",
        ),
        ({"CHRIEK_SIGNING_SECRET": ""}, "CHRIEK_SIGNING_SECRET must be at least 32"),
        ({"CHRIEK_SIGNING_SECRET": "short-signing-secret"}, "CHRIEK_SIGNING_SECRET"),
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
    for secret in (ALICE_KEY, SIGNING_SECRET, "short-secret-key", "short-signing-secret"):
        assert secret not in output
    assert not (tmp_path / "db").exists(), "nothing may start on invalid configuration"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_chriek_serve_round_trip(tmp_path):
    port = free_port()
    env = base_env(tmp_path) | {"CHRIEK_PORT": str(port)}
    process = serve(env)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                if httpx2.get(f"{base}/health").json() == {"status": "ok"}:
                    break
            except httpx2.TransportError:
                pass
            assert process.poll() is None, process.stdout.read()
            assert time.monotonic() < deadline, "chriek serve never became healthy"
            time.sleep(0.1)

        job = httpx2.post(
            f"{base}/jobs", headers=ALICE, json={"prompt": "otter", "image_count": 2, "seed": 1}
        ).json()
        while (body := httpx2.get(f"{base}/jobs/{job['id']}", headers=ALICE).json())[
            "status"
        ] not in {"succeeded", "failed"}:
            time.sleep(0.05)

        assert body["status"] == "succeeded", body
        for image in body["images"]:
            assert image["url"].startswith(f"{base}/images/{job['id']}/")
            png = Image.open(io.BytesIO(httpx2.get(image["url"]).content))
            assert png.size == (1024, 1024)
            assert png.text["Seed"] == str(image["seed"])
        assert (tmp_path / "db" / "chriek.db").is_file()
        assert sorted(p.name for p in (tmp_path / "images" / job["id"]).iterdir()) == [
            "0.png",
            "1.png",
        ]
    finally:
        process.terminate()
        process.wait(timeout=30)
