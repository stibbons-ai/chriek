"""The installed `chriek serve` command: safe startup errors, live HTTP and HTTPS round trips,
and operator walkthroughs for api-key mode, proxy mode and native TLS."""

import io
import os
import shutil
import socket
import ssl
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
from conftest import (
    ALICE,
    ALICE_KEY,
    BOB,
    SIGNING_SECRET,
    SMALL,
    base_env,
    proxy_env,
    submit,
    wait_for,
    with_query,
)
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


def exit_output(process: subprocess.Popen) -> str:
    """Wait for `chriek serve` to exit; kill, reap and fail if it keeps running."""
    try:
        return process.communicate(timeout=30)[0]
    except subprocess.TimeoutExpired:
        process.kill()
        output = process.communicate()[0]
        pytest.fail(f"chriek serve kept running instead of exiting:\n{output}")


EMOJI_KEY = "\N{OTTER}" * 32
LATIN1_KEY = "\N{LATIN SMALL LETTER E WITH ACUTE}" * 32
CONTROL_KEY = ALICE_KEY + "\x07"


def assert_config_error(process: subprocess.Popen, tmp_path: Path, message: str) -> None:
    """Startup stopped with exit code 2, named the problem, leaked no secret, opened nothing."""
    output = exit_output(process)

    assert process.returncode == 2
    assert output.startswith("chriek: configuration error: ")
    assert message in output
    for secret in (ALICE_KEY, SIGNING_SECRET, "short-secret-key", "short-signing-secret"):
        assert secret not in output
    for secret in (EMOJI_KEY, LATIN1_KEY):
        assert secret[:2] not in output
    assert "PRIVATE KEY" not in output
    assert not (tmp_path / "db").exists(), "nothing may start on invalid configuration"


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
        ({"CHRIEK_AUTH_MODE": "Proxy"}, "CHRIEK_AUTH_MODE must be 'api-key' or 'proxy'"),
        ({"CHRIEK_AUTH_MODE": ""}, "CHRIEK_AUTH_MODE must be 'api-key' or 'proxy'"),
        ({"CHRIEK_AUTH_MODE": "proxy"}, "CHRIEK_API_KEYS"),
        (
            {"CHRIEK_AUTH_MODE": "proxy", "CHRIEK_API_KEYS": "", "CHRIEK_SIGNING_SECRET": ""},
            "CHRIEK_SIGNING_SECRET must be at least 32",
        ),
        ({"CHRIEK_MAX_QUEUED_JOBS": "0"}, "CHRIEK_MAX_QUEUED_JOBS must be between 1 and 100000"),
        ({"CHRIEK_MAX_QUEUED_JOBS": "100001"}, "CHRIEK_MAX_QUEUED_JOBS must be between"),
        ({"CHRIEK_MAX_QUEUED_JOBS": "lots"}, "CHRIEK_MAX_QUEUED_JOBS must be an integer"),
        ({"CHRIEK_TLS_CERTFILE": "cert.pem"}, "must be set together"),
        ({"CHRIEK_TLS_KEYFILE": "key.pem"}, "must be set together"),
        (
            {"CHRIEK_TLS_CERTFILE": "", "CHRIEK_TLS_KEYFILE": ""},
            "CHRIEK_TLS_CERTFILE must not be empty",
        ),
        (
            {"CHRIEK_TLS_CERTFILE": "cert.pem", "CHRIEK_TLS_KEYFILE": ""},
            "CHRIEK_TLS_KEYFILE must not be empty",
        ),
    ],
)
def test_invalid_configuration_exits_2_without_leaking_secrets(tmp_path, overrides, message):
    assert_config_error(serve(base_env(tmp_path) | overrides), tmp_path, message)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_healthy(process: subprocess.Popen, base: str, verify: ssl.SSLContext | bool) -> None:
    deadline = time.monotonic() + 30
    while True:
        try:
            if httpx2.get(f"{base}/health", verify=verify).json() == {"status": "ok"}:
                return
        except httpx2.TransportError:
            pass
        assert process.poll() is None, process.stdout.read()
        assert time.monotonic() < deadline, "chriek serve never became healthy"
        time.sleep(0.1)


@contextmanager
def served_process(
    env: dict[str, str], *, ca: Path | None = None, **kwargs
) -> Iterator[tuple[subprocess.Popen, httpx2.Client]]:
    """Run `chriek serve`; with `ca`, talk HTTPS and trust only that certificate."""
    port = free_port()
    process = serve(env | {"CHRIEK_PORT": str(port)}, **kwargs)
    base = f"{'https' if ca else 'http'}://127.0.0.1:{port}"
    verify = ssl.create_default_context(cafile=ca) if ca else True
    try:
        wait_healthy(process, base, verify)
        with httpx2.Client(base_url=base, verify=verify) as client:
            yield process, client
    finally:
        process.terminate()
        process.wait(timeout=30)


@contextmanager
def served(env: dict[str, str], **kwargs) -> Iterator[httpx2.Client]:
    with served_process(env, **kwargs) as (_process, client):
        yield client


def test_chriek_serve_round_trip(tmp_path):
    with served(base_env(tmp_path)) as client:
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


def openssl(*args: str) -> None:
    command = shutil.which("openssl")
    assert command, "the TLS tests need the openssl command"
    subprocess.run([command, *args], check=True, capture_output=True)


def make_tls_files(directory: Path, name: str = "server") -> tuple[Path, Path]:
    """Self-signed certificate for 127.0.0.1 and its unencrypted key."""
    directory.mkdir(parents=True, exist_ok=True)
    cert, key = directory / f"{name}-cert.pem", directory / f"{name}-key.pem"
    openssl(
        "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes",
        "-days", "1", "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
        "-keyout", str(key), "-out", str(cert),
    )  # fmt: skip
    return cert, key


def tls_env(tmp_path: Path, cert: Path, key: Path) -> dict[str, str]:
    return base_env(tmp_path) | {"CHRIEK_TLS_CERTFILE": str(cert), "CHRIEK_TLS_KEYFILE": str(key)}


UNUSABLE_TLS = "must name a readable PEM certificate and its unencrypted private key"


def bad_tls_files(tmp_path: Path, kind: str) -> tuple[Path, Path]:
    cert, key = make_tls_files(tmp_path / "tls")
    if kind == "missing-cert":
        return tmp_path / "tls" / "absent.pem", key
    if kind == "mismatched-key":
        return cert, make_tls_files(tmp_path / "tls", name="other")[1]
    if kind == "not-pem":
        (garbage := tmp_path / "tls" / "garbage.pem").write_text("not a certificate\n")
        return garbage, key
    if kind == "encrypted-key":
        encrypted = tmp_path / "tls" / "encrypted-key.pem"
        openssl("pkey", "-in", str(key), "-out", str(encrypted), "-aes256", "-passout", "pass:pw")
        return cert, encrypted
    raise AssertionError(kind)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing-cert", UNUSABLE_TLS),
        ("mismatched-key", UNUSABLE_TLS),
        ("not-pem", UNUSABLE_TLS),
        ("encrypted-key", "CHRIEK_TLS_KEYFILE must be an unencrypted private key"),
    ],
)
def test_unusable_tls_files_stop_startup_instead_of_serving_plain_http(tmp_path, kind, message):
    cert, key = bad_tls_files(tmp_path, kind)
    process = serve(tls_env(tmp_path, cert, key), stdin=subprocess.DEVNULL)
    assert_config_error(process, tmp_path, message)


def test_walkthrough_api_key_operator_keeps_owners_apart(tmp_path):
    with served(base_env(tmp_path)) as client:
        alice_job = submit(client, headers=ALICE, prompt="alice", **SMALL)
        bob_job = submit(client, headers=BOB, prompt="bob", **SMALL)
        alice = wait_for(client, alice_job["id"], headers=ALICE)
        assert wait_for(client, bob_job["id"], headers=BOB)["status"] == "succeeded"

        assert client.get(f"/jobs/{alice['id']}", headers=BOB).status_code == 404
        assert client.get(f"/jobs/{bob_job['id']}", headers=ALICE).status_code == 404
        assert client.get(f"/jobs/{alice['id']}").status_code == 401
        assert client.post("/jobs", content=b"{").status_code == 401
        assert client.get(alice["images"][0]["url"]).status_code == 200


def test_walkthrough_proxy_operator_without_api_keys_shares_one_owner(tmp_path):
    with served(proxy_env(tmp_path)) as client:
        anonymous = submit(client, headers={}, prompt="anonymous", **SMALL)
        claimed = submit(client, headers={"X-Forwarded-User": "bob"}, prompt="claimed", **SMALL)

        for job in (anonymous, claimed):
            for headers in ({}, ALICE, {"X-Forwarded-User": "someone-else"}):
                assert wait_for(client, job["id"], headers=headers)["status"] == "succeeded"
        url = wait_for(client, claimed["id"])["images"][0]["url"]
        assert client.get(url).status_code == 200
        assert client.get(with_query(url, sig="0" * 64)).status_code == 403


def test_walkthrough_native_tls_operator_downloads_over_https_only(tmp_path):
    cert, key = make_tls_files(tmp_path / "tls")
    with served_process(tls_env(tmp_path, cert, key), ca=cert) as (process, client):
        body = wait_for(client, submit(client, seed=5, width=32, height=16)["id"])
        assert body["status"] == "succeeded", body
        url = body["images"][0]["url"]
        assert url.startswith(f"https://127.0.0.1:{client.base_url.port}/images/")
        download = client.get(url)
        assert download.status_code == 200
        assert Image.open(io.BytesIO(download.content)).text["Seed"] == "5"
        assert client.get(with_query(url, sig="0" * 64)).status_code == 403

        plain = f"http://127.0.0.1:{client.base_url.port}/health"
        with pytest.raises(httpx2.TransportError):
            httpx2.get(plain)
    output = process.stdout.read()
    assert f"Uvicorn running on https://127.0.0.1:{client.base_url.port}" in output, output
