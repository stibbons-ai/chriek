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
MARKER = "public-url-secret-marker"
URL_FORM = "CHRIEK_PUBLIC_URL must be an absolute http or https URL"
PRINTABLE = "CHRIEK_PUBLIC_URL must contain only printable ASCII"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"CHRIEK_API_KEYS": ""}, "CHRIEK_API_KEYS must name at least one key"),
        ({"CHRIEK_API_KEYS": "alice:short-secret-key"}, "entry 1 key must be at least 32"),
        ({"CHRIEK_API_KEYS": f"alice:{EMOJI_KEY}"}, "entry 1 key must be at least 32"),
        ({"CHRIEK_API_KEYS": f"alice:{LATIN1_KEY}"}, "entry 1 key must be at least 32"),
        ({"CHRIEK_API_KEYS": f"alice:{CONTROL_KEY}"}, "entry 1 key must be at least 32"),
        (
            {"CHRIEK_API_KEYS": f"alice:{ALICE_KEY},bob:short-secret-key"},
            "entry 2 key must be at least 32",
        ),
        ({"CHRIEK_API_KEYS": ALICE_KEY}, "entry 1 must be name:key"),
        ({"CHRIEK_API_KEYS": f"bad name:{ALICE_KEY}"}, "entry 1 must be name:key"),
        (
            {"CHRIEK_API_KEYS": f"alice:{ALICE_KEY},alice:{ALICE_KEY}z"},
            "entry 2 repeats an earlier name",
        ),
        # Names may equal secrets; errors must not echo them. base_env signs with SIGNING_SECRET.
        (
            {"CHRIEK_API_KEYS": f"{SIGNING_SECRET}:{SIGNING_SECRET},{SIGNING_SECRET}:{ALICE_KEY}"},
            "entry 2 repeats an earlier name",
        ),
        ({"CHRIEK_API_KEYS": f"{BOB_KEY}:short-secret-key"}, "entry 1 key must be at least 32"),
        (
            {"CHRIEK_API_KEYS": f"alice:{ALICE_KEY},{BOB_KEY}:{ALICE_KEY}"},
            "entry 2 key is shared with another name",
        ),
        (
            {"CHRIEK_API_KEYS": f"alice:{ALICE_KEY},bob:{ALICE_KEY}"},
            "entry 2 key is shared with another name",
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
        ({"CHRIEK_PUBLIC_URL": ""}, "CHRIEK_PUBLIC_URL is required"),
        ({"CHRIEK_PUBLIC_URL": f"https://alice:{MARKER}@images.example"}, "credentials"),
        ({"CHRIEK_PUBLIC_URL": f"https://{MARKER}@images.example"}, "credentials"),
        ({"CHRIEK_PUBLIC_URL": f"https://images.example/?token={MARKER}"}, "query or fragment"),
        ({"CHRIEK_PUBLIC_URL": f"https://images.example/#{MARKER}"}, "query or fragment"),
        ({"CHRIEK_PUBLIC_URL": "https://images.example?"}, "query or fragment"),
        ({"CHRIEK_PUBLIC_URL": "https://images.example#"}, "query or fragment"),
        ({"CHRIEK_PUBLIC_URL": f"ftp://images.example/{MARKER}"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": f"images.example/{MARKER}"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": f"/{MARKER}"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https:images.example"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https:///images.example"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example:"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example:0"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example:65536"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example:8x"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example:-1"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example:80:80"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://[::1"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://[not:an:ipv6]"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images..example"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://ima<ges.example"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": f" https://images.example/{MARKER}"}, PRINTABLE),
        ({"CHRIEK_PUBLIC_URL": f"https://images.example/{MARKER}\t"}, PRINTABLE),
        ({"CHRIEK_PUBLIC_URL": f"https://images.\nexample/{MARKER}"}, PRINTABLE),
        ({"CHRIEK_PUBLIC_URL": f"https://images.example\x07/{MARKER}"}, PRINTABLE),
        ({"CHRIEK_PUBLIC_URL": f"https://b\u00fccher.example/{MARKER}"}, PRINTABLE),
        # Numeric hosts that clients reinterpret as a different IPv4 address, or reject.
        ({"CHRIEK_PUBLIC_URL": "https://017.0.0.1"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://2130706433"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://0x7f.1/chriek"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://1."}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.0x1"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://999.999.999.999"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://1.2.3.4.5"}, URL_FORM),
        # Prefixes that clients normalize into a different path.
        ({"CHRIEK_PUBLIC_URL": "https://images.example/chriek/.."}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example/chriek/%2e%2E/"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example/chriek/./x"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example/%2E/chriek"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example/a\\b"}, URL_FORM),
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
    assert MARKER not in output
    if overrides.get("CHRIEK_PUBLIC_URL"):
        assert overrides["CHRIEK_PUBLIC_URL"].strip() not in output
    for secret in (EMOJI_KEY, LATIN1_KEY):
        assert secret[:2] not in output
    assert not (tmp_path / "db").exists(), "nothing may start on invalid configuration"


def test_missing_public_url_exits_2_without_fallback(tmp_path):
    env = base_env(tmp_path)
    del env["CHRIEK_PUBLIC_URL"]
    process = serve(env)
    output, _ = process.communicate(timeout=30)

    assert process.returncode == 2
    assert output.startswith("chriek: configuration error: CHRIEK_PUBLIC_URL is required")
    assert not (tmp_path / "db").exists(), "nothing may start without a public URL"


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
    env: dict[str, str], *, public_url: str | None = None, **kwargs
) -> Iterator[tuple[subprocess.Popen, httpx2.Client]]:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    process = serve(
        env | {"CHRIEK_PORT": str(port), "CHRIEK_PUBLIC_URL": public_url or base}, **kwargs
    )
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


def test_chriek_serve_links_ignore_host_and_forwarded_headers(tmp_path):
    # Host reaches the request scope directly; uvicorn applies X-Forwarded-Proto from 127.0.0.1
    # by default, turning this http request into https. Forwarded is not interpreted.
    hostile = ALICE | {
        "Host": "evil.example",
        "X-Forwarded-Host": "evil.example",
        "X-Forwarded-Proto": "https",
        "X-Forwarded-For": "192.0.2.1",
        "Forwarded": "for=192.0.2.1;host=evil.example;proto=https",
    }
    origin = "http://images.example/chriek"
    with served(base_env(tmp_path), public_url=f"{origin}/") as client:
        job_id = submit(client, headers=hostile, width=16, height=16)["id"]
        body = wait_for(client, job_id, headers=hostile)
        url = body["images"][0]["url"]

        assert url.startswith(f"{origin}/images/{job_id}/0.png?expires="), url
        # The operator's proxy serves `origin` and strips `/chriek` before forwarding.
        download = client.get(url.removeprefix(origin))
        assert download.status_code == 200
        assert Image.open(io.BytesIO(download.content)).size == (16, 16)
