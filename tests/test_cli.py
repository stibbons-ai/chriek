"""The installed `chriek serve` command: safe startup errors, live HTTP and HTTPS round trips,
and operator walkthroughs for api-key mode, proxy mode and native TLS."""

import io
import os
import shutil
import socket
import ssl
import stat
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
    BOB_KEY,
    SIGNING_SECRET,
    SMALL,
    base_env,
    modes,
    post_job,
    proxy_env,
    stored_owners,
    submit,
    tree_state,
    wait_for,
    with_query,
)
from PIL import Image

from chriek.auth import PROXY_OWNER

CHRIEK = Path(sys.executable).parent / "chriek"


def serve(env: dict[str, str], **kwargs) -> subprocess.Popen:
    return subprocess.Popen(
        [str(CHRIEK), "serve"],
        env={"PATH": os.environ["PATH"]} | env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        umask=0o022,  # the common permissive default; storage must stay private under it
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
DB_PASSWORD = "db-password-0123456789"
LATIN1_KEY = "\N{LATIN SMALL LETTER E WITH ACUTE}" * 32
CONTROL_KEY = ALICE_KEY + "\x07"
MARKER = "public-url-secret-marker"
URL_FORM = "CHRIEK_PUBLIC_URL must be an absolute http or https URL"
PRINTABLE = "CHRIEK_PUBLIC_URL must contain only printable ASCII"
UNENCODED_AT_PASSWORD = ("S3cr", "tTail-9999")  # written as S3cr@tTail-9999


def assert_config_error(process: subprocess.Popen, tmp_path: Path, message: str) -> str:
    """Startup stopped with exit code 2, named the problem, leaked no secret, opened nothing."""
    output = exit_output(process)

    assert process.returncode == 2
    assert output.startswith("chriek: configuration error: ")
    assert message in output
    for secret in (
        ALICE_KEY,
        BOB_KEY,
        SIGNING_SECRET,
        "short-secret-key",
        "short-signing-secret",
        DB_PASSWORD,
        *UNENCODED_AT_PASSWORD,
    ):
        assert secret not in output
    for secret in (EMOJI_KEY, LATIN1_KEY):
        assert secret[:2] not in output
    assert "PRIVATE KEY" not in output
    assert "Traceback" not in output
    assert not (tmp_path / "db").exists(), "nothing may start on invalid configuration"
    return output


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"CHRIEK_API_KEYS": None}, "CHRIEK_API_KEYS must name at least one key"),
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
        # Reversed name:key entries put secrets in the name position; errors cite positions.
        ({"CHRIEK_API_KEYS": f"{ALICE_KEY}:alice"}, "CHRIEK_API_KEYS entry 1 key must"),
        (
            {"CHRIEK_API_KEYS": f"{ALICE_KEY}:{BOB_KEY},{ALICE_KEY}:{BOB_KEY}z"},
            "CHRIEK_API_KEYS entry 2 repeats",
        ),
        (
            {"CHRIEK_API_KEYS": f"{ALICE_KEY}x:{BOB_KEY},{ALICE_KEY}:{BOB_KEY}"},
            "CHRIEK_API_KEYS entry 2 key is shared",
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
        ({"CHRIEK_PUBLIC_URL": f"https://images.example:0/{MARKER}"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example:65536"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example:8x"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example:-1"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example:80:80"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": f"https://[::1/{MARKER}"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": f"https://alice:{MARKER}@[::1"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://[v1.fe]"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://[fe80::1%25eth0]"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://[not:an:ipv6]"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images..example"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://ima<ges.example"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": f" https://images.example/{MARKER}"}, PRINTABLE),
        ({"CHRIEK_PUBLIC_URL": f"https://images.example/{MARKER}\t"}, PRINTABLE),
        ({"CHRIEK_PUBLIC_URL": f"https://images.\nexample/{MARKER}"}, PRINTABLE),
        ({"CHRIEK_PUBLIC_URL": f"https://images.example\x07/{MARKER}"}, PRINTABLE),
        ({"CHRIEK_PUBLIC_URL": f"https://images.example\x7f/{MARKER}"}, PRINTABLE),
        ({"CHRIEK_PUBLIC_URL": f"https://b\u00fccher.example/{MARKER}"}, PRINTABLE),
        # Numeric hosts that clients reinterpret as a different IPv4 address, or reject.
        ({"CHRIEK_PUBLIC_URL": f"https://017.0.0.1/{MARKER}"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://2130706433"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://0x7f.1/chriek"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://1."}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.0x1"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.0X1"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.0x"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://999.999.999.999"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://1.2.3.4.5"}, URL_FORM),
        # Prefixes that clients normalize into a different path.
        ({"CHRIEK_PUBLIC_URL": f"https://images.example/{MARKER}/.."}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example/chriek/%2e%2E/"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example/chriek/./x"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example/%2E/chriek"}, URL_FORM),
        ({"CHRIEK_PUBLIC_URL": "https://images.example/a\\b"}, URL_FORM),
        ({"CHRIEK_AUTH_MODE": "Proxy"}, "CHRIEK_AUTH_MODE must be 'api-key' or 'proxy'"),
        ({"CHRIEK_AUTH_MODE": ""}, "CHRIEK_AUTH_MODE must be 'api-key' or 'proxy'"),
        ({"CHRIEK_AUTH_MODE": "proxy"}, "CHRIEK_API_KEYS"),
        (
            {"CHRIEK_AUTH_MODE": "proxy", "CHRIEK_API_KEYS": "", "CHRIEK_SIGNING_SECRET": ""},
            "CHRIEK_SIGNING_SECRET must be at least 32",
        ),
        (
            {
                "CHRIEK_AUTH_MODE": "proxy",
                "CHRIEK_API_KEYS": None,
                "CHRIEK_SIGNING_SECRET": "short-signing-secret",
            },
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
        ({"CHRIEK_MAX_QUEUED_JOB": "1000"}, "unknown setting CHRIEK_MAX_QUEUED_JOB"),
        ({"CHRIEK_SIGNING_SECRT": SIGNING_SECRET}, "unknown setting CHRIEK_SIGNING_SECRT"),
        ({"chriek_port": "8001"}, "unknown setting chriek_port"),
        ({"CHRIEK_ALLOWED_HOSTS": "localhost"}, "CHRIEK_ALLOWED_HOSTS applies only when"),
        (
            {"CHRIEK_AUTH_MODE": "proxy", "CHRIEK_API_KEYS": None, "CHRIEK_ALLOWED_HOSTS": "*"},
            "CHRIEK_ALLOWED_HOSTS entry 1",
        ),
        (
            {"CHRIEK_DATABASE_URL": f"postgresql://chriek:{DB_PASSWORD}@db:port/chriek"},
            "CHRIEK_DATABASE_URL is not a valid database URL",
        ),
        ({"CHRIEK_DATABASE_URL": ""}, "CHRIEK_DATABASE_URL is not a valid database URL"),
        (
            {"CHRIEK_DATABASE_URL": f"mysql://chriek:{DB_PASSWORD}@db/chriek"},
            "CHRIEK_DATABASE_URL must start with postgresql:// or postgresql+psycopg://",
        ),
        (
            {
                "CHRIEK_DATABASE_URL": "postgresql://chriek:{}@{}@db.invalid/chriek".format(
                    *UNENCODED_AT_PASSWORD
                )
            },
            "CHRIEK_DATABASE_URL must percent-encode @ in the user name and password as %40",
        ),
        # %00 would truncate libpq's connection settings; \udcff is an undecodable 0xff byte.
        *(
            (
                {"CHRIEK_DATABASE_URL": f"postgresql://chriek:{password}@127.0.0.1:1/chriek"},
                "CHRIEK_DATABASE_URL must be UTF-8 text without NUL characters",
            )
            for password in (f"{DB_PASSWORD}%00", f"{DB_PASSWORD}\udcff")
        ),
    ],
)
def test_invalid_configuration_exits_2_without_leaking_secrets(tmp_path, overrides, message):
    # An override of None removes the variable, so nothing configured at all is covered too.
    env = base_env(tmp_path) | overrides
    env = {name: value for name, value in env.items() if value is not None}
    output = assert_config_error(serve(env), tmp_path, message)
    assert MARKER not in output
    if value := overrides.get("CHRIEK_PUBLIC_URL"):
        assert value.strip() not in output
        # Nor the host alone: some branches see only the host, which may name an internal machine.
        try:
            host = urlsplit(value.strip()).hostname
        except ValueError:
            host = None
        assert not host or host not in output.lower()


def test_misspelt_tls_variables_never_start_plain_http(tmp_path):
    cert, key = make_tls_files(tmp_path / "tls")
    env = base_env(tmp_path) | {"CHRIEK_TLS_CERT_FILE": str(cert), "CHRIEK_TLS_KEY_FILE": str(key)}
    assert_config_error(serve(env), tmp_path, "CHRIEK_TLS_CERT_FILE, CHRIEK_TLS_KEY_FILE")


def test_missing_public_url_exits_2_without_fallback(tmp_path):
    env = base_env(tmp_path)
    del env["CHRIEK_PUBLIC_URL"]
    process = serve(env)
    output, _ = process.communicate(timeout=30)

    assert process.returncode == 2
    assert output.startswith("chriek: configuration error: CHRIEK_PUBLIC_URL is required")
    assert not (tmp_path / "db").exists(), "nothing may start without a public URL"


@pytest.mark.parametrize(
    ("password", "host", "detail"),
    [
        (DB_PASSWORD, "127.0.0.1:1", "connection failed: "),
        # A driver message that echoes a password-bearing host name still hides the password.
        (DB_PASSWORD, f"{DB_PASSWORD}.invalid", "'***.invalid'"),
        # Percent-encoded reserved and non-ASCII characters are valid configuration.
        ("db-%40%25-%C3%A9-0123456789", "127.0.0.1:1", "connection failed: "),
    ],
    ids=["refused", "echoed", "encoded-unicode"],
)
def test_unreachable_postgres_exits_1_without_leaking_the_password(
    tmp_path, password, host, detail
):
    url = f"postgresql://chriek:{password}@{host}/chriek"
    process = serve(base_env(tmp_path) | {"CHRIEK_DATABASE_URL": url})
    output, _ = process.communicate(timeout=60)

    assert process.returncode == 1, output
    assert output.startswith("chriek: database error: "), output
    assert detail in output
    assert "0123456789" not in output
    assert "Traceback" not in output
    assert not (tmp_path / "db").exists(), "CHRIEK_DATABASE_URL takes precedence over the path"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_healthy(process: subprocess.Popen, base: str, verify: ssl.SSLContext | bool) -> None:
    deadline = time.monotonic() + 30
    last_error = None
    while True:
        try:
            if httpx2.get(f"{base}/health", verify=verify).json() == {"status": "ok"}:
                return
        except httpx2.TransportError as error:  # not listening yet, or a TLS failure to report
            last_error = error
        assert process.poll() is None, process.stdout.read()
        assert time.monotonic() < deadline, f"chriek serve never became healthy: {last_error!r}"
        time.sleep(0.1)


@contextmanager
def served_process(
    env: dict[str, str], *, public_url: str | None = None, ca: Path | None = None, **kwargs
) -> Iterator[tuple[subprocess.Popen, httpx2.Client]]:
    """Run `chriek serve`; with `ca`, talk HTTPS and trust only that certificate."""
    port = free_port()
    base = f"{'https' if ca else 'http'}://127.0.0.1:{port}"
    verify = ssl.create_default_context(cafile=ca) if ca else True
    process = serve(
        env | {"CHRIEK_PORT": str(port), "CHRIEK_PUBLIC_URL": public_url or base}, **kwargs
    )
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


# A non-ASCII signing secret stays valid: only API keys travel in headers.
@pytest.mark.parametrize("signing_secret", [SIGNING_SECRET, EMOJI_KEY])
def test_chriek_serve_walkthrough_across_restart(tmp_path, signing_secret):
    """Operator starts, stops and restarts `chriek serve`; alice submits, polls and downloads;
    bob can't see her job. Runs on SQLite, or on Postgres with CHRIEK_TEST_POSTGRES_URL."""
    env = base_env(tmp_path) | {"CHRIEK_HOST": "127.0.0.1", "CHRIEK_SIGNING_SECRET": signing_secret}
    with served(env) as client:
        body = wait_for(client, submit(client, prompt="otter", image_count=2, seed=1)["id"])

        assert body["status"] == "succeeded", body
        downloads = []
        for image in body["images"]:
            assert image["url"].startswith(f"{client.base_url}/images/{body['id']}/")
            downloads.append(client.get(image["url"]).content)
            png = Image.open(io.BytesIO(downloads[-1]))
            assert png.size == (1024, 1024)
            assert png.text["Seed"] == str(image["seed"])
        assert client.get(f"/jobs/{body['id']}", headers=BOB).status_code == 404
        job = body["id"]
        expected = {
            "images": 0o700,
            f"images/{job}": 0o700,
            f"images/{job}/0.png": 0o600,
            f"images/{job}/1.png": 0o600,
        }
        if "CHRIEK_DATABASE_URL" not in env:
            # Checked while serving, so the open database's WAL and SHM sidecars exist.
            expected |= {
                "db": 0o700,
                "db/chriek.db": 0o600,
                "db/chriek.db-wal": 0o600,
                "db/chriek.db-shm": 0o600,
            }
        assert modes(tmp_path) == expected

    with served(env) as client:
        again = client.get(f"/jobs/{body['id']}", headers=ALICE).json()

        assert {key: again[key] for key in body if key != "images"} == {
            key: body[key] for key in body if key != "images"
        }
        assert [(image["index"], image["seed"]) for image in again["images"]] == [(0, 1), (1, 2)]
        assert [client.get(image["url"]).content for image in again["images"]] == downloads
        # Bob gets the same 404 as for a missing job, so he never receives a download URL.
        assert client.get(f"/jobs/{body['id']}", headers=BOB).status_code == 404


def test_chriek_serve_shipped_defaults_keep_jobs_private_across_restart(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    project.chmod(0o755)  # an ordinary checkout; the database directory holds the image root
    env = {
        name: value
        for name, value in base_env(tmp_path).items()
        if name not in ("CHRIEK_DB_PATH", "CHRIEK_STORAGE_DIR")
    }
    with served(env, cwd=project) as client:
        body = wait_for(client, submit(client, width=16, height=16, seed=7)["id"])
        assert body["status"] == "succeeded", body
        first = client.get(body["images"][0]["url"]).content

    with served(env, cwd=project) as client:
        again = client.get(f"/jobs/{body['id']}", headers=ALICE).json()
        assert again["status"] == "succeeded"
        download = client.get(again["images"][0]["url"])
        job = body["id"]
        expected = {
            "chriek-data": 0o700,
            "chriek-data/images": 0o700,
            f"chriek-data/images/{job}": 0o700,
            f"chriek-data/images/{job}/0.png": 0o600,
        }
        if "CHRIEK_DATABASE_URL" not in env:
            expected |= {
                "chriek-data/chriek.db": 0o600,
                "chriek-data/chriek.db-wal": 0o600,
                "chriek-data/chriek.db-shm": 0o600,
            }
        assert modes(project) == expected

    assert download.status_code == 200
    assert download.content == first
    assert Image.open(io.BytesIO(first)).text["Seed"] == "7"
    assert stat.S_IMODE(project.stat().st_mode) == 0o755


@pytest.mark.sqlite
def test_chriek_serve_memory_database_runs_jobs_and_resets_on_restart(tmp_path):
    env = base_env(tmp_path) | {"CHRIEK_DB_PATH": ":memory:"}
    with served(env, cwd=tmp_path) as client:
        body = wait_for(client, submit(client, seed=3, width=16, height=16)["id"])
        assert body["status"] == "succeeded", body
        download = client.get(body["images"][0]["url"])
        assert download.status_code == 200
        assert Image.open(io.BytesIO(download.content)).text["Seed"] == "3"
        job = body["id"]
        assert modes(tmp_path) == {
            "images": 0o700,
            f"images/{job}": 0o700,
            f"images/{job}/0.png": 0o600,
        }

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


def test_chriek_serve_refuses_a_shared_database_directory_without_changing_it(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    (shared / "notes.txt").write_text("keep me")
    before = tree_state(shared)
    # A bare filename puts the database directly in the shared working directory.
    process = serve(base_env(tmp_path) | {"CHRIEK_DB_PATH": "chriek.db"}, cwd=shared)
    output, _ = process.communicate(timeout=30)

    assert process.returncode == 2, output
    assert (
        f"chriek: storage error: {shared} is accessible to other users (mode 0755). "
        "chriek needs dedicated directories" in output
    )
    assert "never a shared parent" in output, "the remedy must not be chmodding the shared tree"
    assert "Uvicorn running on" not in output
    assert tree_state(shared) == before


def regular_file(tmp_path: Path) -> Path:
    (tmp_path / "file").write_text("x")
    return tmp_path / "file"


def read_only_directory(tmp_path: Path) -> Path:
    (tmp_path / "read-only").mkdir(mode=0o500)
    return tmp_path / "read-only"


@pytest.mark.parametrize(
    ("variable", "below"),
    [
        pytest.param("CHRIEK_DB_PATH", regular_file, id="database below a regular file"),
        pytest.param("CHRIEK_STORAGE_DIR", regular_file, id="images below a regular file"),
        pytest.param(
            "CHRIEK_STORAGE_DIR",
            read_only_directory,
            id="images below a read-only directory",
            marks=pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes"),
        ),
    ],
)
def test_chriek_serve_reports_an_unusable_storage_path_without_a_traceback(
    tmp_path, variable, below
):
    env = base_env(tmp_path) | {variable: str(below(tmp_path) / "chriek" / "data")}
    before = tree_state(tmp_path)
    process = serve(env)
    output, _ = process.communicate(timeout=30)

    assert process.returncode == 2, output
    assert output.startswith("chriek: storage error: "), output
    assert "Traceback" not in output
    assert tree_state(tmp_path) == before, "no database or directory may be created"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
def test_chriek_serve_reports_a_database_it_cannot_create_without_a_traceback(tmp_path):
    # Both preflights pass (the read-only directory is service-owned and not group- or
    # other-writable); creating the database fails.
    read_only = read_only_directory(tmp_path)
    env = base_env(tmp_path) | {"CHRIEK_DB_PATH": str(read_only / "chriek" / "chriek.db")}
    process = serve(env)
    output, _ = process.communicate(timeout=30)

    assert process.returncode == 2, output
    assert output.startswith(f"chriek: storage error: {read_only / 'chriek'} cannot be used ("), (
        output
    )
    assert "Traceback" not in output
    assert list(read_only.iterdir()) == [], "nothing may be created below the read-only directory"


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


def tls_env(tmp_path: Path, cert: Path, key: Path, env_base=base_env) -> dict[str, str]:
    return env_base(tmp_path) | {"CHRIEK_TLS_CERTFILE": str(cert), "CHRIEK_TLS_KEYFILE": str(key)}


UNUSABLE_TLS = "must name a readable PEM certificate and its unencrypted private key"


def bad_tls_files(tmp_path: Path, kind: str) -> tuple[Path | str, Path | str]:
    cert, key = make_tls_files(tmp_path / "tls")
    if kind == "missing-cert":
        return tmp_path / "tls" / "absent.pem", key
    if kind == "missing-key":
        return cert, tmp_path / "tls" / "absent.pem"
    if kind == "directory-cert":
        return tmp_path / "tls", key
    if kind == "unreadable-key":
        if os.geteuid() == 0:
            pytest.skip("root can read files without read permission")
        key.chmod(0)
        return cert, key
    if kind == "mismatched-key":
        return cert, make_tls_files(tmp_path / "tls", name="other")[1]
    if kind == "not-pem":
        (garbage := tmp_path / "tls" / "garbage.pem").write_text("not a certificate\n")
        return garbage, key
    if kind == "key-contents":  # the key pasted where its path belongs
        return cert, key.read_text()
    if kind == "encrypted-key":
        encrypted = tmp_path / "tls" / "encrypted-key.pem"
        openssl("pkey", "-in", str(key), "-out", str(encrypted), "-aes256", "-passout", "pass:pw")
        return cert, encrypted
    raise AssertionError(kind)


CERT_UNREADABLE = "CHRIEK_TLS_CERTFILE must name a readable file"
KEY_UNREADABLE = "CHRIEK_TLS_KEYFILE must name a readable file"


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing-cert", CERT_UNREADABLE),
        ("missing-key", KEY_UNREADABLE),
        ("directory-cert", CERT_UNREADABLE),
        ("unreadable-key", KEY_UNREADABLE),
        ("key-contents", KEY_UNREADABLE),
        ("mismatched-key", UNUSABLE_TLS),
        ("not-pem", UNUSABLE_TLS),
        ("encrypted-key", "CHRIEK_TLS_KEYFILE must be an unencrypted private key"),
    ],
)
def test_unusable_tls_files_stop_startup_instead_of_serving_plain_http(tmp_path, kind, message):
    cert, key = bad_tls_files(tmp_path, kind)
    process = serve(tls_env(tmp_path, cert, key), stdin=subprocess.DEVNULL)
    output = assert_config_error(process, tmp_path, message)
    if message == CERT_UNREADABLE:  # names only the file that failed
        assert "KEYFILE" not in output
    if message == KEY_UNREADABLE:
        assert "CERTFILE" not in output
    if kind == "key-contents":
        assert not any(line in output for line in key.splitlines()[1:-1])


def test_walkthrough_api_key_operator_keeps_owners_apart(tmp_path):
    with served_process(base_env(tmp_path)) as (process, client):
        alice_job = submit(client, headers=ALICE, prompt="alice", **SMALL)
        bob_job = submit(client, headers=BOB, prompt="bob", **SMALL)
        alice = wait_for(client, alice_job["id"], headers=ALICE)
        assert wait_for(client, bob_job["id"], headers=BOB)["status"] == "succeeded"

        assert client.get(f"/jobs/{alice['id']}", headers=BOB).status_code == 404
        assert client.get(f"/jobs/{bob_job['id']}", headers=ALICE).status_code == 404
        assert client.get(f"/jobs/{alice['id']}").status_code == 401
        assert client.post("/jobs", content=b"{").status_code == 401
        assert client.get(alice["images"][0]["url"]).status_code == 200
    output = process.stdout.read()
    assert auth_mode_lines(output) == [("INFO", "api-key")], output
    assert ALICE_KEY not in output
    assert stored_owners(tmp_path) == ["alice", "bob"]


def auth_mode_lines(output: str) -> list[tuple[str, str]]:
    """(level, mode) of each `chriek` log line that reports the authentication mode."""
    return [
        (line.split()[2], "proxy" if "proxy" in line else "api-key")
        for line in output.splitlines()
        if " chriek auth mode " in line
    ]


def loopback_proxy_env(tmp_path: Path) -> dict[str, str]:
    """Proxy mode with the default allowed Host names, as for a proxy on the same host."""
    env = proxy_env(tmp_path)
    del env["CHRIEK_ALLOWED_HOSTS"]
    return env


# What the reverse proxy sends: the public host, and that the client used HTTPS.
FORWARDED_HTTPS = {"Host": "img.example.com", "X-Forwarded-Proto": "https"}


@pytest.mark.parametrize("allowed", ["127.0.0.1", "192.0.2.1"], ids=["trusted", "untrusted"])
def test_walkthrough_proxy_operator_without_api_keys_shares_one_owner(tmp_path, allowed):
    env = proxy_env(tmp_path) | {
        "CHRIEK_ALLOWED_HOSTS": "127.0.0.1,img.example.com",
        "FORWARDED_ALLOW_IPS": allowed,
    }
    with served_process(env) as (process, client):
        anonymous = submit(client, headers={}, prompt="anonymous", **SMALL)
        claimed = submit(
            client,
            headers={"X-Forwarded-User": "bob", "X-Forwarded-For": "203.0.113.7"},
            prompt="claimed",
            **SMALL,
        )

        for job in (anonymous, claimed):
            for headers in (
                {},
                ALICE,
                {"X-Forwarded-User": "someone-else", "X-Forwarded-For": "198.51.100.9"},
            ):
                assert wait_for(client, job["id"], headers=headers)["status"] == "succeeded"
        url = wait_for(client, claimed["id"])["images"][0]["url"]
        assert client.get(url).status_code == 200
        assert client.get(with_query(url, sig="0" * 64)).status_code == 403

        # The public host the proxy forwards: submit, poll and download through it.
        public = submit(client, headers=FORWARDED_HTTPS, prompt="public", **SMALL)
        forwarded = wait_for(client, public["id"], headers=FORWARDED_HTTPS)["images"][0]["url"]
        # Links name CHRIEK_PUBLIC_URL even when a trusted proxy forwards another host and scheme.
        assert forwarded.startswith(f"{client.base_url}/images/{public['id']}/0.png?"), forwarded
        path = urlsplit(forwarded)._replace(scheme="", netloc="").geturl()
        assert client.get(path, headers=FORWARDED_HTTPS).status_code == 200

        # A Host the operator didn't allow, as a DNS rebinding page would send it.
        hostile = {"Host": "evil.example", "X-Forwarded-Host": "img.example.com"}
        assert post_job(client, "rebound", headers=hostile, **SMALL).status_code == 400
        assert client.get(f"/jobs/{public['id']}", headers=hostile).status_code == 400
        assert client.get(path, headers=hostile).status_code == 400
    assert stored_owners(tmp_path) == [PROXY_OWNER] * 3
    output = process.stdout.read()
    assert auth_mode_lines(output) == [("WARNING", "proxy")], output
    assert SIGNING_SECRET not in output


def raw_status(port: int, request: bytes) -> bytes:
    """Status line of one raw HTTP request, for Host headers httpx won't send."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(request)
        return sock.makefile("rb").readline().rstrip()


def test_proxy_server_checks_the_raw_host_header(tmp_path):
    with served(loopback_proxy_env(tmp_path)) as client:
        port = client.base_url.port
        for host in (b"localhost", b"LOCALHOST:8000", b"127.0.0.1", b"[::1]", b"[::1]:8000"):
            request = b"GET /health HTTP/1.1\r\nHost: " + host + b"\r\nConnection: close\r\n\r\n"
            assert raw_status(port, request) == b"HTTP/1.1 200 OK", host
        for request in (
            b"GET /health HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n",
            b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nHost: evil.example\r\n\r\n",
            b"GET /health HTTP/1.1\r\nHost: evil.example\r\nHost: 127.0.0.1\r\n\r\n",
            b"GET /health HTTP/1.1\r\nConnection: close\r\n\r\n",
            b"GET /health HTTP/1.0\r\n\r\n",
        ):
            assert raw_status(port, request) == b"HTTP/1.1 400 Bad Request", request


@pytest.mark.parametrize("env_base", [base_env, loopback_proxy_env], ids=["api-key", "proxy"])
def test_walkthrough_native_tls_operator_downloads_over_https_only(tmp_path, env_base):
    cert, key = make_tls_files(tmp_path / "tls")
    # uvicorn ignores SSLKEYLOGFILE, so a key log path in a missing directory must not stop TLS.
    env = tls_env(tmp_path, cert, key, env_base) | {
        "SSLKEYLOGFILE": str(tmp_path / "absent" / "keys.log")
    }
    with served_process(env, ca=cert) as (process, client):
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
