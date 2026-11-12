"""Boundaries of the fail-closed dependency audit gate (scripts/audit_dependencies.py).

Offline and deterministic: uv runs for real on fixture projects; pip-audit reports are
injected, or the real pinned pip-audit runs against canned PyPI HTTP responses. Real PyPI
runs are documented in README.md.
"""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "audit_dependencies", ROOT / "scripts" / "audit_dependencies.py"
)
assert _spec is not None and _spec.loader is not None
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

LOCK = """
version = 1

[[package]]
name = "app"
version = "0.1.0"
source = { editable = "." }

[[package]]
name = "colorama"
version = "0.4.6"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "Typing_Extensions"
version = "4.16.0"
source = { registry = "https://pypi.org/simple" }
"""
EXPORT = "colorama==0.4.6 ; sys_platform == 'win32'\ntyping-extensions==4.16.0\n"
INVENTORY = audit.collect_inventory(EXPORT, LOCK)


def report(*entries: dict) -> str:
    return json.dumps({"dependencies": list(entries), "fixes": []})


def dep(name: str, version: str, *vulns: dict) -> dict:
    return {"name": name, "version": version, "vulns": list(vulns)}


VULN = {"id": "PYSEC-1", "aliases": ["CVE-1"], "fix_versions": ["0.4.7"], "description": ""}
CLEAN = report(dep("colorama", "0.4.6"), dep("typing-extensions", "4.16.0"))
FIXTURE_CLEAN = ROOT / "tests" / "fixtures" / "dependency_audit" / "clean"


def test_inventory_covers_marker_guarded_packages_and_excludes_only_the_project() -> None:
    assert {name: str(v) for name, v in INVENTORY.items()} == {
        "colorama": "0.4.6",
        "typing-extensions": "4.16.0",
    }


@pytest.mark.parametrize(
    ("export", "lock", "message"),
    [
        ("typing-extensions==4.16.0\n", LOCK, "missing colorama==0.4.6"),
        (EXPORT.replace("4.16.0", "4.15.0"), LOCK, "expected 4.16.0, got 4.15.0"),
        (EXPORT + "extra==1.0\n", LOCK, "unexpected extra==1.0"),
        (EXPORT + "pkg @ git+https://example.com/pkg\n", LOCK, "unauditable requirement"),
        (EXPORT + "pkg>=1\n", LOCK, "unauditable requirement"),
        (EXPORT + "colorama==0.4.5 ; sys_platform == 'linux'\n", LOCK, "more than once"),
        (
            EXPORT,
            LOCK + '[[package]]\nname = "lib"\nversion = "1"\nsource = { git = "https://x" }\n',
            "not from a package registry",
        ),
        ("", '[[package]]\nname = "app"\nversion = "0"\nsource = { virtual = "." }\n', "no third"),
    ],
)
def test_incomplete_or_unauditable_inventory_is_rejected(
    export: str, lock: str, message: str
) -> None:
    with pytest.raises(audit.AuditError, match=message):
        audit.collect_inventory(export, lock)


def test_complete_clean_report_passes() -> None:
    assert audit.evaluate(INVENTORY, 0, CLEAN) == []


def test_vulnerable_report_returns_each_advisory_once() -> None:
    text = report(dep("colorama", "0.4.6", VULN, VULN), dep("typing-extensions", "4.16.0"))
    assert audit.evaluate(INVENTORY, 1, text) == [
        audit.Finding("colorama", "0.4.6", "PYSEC-1", ("CVE-1",), ("0.4.7",))
    ]


@pytest.mark.parametrize(
    ("returncode", "text", "message"),
    [
        (1, None, "wrote no report"),
        (0, "{not json", "not JSON"),
        (0, json.dumps({"fixes": []}), "no dependency list"),
        (0, report(dep("colorama", "0.4.6")), "missing typing-extensions==4.16.0"),
        (
            0,
            report(dep("colorama", "0.4.6"), {"name": "typing-extensions", "skip_reason": "404"}),
            "skipped packages: typing-extensions: 404",
        ),
        (0, report(dep("colorama", "0.4.5"), dep("typing-extensions", "4.16.0")), "expected"),
        (0, report(dep("colorama", "0.4.6"), {"name": "typing-extensions"}), "version"),
        (
            0,
            report(dep("colorama", "0.4.6", {"aliases": []}), dep("typing-extensions", "4.16.0")),
            "malformed vulnerability",
        ),
        (1, CLEAN, "exited 1 with 0 findings"),
        (2, CLEAN, "exited 2 with 0 findings"),
        (
            0,
            report(dep("colorama", "0.4.6", VULN), dep("typing-extensions", "4.16.0")),
            "exited 0 with 1 findings",
        ),
    ],
)
def test_incomplete_malformed_or_inconsistent_results_never_pass(
    returncode: int, text: str | None, message: str
) -> None:
    with pytest.raises(audit.AuditError, match=message):
        audit.evaluate(INVENTORY, returncode, text)


def test_stale_lock_is_an_operational_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    shutil.copytree(FIXTURE_CLEAN, project)
    pyproject = project / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace('["iniconfig==2.3.1"]', "[]"))

    assert audit.main(["--project", str(project)]) == audit.EXIT_OPERATIONAL
    assert "uv export --locked failed" in capsys.readouterr().err


def test_missing_lock_is_an_operational_failure_and_nothing_is_resolved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(FIXTURE_CLEAN / "pyproject.toml", project)

    assert audit.main(["--project", str(project)]) == audit.EXIT_OPERATIONAL
    assert "uv export --locked failed" in capsys.readouterr().err
    assert not (project / "uv.lock").exists()


_real_run = subprocess.run


def is_pip_audit(command: list[str]) -> bool:
    return str(audit.PIP_AUDIT) in command


def fake_pip_audit(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    report_text: str | None,
    stderr: str = "",
    validated: str | None = None,
) -> None:
    """Replace only the pip-audit process; uv runs for real.

    It writes `report_text` where asked and records `validated` as the PyPI responses it
    checked (default: one per requested package).
    """

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if not is_pip_audit(command):
            return _real_run(command, **kwargs)  # type: ignore[call-overload]
        requested = Path(command[command.index("--requirement") + 1]).read_text()
        Path(command[2]).write_text(requested if validated is None else validated)
        if report_text is not None:
            Path(command[command.index("--output") + 1]).write_text(report_text)
        return subprocess.CompletedProcess(command, returncode, "", stderr)

    monkeypatch.setattr(audit.subprocess, "run", run)


def iniconfig(*vulns: dict) -> str:
    return report(dep("iniconfig", "2.3.1", *vulns))


def test_gate_passes_only_a_complete_clean_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_pip_audit(monkeypatch, 0, iniconfig())

    assert audit.main(["--project", str(FIXTURE_CLEAN)]) == audit.EXIT_CLEAN
    assert "DEPENDENCY AUDIT PASSED" in capsys.readouterr().out


def test_gate_fails_and_emits_every_distinct_advisory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    other = {"id": "GHSA-2", "aliases": [], "fix_versions": []}
    fake_pip_audit(monkeypatch, 1, iniconfig(VULN, other, VULN))

    assert audit.main(["--project", str(FIXTURE_CLEAN)]) == audit.EXIT_VULNERABLE
    out = capsys.readouterr()
    assert "PASSED" not in out.out
    assert "VULNERABLE iniconfig==2.3.1: PYSEC-1 aliases [CVE-1] fixed in [0.4.7]" in out.err
    assert "VULNERABLE iniconfig==2.3.1: GHSA-2 aliases [] fixed in []" in out.err
    assert "DEPENDENCY AUDIT FAILED: 2 known vulnerabilities" in out.err


@pytest.mark.parametrize(
    ("returncode", "text", "message"),
    [
        (1, None, "wrote no report (exit 1)"),
        (0, "{not json", "not JSON"),
        (0, report(), "missing iniconfig==2.3.1"),
        (0, report({"name": "iniconfig", "skip_reason": "404"}), "skipped packages: iniconfig"),
    ],
)
def test_gate_reports_unusable_auditor_results_as_operational_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    returncode: int,
    text: str | None,
    message: str,
) -> None:
    fake_pip_audit(monkeypatch, returncode, text, stderr="ConnectionError: pypi.org unreachable")

    assert audit.main(["--project", str(FIXTURE_CLEAN)]) == audit.EXIT_OPERATIONAL
    out = capsys.readouterr()
    assert "PASSED" not in out.out
    assert "DEPENDENCY AUDIT ERROR (operational, nothing proven)" in out.err
    assert message in out.err
    assert "ConnectionError: pypi.org unreachable" in out.err


def test_gate_reports_an_auditor_that_cannot_start_as_an_operational_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if not is_pip_audit(command):
            return _real_run(command, **kwargs)  # type: ignore[call-overload]
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(audit.subprocess, "run", run)

    assert audit.main(["--project", str(FIXTURE_CLEAN)]) == audit.EXIT_OPERATIONAL
    out = capsys.readouterr()
    assert "PASSED" not in out.out
    assert "DEPENDENCY AUDIT ERROR (operational, nothing proven): cannot run" in out.err


def test_gate_rejects_a_report_whose_service_responses_were_not_all_validated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_pip_audit(monkeypatch, 0, iniconfig(), validated="")

    assert audit.main(["--project", str(FIXTURE_CLEAN)]) == audit.EXIT_OPERATIONAL
    out = capsys.readouterr()
    assert "PASSED" not in out.out
    assert "not validated for exactly the locked packages: missing iniconfig==2.3.1" in out.err


# Loaded into the real pip-audit process: replaces only the HTTP transport under pip-audit's
# own requests session, so its cache layer, PyPI service, CLI and JSON report all run for real.
TRANSPORT = """
import io, json, os
import requests.adapters
from urllib3 import HTTPResponse

RESPONSES = json.loads(os.environ["AUDIT_TEST_PYPI_RESPONSES"])

def send(self, request, *args, **kwargs):
    if request.url not in RESPONSES:
        raise requests.ConnectionError(f"offline test: no response for {request.url}")
    status, body = RESPONSES[request.url]
    raw = HTTPResponse(
        body=io.BytesIO(body.encode()),
        headers={"Content-Type": "application/json"},
        status=status,
        preload_content=False,
        request_method=request.method,
        request_url=request.url,
    )
    return self.build_response(request, raw)

requests.adapters.HTTPAdapter.send = send
"""
INICONFIG_URL = "https://pypi.org/pypi/iniconfig/2.3.1/json"
ADVISORY = {
    "id": "PYSEC-9",
    "aliases": ["CVE-9"],
    "fixed_in": ["2.3.2"],
    "summary": None,
    "details": "details",
    "withdrawn": None,
    "link": "https://osv.dev/vulnerability/PYSEC-9",
    "source": "osv",
}


def pypi_serves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int, body: str) -> int:
    """Run the real gate and pinned pip-audit; PyPI answers iniconfig 2.3.1 with `body`."""
    (tmp_path / "sitecustomize.py").write_text(TRANSPORT)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("AUDIT_TEST_PYPI_RESPONSES", json.dumps({INICONFIG_URL: [status, body]}))
    return audit.main(["--project", str(FIXTURE_CLEAN)])


def payload(*vulns: dict) -> str:
    return json.dumps({"info": {"name": "iniconfig"}, "vulnerabilities": list(vulns)})


@pytest.mark.parametrize(
    ("body", "code", "message"),
    [
        (payload(), audit.EXIT_CLEAN, "DEPENDENCY AUDIT PASSED"),
        (payload(dict(ADVISORY, withdrawn="2026-01-01T00:00:00Z")), audit.EXIT_CLEAN, "PASSED"),
        # Lower-case "t", fraction and offset are all valid RFC 3339.
        (
            payload(dict(ADVISORY, withdrawn="2016-12-31t23:59:59.5+05:30")),
            audit.EXIT_CLEAN,
            "PASSED",
        ),
        (
            payload(ADVISORY),
            audit.EXIT_VULNERABLE,
            "VULNERABLE iniconfig==2.3.1: PYSEC-9 aliases [CVE-9] fixed in [2.3.2]",
        ),
    ],
)
def test_valid_pypi_responses_keep_their_meaning_through_the_real_auditor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    body: str,
    code: int,
    message: str,
) -> None:
    assert pypi_serves(monkeypatch, tmp_path, 200, body) == code
    out = capsys.readouterr()
    assert message in out.out + out.err


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        # Each 200 body below is one pinned pip-audit 2.10.1 turns into a clean result.
        (200, json.dumps({"vulnerabilities": {}}), "vulnerabilities is not a list: {}"),
        (200, json.dumps({"vulnerabilities": ""}), "vulnerabilities is not a list: ''"),
        (200, json.dumps({"vulnerabilities": None}), "vulnerabilities is not a list: None"),
        (200, json.dumps({"info": {}}), "vulnerabilities is not a list: None"),
        (200, payload(dict(ADVISORY, withdrawn=False)), "withdrawn is not a timestamp: False"),
        (200, payload(dict(ADVISORY, withdrawn="")), "withdrawn is not a timestamp: ''"),
        (200, payload(dict(ADVISORY, withdrawn="not-a-timestamp")), "withdrawn is not a"),
        (200, payload(dict(ADVISORY, withdrawn="2026-02-30T00:00:00Z")), "withdrawn is not a"),
        # Python's datetime accepts both; RFC 3339 requires a "T" and a UTC offset.
        (200, payload(dict(ADVISORY, withdrawn="2026-01-01 00:00:00Z")), "withdrawn is not a"),
        (200, payload(dict(ADVISORY, withdrawn="2026-01-01T00:00:00")), "withdrawn is not a"),
        (200, payload(dict(ADVISORY, withdrawn="2026-01-01T00:00:00+24:00")), "withdrawn is not"),
        # Leap seconds are unsupported: midday, wrong-offset and a real one (2016-12-31) all fail.
        (200, payload(dict(ADVISORY, withdrawn="2026-01-01T12:34:60Z")), "withdrawn is not"),
        (200, payload(dict(ADVISORY, withdrawn="2016-12-31t23:59:60.5+05:30")), "withdrawn is"),
        (200, payload(dict(ADVISORY, withdrawn="2016-12-31T23:59:60Z")), "withdrawn is not"),
        # Lossy rather than clean: pip-audit would split the string into one alias per character.
        (200, payload(dict(ADVISORY, aliases="CVE-9")), "aliases is not a list of strings"),
        (200, payload(dict(ADVISORY, id=None)), "entry without a string id"),
        (200, json.dumps([]), "vulnerabilities is not a list"),
        (200, "<html>maintenance</html>", "not JSON"),
        (503, "", "ServiceError"),
    ],
)
def test_malformed_or_failed_pypi_responses_are_operational_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: int,
    body: str,
    message: str,
) -> None:
    assert pypi_serves(monkeypatch, tmp_path, status, body) == audit.EXIT_OPERATIONAL
    out = capsys.readouterr()
    assert "PASSED" not in out.out
    assert "DEPENDENCY AUDIT ERROR (operational, nothing proven)" in out.err
    assert message in out.err
