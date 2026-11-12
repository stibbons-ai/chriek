"""Boundaries of the fail-closed dependency audit gate (scripts/audit_dependencies.py).

Offline and deterministic: uv runs for real on fixture projects, pip-audit reports are
injected; real PyPI runs are documented in README.md.
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


def fake_pip_audit(
    monkeypatch: pytest.MonkeyPatch, returncode: int, report_text: str | None, stderr: str = ""
) -> None:
    """Replace only the pip-audit process: it writes `report_text` where asked; uv runs for real."""

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "pip_audit" not in command:
            return _real_run(command, **kwargs)  # type: ignore[call-overload]
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
        if "pip_audit" not in command:
            return _real_run(command, **kwargs)  # type: ignore[call-overload]
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(audit.subprocess, "run", run)

    assert audit.main(["--project", str(FIXTURE_CLEAN)]) == audit.EXIT_OPERATIONAL
    out = capsys.readouterr()
    assert "PASSED" not in out.out
    assert "DEPENDENCY AUDIT ERROR (operational, nothing proven): cannot run" in out.err
