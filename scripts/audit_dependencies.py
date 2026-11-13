"""Fail-closed vulnerability audit of every third-party package locked in uv.lock.

Run with `uv run --locked python scripts/audit_dependencies.py`. Exit codes:

- 0: every locked third-party package was audited; no known vulnerabilities.
- 1: known vulnerabilities found (each advisory is printed).
- 2: operational failure: stale lock, incomplete or malformed results, skipped
  packages, tool or service errors. Nothing was proven either way.

Only exit 0 passes. There is no ignore list, severity threshold or fallback.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

EXIT_CLEAN = 0
EXIT_VULNERABLE = 1
EXIT_OPERATIONAL = 2
SERVICE = "pypi"
# `-s pypi` audits by name and version on PyPI, so only packages locked from PyPI qualify.
PYPI_REGISTRY = "https://pypi.org/simple"
REPO_ROOT = Path(__file__).resolve().parents[1]
# pip-audit with PyPI responses validated before its lossy parsing; see that script.
PIP_AUDIT = Path(__file__).with_name("pip_audit_checked.py")

Inventory = dict[str, Version]


class AuditError(Exception):
    """The audit could not prove anything: an operational failure, never a pass."""


@dataclass(frozen=True)
class Finding:
    name: str
    version: str
    id: str
    aliases: tuple[str, ...]
    fix_versions: tuple[str, ...]

    def describe(self) -> str:
        return (
            f"{self.name}=={self.version}: {self.id}"
            f" aliases [{', '.join(self.aliases)}]"
            f" fixed in [{', '.join(self.fix_versions)}]"
        )


def _version(name: str, raw: object) -> Version:
    if not isinstance(raw, str):
        raise AuditError(f"{name}: version is not a string: {raw!r}")
    try:
        return Version(raw)
    except InvalidVersion as exc:
        raise AuditError(f"{name}: invalid version {raw!r}") from exc


def _add(inventory: Inventory, name: str, version: Version, origin: str) -> str:
    key = canonicalize_name(name)
    if key in inventory:
        raise AuditError(
            f"{origin}: {key} appears more than once ({inventory[key]}, {version}); "
            "the gate audits exactly one locked version per package"
        )
    inventory[key] = version
    return key


def parse_export(text: str) -> Inventory:
    """Parse `uv export --no-hashes` output; every requirement must be `name==version`."""
    inventory: Inventory = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # The environment marker is dropped on purpose: pip-audit silently skips
        # requirements whose marker does not match the auditing machine.
        requirement = line.split(";", 1)[0].strip()
        name, sep, version = requirement.partition("==")
        if not sep or not name or " " in requirement or "@" in requirement:
            raise AuditError(f"uv export: unauditable requirement line: {raw_line!r}")
        _add(inventory, name, _version(name, version), "uv export")
    return inventory


def parse_lock(text: str) -> Inventory:
    """Every third-party package in uv.lock, each of which must be locked from PyPI.

    Only the local project at `.` is excluded. A package from another registry, a git, URL or
    path source, or a malformed entry raises AuditError.
    """
    try:
        packages = tomllib.loads(text)["package"]
    except (tomllib.TOMLDecodeError, KeyError) as exc:
        raise AuditError(f"uv.lock: unreadable: {exc}") from exc
    if not isinstance(packages, list):
        raise AuditError(f"uv.lock: package is not a list of tables: {packages!r}")
    inventory: Inventory = {}
    for package in packages:
        if not (
            isinstance(package, dict)
            and isinstance(package.get("name"), str)
            and isinstance(package.get("source"), dict)
        ):
            raise AuditError(f"uv.lock: malformed package entry {package!r}")
        name, source = package["name"], package["source"]
        if source.get("registry") == PYPI_REGISTRY:
            _add(inventory, name, _version(name, package.get("version")), "uv.lock")
        elif "registry" in source:
            raise AuditError(
                f"uv.lock: {name} is locked from registry {source['registry']}; "
                f"only {PYPI_REGISTRY} can be audited against PyPI"
            )
        elif source.get("editable") != "." and source.get("virtual") != ".":
            raise AuditError(f"uv.lock: {name} is not from a package registry ({source})")
    return inventory


def collect_inventory(export_text: str, lock_text: str) -> Inventory:
    """The exported inventory, proven to cover exactly the lock's third-party packages."""
    exported = parse_export(export_text)
    _require_same("uv export does not match uv.lock", parse_lock(lock_text), exported)
    if not exported:
        raise AuditError("no third-party packages to audit")
    return exported


def _require_same(label: str, expected: Inventory, actual: Inventory) -> None:
    """AuditError naming every difference unless `actual` holds exactly `expected`."""
    if actual == expected:
        return
    problems = [f"missing {name}=={expected[name]}" for name in sorted(expected.keys() - actual)]
    problems += [f"unexpected {name}=={actual[name]}" for name in sorted(actual.keys() - expected)]
    problems += [
        f"{name} expected {expected[name]}, got {actual[name]}"
        for name in sorted(expected.keys() & actual.keys())
        if expected[name] != actual[name]
    ]
    raise AuditError(f"{label}: {'; '.join(problems)}")


def _strings(owner: str, field: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AuditError(f"pip-audit report: {owner}: {field} is not a list of strings")
    return tuple(value)


def evaluate(inventory: Inventory, returncode: int, report_text: str | None) -> list[Finding]:
    """Findings from a complete pip-audit JSON report; AuditError for anything less."""
    if report_text is None:
        raise AuditError(f"pip-audit wrote no report (exit {returncode})")
    try:
        report = json.loads(report_text)
    except json.JSONDecodeError as exc:
        raise AuditError(f"pip-audit report is not JSON: {exc}") from exc
    entries = report.get("dependencies") if isinstance(report, dict) else None
    if not isinstance(entries, list):
        raise AuditError("pip-audit report has no dependency list")

    audited: Inventory = {}
    skipped: list[str] = []
    # Keyed by (package, advisory): the PyPI service can report one advisory more than once.
    findings: dict[tuple[str, str], Finding] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise AuditError(f"pip-audit report: malformed dependency entry {entry!r}")
        name = entry["name"]
        if "skip_reason" in entry:
            skipped.append(f"{name}: {entry['skip_reason']}")
            continue
        version = _version(name, entry.get("version"))
        key = _add(audited, name, version, "pip-audit report")
        vulns = entry.get("vulns")
        if not isinstance(vulns, list):
            raise AuditError(f"pip-audit report: {name}: vulns is not a list")
        for vuln in vulns:
            if not isinstance(vuln, dict) or not isinstance(vuln.get("id"), str):
                raise AuditError(f"pip-audit report: {name}: malformed vulnerability {vuln!r}")
            finding = Finding(
                name=key,
                version=str(version),
                id=vuln["id"],
                aliases=_strings(name, "aliases", vuln.get("aliases", [])),
                fix_versions=_strings(name, "fix_versions", vuln.get("fix_versions")),
            )
            findings.setdefault((key, finding.id), finding)

    if skipped:
        raise AuditError("pip-audit skipped packages: " + "; ".join(skipped))
    _require_same("pip-audit did not audit exactly the locked packages", inventory, audited)
    # pip-audit's own exit codes: 1 when it reports vulnerabilities, 0 when it reports none.
    if returncode != (1 if findings else 0):
        raise AuditError(
            f"pip-audit exited {returncode} with {len(findings)} findings in its report"
        )
    return list(findings.values())


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    except OSError as exc:
        raise AuditError(f"cannot run {command[0]}: {exc}") from exc


def _export_requirements(project: Path) -> str:
    """Exact locked versions; `--locked` fails when uv.lock is stale against pyproject.toml."""
    uv = os.environ.get("UV") or shutil.which("uv")
    if uv is None:
        raise AuditError("uv not found")
    export = _run(
        [
            uv,
            "export",
            "--project",
            str(project),
            "--locked",
            "--format",
            "requirements.txt",
            "--all-groups",
            "--all-extras",
            "--no-emit-project",
            "--no-hashes",
            "--no-header",
            "--quiet",
        ]
    )
    if export.returncode != 0:
        raise AuditError(f"uv export --locked failed (exit {export.returncode}):\n{export.stderr}")
    return export.stdout


def _run_pip_audit(
    inventory: Inventory,
) -> tuple[subprocess.CompletedProcess[str], str | None, str]:
    """pip-audit on exactly `inventory`, markers dropped, with a fresh HTTP cache.

    Returns the process, its JSON report (None if it wrote none) and the record of PyPI
    responses validated before pip-audit parsed them.
    """
    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements.txt"
        report = Path(tmp) / "report.json"
        validated = Path(tmp) / "validated.txt"
        requirements.write_text("".join(f"{name}=={v}\n" for name, v in inventory.items()))
        result = _run(
            [
                sys.executable,
                str(PIP_AUDIT),
                str(validated),
                "--requirement",
                str(requirements),
                "--no-deps",
                "--disable-pip",
                "--strict",
                "--vulnerability-service",
                SERVICE,
                "--format",
                "json",
                "--output",
                str(report),
                "--progress-spinner",
                "off",
                "--cache-dir",
                str(Path(tmp) / "cache"),
            ]
        )
        report_text = report.read_text() if report.exists() else None
        return result, report_text, validated.read_text() if validated.exists() else ""


def check_validated(inventory: Inventory, record_text: str) -> None:
    """Every audited package's PyPI response passed validation, exactly once."""
    validated: Inventory = {}
    for line in record_text.splitlines():
        name, _, version = line.partition("==")
        _add(validated, name, _version(name, version), "validated PyPI responses")
    _require_same(
        "PyPI responses were not validated for exactly the locked packages", inventory, validated
    )


def audit(project: Path) -> list[Finding]:
    export_text = _export_requirements(project)
    try:
        lock_text = (project / "uv.lock").read_text()
    except OSError as exc:
        raise AuditError(f"uv.lock: {exc}") from exc
    inventory = collect_inventory(export_text, lock_text)
    print(f"inventory: {len(inventory)} locked third-party packages")
    result, report_text, record_text = _run_pip_audit(inventory)
    findings: list[Finding] = []
    try:
        findings = evaluate(inventory, result.returncode, report_text)
        check_validated(inventory, record_text)
    except AuditError as exc:
        # Findings parsed before validation failed: shown, but not proven by validated responses.
        unconfirmed = "".join(f"\nUNCONFIRMED {f.describe()}" for f in findings)
        raise AuditError(f"{exc}{unconfirmed}\npip-audit stderr:\n{result.stderr}") from exc
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--project",
        type=Path,
        default=REPO_ROOT,
        help="uv project whose uv.lock is audited (default: this repository)",
    )
    project = parser.parse_args(argv).project.resolve()
    try:
        print(
            f"dependency audit {datetime.now(UTC).isoformat(timespec='seconds')}: "
            f"project {project}, Python {platform.python_version()}, {platform.platform()}, "
            f"pip-audit {metadata.version('pip-audit')}, service {SERVICE}"
        )
        findings = audit(project)
    except AuditError as exc:
        print(f"DEPENDENCY AUDIT ERROR (operational, nothing proven): {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL
    except Exception:
        # An unexpected crash must not exit 1, which means "vulnerable".
        traceback.print_exc()
        print("DEPENDENCY AUDIT ERROR (operational, nothing proven): crashed", file=sys.stderr)
        return EXIT_OPERATIONAL
    if findings:
        for finding in findings:
            print(f"VULNERABLE {finding.describe()}", file=sys.stderr)
        print(f"DEPENDENCY AUDIT FAILED: {len(findings)} known vulnerabilities", file=sys.stderr)
        return EXIT_VULNERABLE
    print("DEPENDENCY AUDIT PASSED: every locked third-party package audited, none vulnerable")
    return EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
