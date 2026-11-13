"""Run the pinned pip-audit CLI with every PyPI response validated before pip-audit parses it.

pip-audit 2.10.1's PyPI service iterates whatever `vulnerabilities` holds and skips entries
with any non-None `withdrawn`, so a malformed HTTP 200 response such as
`{"vulnerabilities": {}}` silently becomes a clean result. A requests response hook on the
service's own session rejects such responses with a ServiceError, which aborts pip-audit
before it writes a report.

Usage: python scripts/pip_audit_checked.py RECORD [pip-audit arguments]

Each validated response appends `name==version` to RECORD; the gate requires exactly one
line per audited package, so a hook that stops firing cannot pass silently.
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import requests
from pip_audit._cli import audit
from pip_audit._service import PyPIService
from pip_audit._service.interface import ServiceError

# The only URL pip-audit's PyPI service fetches: /pypi/<canonical name>/<version>/json.
PYPI_URL = re.compile(r"https://pypi\.org/pypi/([^/]+)/([^/]+)/json")
# RFC 3339 section 5.6 date-time; PyPI writes `withdrawn` as %Y-%m-%dT%H:%M:%SZ.
RFC3339 = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[Tt](\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:[Zz]|[+-](\d{2}):(\d{2}))",
    re.ASCII,
)


def _is_timestamp(value: object) -> bool:
    """Whether value is an RFC 3339 date-time naming a real calendar instant, leap seconds excluded.

    datetime rejects second 60, so a leap-second `withdrawn` is an operational error rather
    than a reason to skip the advisory; a leap-second calendar is not worth the risk.
    """
    match = RFC3339.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return False
    year, month, day, hour, minute, second, *offset = (int(g or 0) for g in match.groups())
    if offset[0] > 23 or offset[1] > 59:
        return False
    try:
        datetime(year, month, day, hour, minute, second)
    except ValueError:
        return False
    return True


def _is_strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _check_vulnerability(vuln: object) -> str | None:
    """The problem with one advisory entry, in the fields pip-audit consumes, or None."""
    if not isinstance(vuln, dict):
        return f"entry is not an object: {vuln!r}"
    if not isinstance(vuln.get("id"), str) or not vuln["id"]:
        return f"entry without a string id: {vuln!r}"
    for field in ("aliases", "fixed_in"):
        if not _is_strings(vuln.get(field)):
            return f"{vuln['id']}: {field} is not a list of strings"
    # pip-audit drops an entry whose `withdrawn` is anything but None; only a timestamp may.
    withdrawn = vuln.get("withdrawn")
    if withdrawn is not None and not _is_timestamp(withdrawn):
        return f"{vuln['id']}: withdrawn is not a timestamp: {withdrawn!r}"
    for field in ("summary", "details"):
        if not isinstance(vuln.get(field), str | None):
            return f"{vuln['id']}: {field} is not a string"
    return None


def _check_payload(payload: object) -> str | None:
    vulns = payload.get("vulnerabilities") if isinstance(payload, dict) else None
    if not isinstance(vulns, list):
        return f"vulnerabilities is not a list: {vulns!r}"
    for vuln in vulns:
        if problem := _check_vulnerability(vuln):
            return problem
    return None


def validating_hook(record: Path):
    def check(response: requests.Response, *args: object, **kwargs: object) -> None:
        # Only 200 bodies are validated and recorded. pip-audit raises 4xx/5xx itself (404 is
        # a skip, fatal under --strict); any other status, such as 203 or an unfollowed 3xx, is
        # parsed unvalidated and fails the gate only because its record line is missing.
        if response.status_code != 200:
            return
        url = PYPI_URL.fullmatch(response.url)
        if url is None:
            raise ServiceError(f"unexpected PyPI response URL {response.url}")
        try:
            problem = _check_payload(response.json())
        except requests.JSONDecodeError as exc:
            problem = f"not JSON: {exc}"
        if problem:
            raise ServiceError(f"malformed PyPI response for {response.url}: {problem}")
        with record.open("a") as out:
            out.write(f"{url[1]}=={url[2]}\n")

    return check


def main() -> None:
    record = Path(sys.argv[1])
    init = PyPIService.__init__

    def init_validated(self: PyPIService, *args: object, **kwargs: object) -> None:
        init(self, *args, **kwargs)
        self.session.hooks["response"].append(validating_hook(record))

    PyPIService.__init__ = init_validated  # type: ignore[method-assign]
    sys.argv = ["pip-audit", *sys.argv[2:]]
    audit()


if __name__ == "__main__":
    main()
