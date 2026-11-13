# chriek

Self-hostable image generation server for FLUX 3 Image Klein, with an asynchronous job API: submit a job, poll it, download the PNGs through signed, expiring URLs. Architecture: [ADR-0001](docs/adr/0001-mvp-architecture.md).

> **Status: walking skeleton.** Only **home mode** with a **fake image backend** exists. The fake backend draws deterministic placeholder PNGs (two coloured rectangles derived from prompt, size and seed). It does **not** run FLUX or any other model, needs no GPU and downloads no weights. The real MLX (Apple Silicon) and CUDA backends come later; the HTTP contract below stays the same.

## Install and run (uv)

Requires [uv](https://docs.astral.sh/uv/); uv installs Python 3.14 itself if needed.

```sh
git clone https://github.com/stibbons-ai/chriek.git
cd chriek
uv sync --locked

export CHRIEK_API_KEYS="fabula:$(openssl rand -hex 32),ikonograf:$(openssl rand -hex 32)"
export CHRIEK_SIGNING_SECRET="$(openssl rand -hex 32)"
uv run chriek serve
```

Generate every API key and the signing secret independently from a cryptographically secure random source, as above. chriek only checks lengths, characters and that no value is reused; it can't tell a guessable value from a random one.

chriek runs only on POSIX hosts and filesystems (Linux, macOS) and refuses to start when a directory above its data is group- or other-writable, such as a `0775` checkout under umask `002`; see [Storage permissions](#storage-permissions).

`chriek serve` runs the API and the in-process worker in one process (home mode). It listens on `127.0.0.1:8000` by default. A setting rejected by configuration validation (see the ranges and formats in [Configuration](#configuration)) stops it before it opens the database or a port, with exit code 2 and a message that names the variable but never prints secret values. Values that aren't validated, such as an unresolvable `CHRIEK_HOST`, fail later, when the server binds.

Logs go to standard output/error: startup, shutdown and errors. There is no HTTP access log, because download URLs carry their signature, which works like a password until it expires; put any access logging you need in a reverse proxy and strip the query string there.

chriek has no TLS of its own. Expose it only on a private network or behind a TLS-terminating reverse proxy: API keys and download signatures travel in requests.

### Backend dependencies

Today's supported installation remains `uv sync --locked`: only the fake backend is available, with no GPU runtimes or model weights.

Shared runtime dependencies stay in `pyproject.toml` under `[project].dependencies`; development tools stay in `[dependency-groups].dev`. Real backend implementations will add `mlx` and `cuda` extras under `[project.optional-dependencies]`, containing their actual inference dependencies, not just a tensor library.

The future commands `uv sync --locked --extra mlx` and `uv sync --locked --extra cuda` are **not available today**: neither extra exists yet. Extras install dependencies; they neither detect hardware nor select a runtime backend. CUDA wheel source and version configuration belongs with the real CUDA implementation; naming an extra `cuda` does not select a CUDA build.

Keep the committed `uv.lock` as the single lockfile; do not introduce hand-maintained `requirements_mlx.txt` or `requirements_cuda.txt` copies.

## Configuration

All configuration comes from environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `CHRIEK_API_KEYS` | required | Named static API keys, `name:key,name:key`. The name is the owner of the jobs created with that key. Names: 1–64 characters from `A-Z a-z 0-9 _ . -`, unique. Names are not secret: each job stores its owner's name in plain text, so never use a key or the signing secret as a name. Keys: at least 32 printable ASCII characters (`!` to `~`; no spaces, control or non-ASCII characters, no commas), unique. Keys travel in an HTTP header, so other characters are rejected at startup. Startup errors identify an invalid entry by its position, never by its name or key. |
| `CHRIEK_SIGNING_SECRET` | required | HMAC secret for download URLs, at least 32 characters. Changing it invalidates every issued URL (polling the job again mints new ones). Anyone who knows it can sign download URLs, so it must not be one of the API keys: startup rejects a signing secret equal to any key or name in `CHRIEK_API_KEYS`. |
| `CHRIEK_URL_TTL_SECONDS` | `900` | Lifetime of a download URL, 1–604800 seconds. |
| `CHRIEK_DB_PATH` | `chriek-data/chriek.db` | SQLite database file, used literally (`?` and `%20` are ordinary filename characters); missing parent directories are created private. Its parent must be a dedicated directory; see [Storage permissions](#storage-permissions). Put it on a volume to keep jobs across restarts. The exact value `:memory:` (or `./:memory:`) selects an ephemeral in-memory database instead; see [In-memory database](#in-memory-database). |
| `CHRIEK_STORAGE_DIR` | `chriek-data/images` | Dedicated directory for generated PNGs, one subdirectory per job; see [Storage permissions](#storage-permissions). |
| `CHRIEK_HOST` | `127.0.0.1` | Listen address. |
| `CHRIEK_PORT` | `8000` | Listen port. |
| `CHRIEK_MAX_IMAGES` | `4` | Maximum `image_count` per job (configurable 1–64). |
| `CHRIEK_MAX_DIMENSION` | `2048` | Maximum `width` and `height`; a multiple of 16 (16–8192). |
| `CHRIEK_MAX_PIXELS` | `1048576` | Maximum `width * height` (256–67108864). |
| `CHRIEK_MAX_PROMPT_CHARS` | `2000` | Maximum prompt length in characters (1–100000). |

Relative paths resolve against the working directory. The database schema is created on first start; there are no migrations yet.

### In-memory database

`CHRIEK_DB_PATH=:memory:` keeps the job database in memory, meant for tests and local development. No database file is written. The database lives exactly as long as one `chriek serve` process: submissions, the worker and polling share it while the process runs, and every restart starts empty, so job IDs from an earlier run return `404` and their download URLs stop working. The restart guarantees in [Job behaviour](#job-behaviour) (queued jobs survive, running jobs fail as interrupted) hold only for a database file. PNGs still go to `CHRIEK_STORAGE_DIR` on disk and aren't deleted on restart.

### Storage permissions

Jobs and images are private to the user chriek runs as (the service user) on POSIX filesystems. The database's directory (the parent of `CHRIEK_DB_PATH`) and `CHRIEK_STORAGE_DIR` must be dedicated to chriek and owned by the service user; the defaults (`chriek-data/` and `chriek-data/images/`) are. chriek creates missing directories with mode `0700`, the database with `0600` (SQLite gives its `-wal` and `-shm` sidecars the database's mode, also when it recreates them) and every PNG with `0600` from its first byte, including while a job is still rendering. Nothing chriek creates is ever open to group or others, whatever the process umask is, for `chriek serve` and for code that embeds chriek's app, store or worker. Working additionally needs a umask that leaves the owner read, write and search rights (such as `022` or `077`): a umask like `0277` makes a first start fail and, on an existing installation, makes every job fail.

Before the database is opened, any job runs or any missing directory is created, chriek checks both configured storage sides and refuses to start if any check fails (`chriek serve` prints `chriek: storage error: <path> <problem>. <guidance>` and exits with status 2 without listening):

- The database's directory and `CHRIEK_STORAGE_DIR`, if they exist: a directory, not a symbolic link, owned by the service user (effective UID), no group or other permission bits (`mode & 077 == 0`).
- The database file and its `-wal`, `-shm` and `-journal` sidecars, if they exist: a regular file, not a symbolic link, owned by the service user, no group or other permission bits. Other files in the database's directory are not checked.
- Every directory and symbolic link followed to reach those two directories, including link targets and chains of links: each link owned by root or the service user; each directory owned by root or the service user and not writable by group or others unless the sticky bit is set (as on `/tmp`). So a link inside a directory others can modify without the sticky bit is refused with that directory, because anyone could re-point it; in a sticky directory only the link's owner can, so a link there must be owned by root or the service user. Links such as macOS's `/var` are resolved once at startup and chriek uses the real path from then on: re-pointing a link while chriek runs doesn't move its storage before a restart.
- Every file and directory below `CHRIEK_STORAGE_DIR`: not a symbolic link, owned by the service user, no group or other permission bits. A directory chriek can't list fails the check. This includes files chriek didn't write, such as a `.DS_Store`: with chriek stopped, delete that file or remove its group and other permissions (`chmod go= <file>`).

A path chriek can't check (for example below a regular file, inside a directory it can't search, through a symbolic link loop, or through a link whose target continues with `..` after a missing directory) is reported the same way. If a missing directory appears while chriek is creating it, chriek refuses it rather than using a directory it didn't create. The checks cover what already exists, not every later failure: creating the missing parts or opening the database can still fail, for example in a directory chriek can't write, with a file name too long for the filesystem, or with a database SQLite can't open or write. Such an error is reported as a storage error or, for some database errors, as a traceback; nothing is rolled back, so private directories and an empty private database file chriek created so far stay. Remove them or fix the configuration and restart.

chriek creates `CHRIEK_STORAGE_DIR` only at startup. If it is deleted while chriek runs, every later job fails (`image 1 of N failed`) until a restart creates it again.

chriek never changes permissions or ownership of anything that already exists: not your data, not the directories above it. A bare filename such as `CHRIEK_DB_PATH=chriek.db` puts the database in the working directory, which is usually shared (`0755`) and is refused; point it into a dedicated directory such as `chriek-data/chriek.db` instead. With `CHRIEK_DB_PATH=:memory:` there is no database file, and `CHRIEK_STORAGE_DIR` is checked and protected the same way.

A group-writable ancestor is refused even when the group is private to you: mode bits can't tell a private group from a shared one. Under umask `002` (the default for user-private groups on some Linux distributions) your own project directory is `0775`, so chriek refuses the default `chriek-data/` inside it. Configure dedicated paths under an ancestor only you or root can write, such as a directory in your home created with `mkdir -m 700`, or remove group write from your own directory (`chmod g-w <dir>`) if no one else relies on it.

**Migrating an existing installation.** Earlier versions created directories `0755` and files `0644`, so chriek refuses to start on their data. Stop chriek, then as the service user make only chriek's dedicated directories private and restart:

```sh
# chriek stopped; run as the service user. Only chriek's own directory, never a shared parent.
chmod -R go= chriek-data
```

If the database sits in a shared directory, stop every process that uses it, then move it together with each of its sidecars that exists into a new dedicated directory and set `CHRIEK_DB_PATH` to the new location. A `-wal` file left by an unclean stop holds jobs not yet in `chriek.db`, so it must move with it:

```sh
# Every chriek process stopped; run as the service user in the shared directory.
mkdir -m 700 chriek-db
for file in chriek.db chriek.db-wal chriek.db-shm chriek.db-journal; do
  if [ -e "$file" ]; then mv "$file" chriek-db/ && chmod go= "chriek-db/$file"; fi
done
```

The names are exact on purpose: a pattern such as `chriek.db*` would also move unrelated files like backups.

A configured path that is a symbolic link is refused, also when its target doesn't exist yet; configure its real target instead. Data owned by another user must be copied or `chown`ed by an administrator before chriek will use it.

chriek removes `..` from `CHRIEK_DB_PATH` and `CHRIEK_STORAGE_DIR` before following links in them, so `alias/../images` names `images` beside `alias`, while earlier versions followed `alias` first and used `images` beside its target; the database file name itself is still used literally. Before upgrading with such a path, set `CHRIEK_DB_PATH` to the real path of the existing database file and `CHRIEK_STORAGE_DIR` to the real directory the existing images are in, so they stay where they are; otherwise chriek starts on a new, empty database.

**Limits.** Mode bits keep other local users out; they don't protect against root, against other processes running as the same user, or against access granted separately through ACLs (chriek neither inspects nor manages ACLs, and mode bits don't revoke them). Checks run at startup; chriek doesn't watch for permission changes made while it runs. Only POSIX filesystems with real ownership and modes are supported: on FAT/exFAT, some network or FUSE mounts, or other filesystems that synthesize modes, the checks can pass without protecting anything.

## API

Every `/jobs` request needs `Authorization: Bearer <key>`. A missing or invalid key returns `401` before the request body is read, so even a malformed body gets `401`. A job created with another key returns `404`, exactly like a job that doesn't exist. `GET /health` and image downloads need no key.

### `POST /jobs`

```sh
curl -s -X POST http://127.0.0.1:8000/jobs \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "a lighthouse at dusk", "image_count": 2, "seed": 41, "width": 1024, "height": 768}'
```

| Field | Type | Default | Rule |
|---|---|---|---|
| `prompt` | string | required | not blank, at most `CHRIEK_MAX_PROMPT_CHARS` |
| `image_count` | integer | `1` | 1 to `CHRIEK_MAX_IMAGES` |
| `seed` | integer or null | random | `0 ≤ seed` and `seed + image_count - 1 ≤ 4294967295` |
| `width`, `height` | integer | `1024` | multiples of 16, 16 to `CHRIEK_MAX_DIMENSION`, `width * height ≤ CHRIEK_MAX_PIXELS` |

Types are strict (`"1024"` or `1024.0` is not an integer) and unknown fields are rejected. Nonfinite numbers (`NaN`, `Infinity`, `1e999`) and prompts with unpaired UTF-16 surrogates (`"\ud800"`) are rejected too. Requests that fail validation, including malformed JSON syntax such as `{`, return `422` with a `{"detail": [{"type": ..., "loc": [...], "msg": ...}]}` error list. Other JSON decoding failures, such as a byte `0xFF` inside an otherwise UTF-8 JSON string or an integer literal thousands of digits long, return `400 {"detail": "There was an error parsing the body"}` instead; chriek JSON-decodes only bodies sent as `application/json` or `application/*+json`. The JSON decoder also accepts UTF-16 and UTF-32; the request byte cap below applies to those encodings too. A body without a `Content-Type` or of any other type, such as `text/plain` (even invalid bytes), is not decoded and fails validation with `422`. Neither kind of rejected request enters the queue. A valid request returns `202`, a `Location: /jobs/<id>` header and the job (below) in status `queued`. Job IDs are UUIDv7. When `seed` is omitted, chriek picks one at random and reports it in `params.seed`; image `i` uses seed `seed + i`.

Two fixed byte budgets apply; neither is configurable:

- **Request body: at most 2,097,152 bytes (2 MiB).** After the key check, both authenticated routes (`POST /jobs` and `GET /jobs/{id}`) reject a `Content-Length` above the limit with `413 {"detail": "request body must be at most 2097152 bytes"}` before any body is read. When `POST /jobs` reads a body, chriek also counts the bytes it actually receives from the server, for example a chunked (`Transfer-Encoding: chunked`) body, and answers `413` at the first chunk that crosses the limit, without reading on. `GET /jobs/{id}` does not read a body. HTTP framing is the server's job: with a declared `Content-Length`, the server hands chriek at most that many bytes, and anything sent after them is not part of the request. A `POST /jobs` body of exactly 2 MiB is validated normally. The budget is a byte limit, independent of `CHRIEK_MAX_PROMPT_CHARS` (which counts characters): a UTF-8 JSON body with a prompt at the highest allowed setting, 100,000 characters, fits even when every character is written as `\uXXXX` escapes. Multipart uploads ([#5](https://github.com/stibbons-ai/chriek/issues/5)) will need their own size policy.
- **`422` response: at most 16,384 bytes (16 KiB).** Error entries carry only `type`, `loc` and `msg`; the rejected input and the validation error's `ctx` are never echoed. Each text value, including field names you sent that show up in `loc`, is first clipped to 200 characters; unpaired surrogates in it are then written as escape text such as `\ud800`, which may make it longer. Locations retain at most eight parts. Errors that no longer fit are dropped from the end, so the list may be incomplete.

An early rejection can leave the request body unread. Clients that finish uploading before reading the response may see a broken pipe or connection reset instead of the `401` or `413` response.

Once the worker has [stopped on an unsaved result](#job-behaviour), a request that passes authentication, the body-size limit and validation returns `503 {"detail": "worker stopped: a job result could not be saved; restart chriek"}` instead of `202`: no `Location` header, no job stored, nothing run. Authentication (`401`), the body-size limit (`413`) and validation (`422`, `400`) are checked first and keep their answers. The check sees the worker state at the time of the request, so a job accepted just before a result fails to save stays `queued` until restart. chriek has no queue cap; if one is added, this `503` takes precedence over a full-queue answer.

### `GET /jobs/{id}`

```sh
curl -s http://127.0.0.1:8000/jobs/$JOB_ID -H "Authorization: Bearer $KEY"
```

```json
{
  "id": "019a7c1e-4b2a-7c3d-9e8f-0123456789ab",
  "status": "succeeded",
  "params": {"prompt": "a lighthouse at dusk", "image_count": 2, "seed": 41, "width": 1024, "height": 768},
  "created_at": "2026-11-11T12:00:00.000000+00:00",
  "started_at": "2026-11-11T12:00:00.010000+00:00",
  "finished_at": "2026-11-11T12:00:00.090000+00:00",
  "error": null,
  "run": {"backend": "fake", "model": "chriek-fake-backend", "revision": "fake-v1", "render_seconds": 0.08},
  "images": [
    {"index": 0, "seed": 41, "url": "http://127.0.0.1:8000/images/019a.../0.png?expires=1793887200&sig=...", "expires_at": "2026-11-11T12:15:00+00:00"},
    {"index": 1, "seed": 42, "url": "http://127.0.0.1:8000/images/019a.../1.png?expires=1793887200&sig=...", "expires_at": "2026-11-11T12:15:00+00:00"}
  ]
}
```

- `status` is `queued`, `running`, `succeeded` or `failed`.
- `run` is the run record, set once the worker finishes the job: which backend, model and revision rendered it and how long rendering took. The fake backend reports no hardware or memory figures because it measures none. A job failed by restart recovery (see [Job behaviour](#job-behaviour)) has `run: null`, because no run record was persisted.
- `images` is filled only when `succeeded`. Every poll mints fresh URLs that expire `CHRIEK_URL_TTL_SECONDS` later.
- `error` is `null` unless the job `failed`. A failed image is reported as exactly `image N of M failed` (N is the one-based image that failed, M the requested `image_count`); a job failed by restart recovery carries the interrupted message below. The error never includes exception text, classes or other backend details: the server log keeps the full traceback with the job ID and the zero-based image index.

### `GET /images/{id}/{index}.png?expires=…&sig=…`

Downloads one PNG without any API key: the HMAC signature covers exactly that job, image index and expiry. A wrong or missing signature returns `403 invalid signature`; an expired URL returns `403 link expired; poll the job again`. URLs are built from the host the client used to reach chriek.

```sh
curl -s -o image0.png "$URL"
```

### `GET /health`

Returns `200 {"status": "ok"}`. If the worker could not save a job's result (succeeded or failed) to the database, it stops and this returns `503 {"status": "error", "detail": "worker stopped: a job result could not be saved; restart chriek"}`; see [Job behaviour](#job-behaviour).

## Job behaviour

- **Queue.** Jobs are stored in SQLite and run one at a time, oldest first, by a single worker thread in the `chriek serve` process. There is one global queue across all owners: no round-robin or per-owner fairness. If owner A submits A1 and A2 before owner B submits B1, both A jobs run before B1; an A3 submitted after B1 runs after B1, even if A1 and A2 finish first. An admitted job is delayed only by jobs submitted before it, never by later ones, but chriek promises no maximum wait: a long or stalled earlier job holds up everything behind it.
- **No admission cap.** Nothing limits how many jobs one owner can queue. A per-owner queue cap is proposed ([#8](https://github.com/stibbons-ai/chriek/issues/8), [PR #13](https://github.com/stibbons-ai/chriek/pull/13), not merged). Even then, a count cap would not bound how long jobs run, would not guarantee another owner's admission or latency, and is no SLA.
- **All or nothing.** If any image of a job fails, the job fails and the images it already rendered are discarded. A job is never partly successful.
- **Restarts.** With a database file, queued jobs survive a restart and run afterwards. A job that was `running` when the process stopped is marked `failed` on the next start (`interrupted: the server stopped while this job was running`) and isn't retried. An [in-memory database](#in-memory-database) starts empty instead.
- **Unsaved results stop the worker.** If saving a finished job's result to the database fails, the worker logs the error and stops instead of moving on: that job stays `running`, no later job runs, jobs accepted before the failure stay `queued`, new submissions are rejected with `503` (see [`POST /jobs`](#post-jobs)), and `GET /health` returns `503`. Fix the database or disk problem and restart `chriek serve`; restart recovery then fails the stuck job as interrupted, the queued jobs run, and submissions are accepted again. Rejected submissions were never stored, so they don't run after the restart.
- **AI disclosure in every PNG**, always on: PNG text chunks `AI-Generated: true`, `Model`, `Model-Revision`, `Seed` and `Description`, plus XMP (`XML:com.adobe.xmp`) with IPTC `Iptc4xmpExt:DigitalSourceType` = `http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia`. With the fake backend, `Model` is `chriek-fake-backend` and `Model-Revision` is `fake-v1`.
- No retention: generated files stay in `CHRIEK_STORAGE_DIR` until you delete them.

## Operator obligations under the FLUX Kommunity License

This section matters once a real FLUX backend runs; the fake backend uses no FLUX model. chriek is MIT-licensed, but the FLUX 3 Image Klein weights are licensed by Black Forest Labs under the **FLUX Kommunity License v1.0**. chriek never ships or redistributes the weights: you download them with your own Hugging Face account and accept that license yourself. Read the [license text](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/blob/main/LICENSE.md) (wording below checked 2026-11-11); this summary is not legal advice.

- **Revenue cap.** Commercial use of outputs, and running the model in production to generate them, is allowed only for a "Qualifying User": you together with your affiliates have gross annualized revenue below US$5,000,000 (§1.e, §2.a). Otherwise the license covers only non-commercial purposes, and commercial use needs a separate license from Black Forest Labs (§2.b).
- **Content filtering.** You must implement and maintain content filters that prevent unlawful, unsafe or infringing content, or review outputs for unlawful or infringing content before distributing or displaying them (§2.e(i)). chriek has no built-in filter; this is the operator's job.
- **AI disclosure.** Outputs must carry disclosures that they were generated or modified with AI to the extent applicable law requires (§2.e(ii)). chriek writes the PNG text chunk and IPTC XMP marker above into every image; keep them intact and add any disclosure your distribution channel needs.
- **Biometric clause.** The license prohibits using the model or its outputs for "biometric processing", surveillance and military purposes, among other restrictions, and requires compliance with laws governing biometric information (§4.a). Applications that work on photos of real people (for example profile pictures from reference photos) need a license review before going live.

## Development

Install development dependencies and activate the commit hook once per checkout (uv must be on PATH when committing):

```sh
uv sync --locked
uv run --locked pre-commit install
```

The hook scans all Python production code under `src/` on every commit, including documentation-only commits. CI runs the same gate on every PR and pushes to `main`. Zero qualifying clones are allowed, including existing clones; there is no baseline or percentage allowance. Tests and tooling outside `src/` are excluded.

`.jscpd.json` requires both 10 lines and 70 tokens per clone. Weak mode ignores comments and whitespace, but Python docstrings are strings and may still match. Identifiers and literals remain significant; this is not fuzzy or semantic detection. Missing roots or an empty corpus fail the check.

Run the hook manually:

```sh
uv run --locked pre-commit run jscpd --all-files
```

Local hooks can be bypassed; CI remains authoritative. Requiring the CI check before merging is a separate branch-protection setting.

The full check, also run by CI (`.github/workflows/ci.yml`):

```sh
uv sync --locked && uv run --locked pre-commit run jscpd --all-files && uv run ruff check . && uv run ruff format --check . && uv run pytest && uv run --locked python scripts/audit_dependencies.py
```

Application tests drive chriek through its HTTP API and the installed `chriek` command, with the fake backend, plus direct store, worker and app entry-point checks where storage privacy must hold without the CLI. `tests/test_dependency_audit.py` checks the dependency-audit gate's boundaries offline: injected pip-audit reports, and the real pinned pip-audit with only its HTTP transport answering canned PyPI responses.

### Dependency audit

CI runs the same command on every PR and push to `main`:

```sh
uv run --locked python scripts/audit_dependencies.py
```

It audits every third-party package locked in `uv.lock` against PyPI's vulnerability service with `pip-audit`, pinned in the `dev` group (`pip-audit==2.10.1`, its own dependencies locked too):

1. `uv export --locked --all-groups --all-extras --no-emit-project` lists the exact locked versions. A lockfile that is stale against `pyproject.toml` fails here; nothing is resolved anew.
2. The export must match the registry packages in `uv.lock` exactly. Only the local project at `.` (editable or virtual) is excluded. Registry packages must come from `https://pypi.org/simple`: the PyPI service audits by name and version, so a package from any other index fails the gate rather than being audited as its PyPI namesake. Git, URL or path dependencies fail for the same reason. A package locked at two versions (for example per platform) also fails: the gate's inventory and the requirements file it hands `pip-audit` hold one version per package.
3. Environment markers are dropped before auditing, so platform-specific packages (for example `colorama` on Windows) are audited on every machine; `pip-audit` would otherwise skip them silently.
4. `pip-audit --no-deps --disable-pip --strict -s pypi -f json` runs with a fresh, empty HTTP cache, so no stale advisory data is reused. The gate then reads its JSON report: every package from step 1 must appear, once, at its locked version, with none skipped.
5. `pip-audit` 2.10.1 parses PyPI's responses lossily: a missing or `null` `vulnerabilities` value, or an empty non-list one (`{}`, `""`), silently counts as "no vulnerabilities", and an advisory whose `withdrawn` is not `null` is silently skipped while the response's other advisories still count. So the gate starts `pip-audit` through `scripts/pip_audit_checked.py`, which adds a response hook to `pip-audit`'s own PyPI session: every HTTP 200 body it is about to parse must be JSON with a `vulnerabilities` list whose entries have a string `id`, string lists `aliases` and `fixed_in`, a `withdrawn` that is `null` or an RFC 3339 date-time on a real calendar date (the format PyPI's JSON API specifies; leap seconds, second `60`, are unsupported and fail as operational errors, never as a skipped advisory), and string-or-null `summary`/`details`. Anything else aborts `pip-audit` before it writes a report. The hook records each validated `name==version`; the gate requires exactly one record per locked package, so a hook that stops firing (for example after a `pip-audit` upgrade changes its internals) fails instead of passing. The hook relies on `pip-audit` internals (`PyPIService`, its `session`); upgrading `pip-audit` means re-checking that parser and the offline tests.

| Exit | Meaning |
|---|---|
| 0 | `DEPENDENCY AUDIT PASSED`: every locked package audited, no known advisory |
| 1 | `DEPENDENCY AUDIT FAILED`: each `VULNERABLE name==version: ID aliases [...] fixed in [...]` |
| 2 | `DEPENDENCY AUDIT ERROR (operational, nothing proven)`: stale lock, network or service failure, malformed PyPI response, skipped or missing packages, malformed report, or a pip-audit exit code that contradicts its report. When report evaluation succeeds but subsequent PyPI-response validation fails, the report's advisories follow as `UNCONFIRMED name==version: ID aliases [...] fixed in [...]`: worth a look, but not proven by validated responses, so the run stays exit 2, never 1 |

Only exit 0 passes. There is no ignore list, severity threshold or fallback; a vulnerability without a compatible fix blocks the build until a person decides. `pip-audit` exits 1 both for findings and for its own errors, so the gate never reads its exit code alone.

Coverage limits: the gate covers the application dependencies, the `dev` group (including `pip-audit`'s own dependencies) and any future optional extras or groups, on all platforms in the universal lock. It does not cover the Python interpreter, `uv`, system libraries, anything a locked package downloads or bundles itself, packages outside `uv.lock` (such as GPU backends installed separately), or vulnerabilities with no published PyPI advisory. A pass means no known advisory on PyPI at that moment, not that the dependencies are secure.

Fixture projects in `tests/fixtures/dependency_audit/` demonstrate both outcomes against the real service without touching this project's lock (`--project` points the gate at another uv project):

```sh
uv run --locked python scripts/audit_dependencies.py --project tests/fixtures/dependency_audit/clean       # exit 0
uv run --locked python scripts/audit_dependencies.py --project tests/fixtures/dependency_audit/vulnerable  # exit 1, urllib3 1.26.4
```

Recorded run, 2026-11-12, Python 3.14.6 on macOS 27.2.1 arm64, pip-audit 2.10.1, PyPI service, with PyPI response validation: this lock, 61 third-party packages, no known vulnerabilities (exit 0). Same day: clean fixture exit 0; vulnerable fixture exit 1 with 11 advisories for `urllib3==1.26.4` (PYSEC-2021-108, PYSEC-2023-192, PYSEC-2023-212, PYSEC-2026-141, PYSEC-2026-1994, PYSEC-2026-1995, PYSEC-2026-1996, PYSEC-2026-1998, PYSEC-2026-1999, PYSEC-2026-4175, PYSEC-2026-4177); unreachable service (`HTTPS_PROXY=http://127.0.0.1:9`) exit 2; stale lock exit 2. Malformed PyPI responses are demonstrated offline only (`tests/test_dependency_audit.py`); real PyPI was not observed serving one.
