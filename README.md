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

`chriek serve` runs the API and the in-process worker in one process (home mode). It listens on `127.0.0.1:8000` by default. A setting rejected by configuration validation (see the ranges and formats in [Configuration](#configuration)) stops it before it opens the database or a port, with exit code 2 and a message that names the variable but never prints secret values. Values that aren't validated, such as an unresolvable `CHRIEK_HOST`, fail later, when the server binds.

Logs go to standard output/error: startup, shutdown and errors. There is no HTTP access log, because download URLs carry their signature, which works like a password until it expires; put any access logging you need in a reverse proxy and strip the query string there.

> **Warning: without TLS, use chriek only on a private network or behind a TLS-terminating reverse proxy.** API keys and download signatures travel in every request; over plain HTTP anyone on the path can read and reuse them. Either let a reverse proxy terminate TLS, or turn on [native TLS](#native-tls).

### Native TLS

chriek can serve HTTPS itself, without a proxy. Point it at a PEM certificate (chain) and its unencrypted PEM private key:

```sh
export CHRIEK_TLS_CERTFILE=/etc/chriek/fullchain.pem
export CHRIEK_TLS_KEYFILE=/etc/chriek/privkey.pem
uv run chriek serve   # logs "Uvicorn running on https://127.0.0.1:8000"
```

Set both variables or neither. With TLS configured, chriek never falls back to plain HTTP. Configuration validation loads the pair once: if only one variable is set, either is set but empty, or the files can't be loaded as a matching certificate and key (missing, unreadable, not PEM, mismatched, or an encrypted key), startup stops with exit code 2. The server then loads the files again when it starts; if they changed in between, that second load fails closed: no listener opens and plain HTTP is never served, but the outcome is the server's own load error rather than chriek's exit code 2 (a key swapped for an encrypted one, for example, makes OpenSSL ask for a pass phrase on a terminal). A plain-HTTP request to a TLS port gets no HTTP answer. chriek doesn't obtain or renew certificates (no ACME): renew them yourself and restart `chriek serve` to load the new files. Download URLs are built from the request's scheme and `Host` header, so for clients connecting directly over TLS they are `https://` URLs. The server also honours `X-Forwarded-Proto`/`X-Forwarded-For` from trusted clients, which then decide the scheme of those URLs: by default only `127.0.0.1` and `::1` are trusted, so any process on the same host (such as a local proxy) counts, and uvicorn's `FORWARDED_ALLOW_IPS` environment variable replaces that list.

### Behind a reverse proxy (`CHRIEK_AUTH_MODE=proxy`)

In `proxy` mode chriek does no authentication of its own: the reverse proxy in front of it must authenticate every client before forwarding a request. chriek ignores `Authorization` and identity headers such as `X-Forwarded-User`; it never derives an owner from headers. Every request belongs to one fixed, shared owner, so **all clients share all jobs**: any client can poll any job, and all of them share one queue capacity (`CHRIEK_MAX_QUEUED_JOBS`). Run separate chriek instances, or use `api-key` mode, when clients must not see each other's jobs.

Anyone who reaches chriek's port directly bypasses the proxy and gets full access. Bind chriek where only the proxy can reach it: the default `CHRIEK_HOST=127.0.0.1` with the proxy on the same host, or a private network. `CHRIEK_API_KEYS` must be unset or empty in proxy mode (startup stops otherwise, so nobody relies on keys that are never checked); `CHRIEK_SIGNING_SECRET` is still required, and downloads still need valid signed URLs. Polling is separate per mode: jobs created in one mode can't be polled in the other, even on the same database. Signed download URLs are bearer links and don't depend on the mode: one issued before a mode change keeps working until it expires, as long as the storage and `CHRIEK_SIGNING_SECRET` stay the same.

```sh
export CHRIEK_AUTH_MODE=proxy
unset CHRIEK_API_KEYS
export CHRIEK_SIGNING_SECRET="$(openssl rand -hex 32)"
uv run chriek serve
```

## Configuration

All configuration comes from environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `CHRIEK_AUTH_MODE` | `api-key` | `api-key`: every `/jobs` request needs a key from `CHRIEK_API_KEYS`, and each key owns its jobs. `proxy`: no key check; a trusted reverse proxy authenticates, and all clients are one shared owner; see [Behind a reverse proxy](#behind-a-reverse-proxy-chriek_auth_modeproxy). |
| `CHRIEK_API_KEYS` | required in `api-key` mode | Named static API keys, `name:key,name:key`. The name is the owner of the jobs created with that key. Names: 1–64 characters from `A-Z a-z 0-9 _ . -`, unique. Keys: at least 32 printable ASCII characters (`!` to `~`; no spaces, control or non-ASCII characters, no commas), unique. Keys travel in an HTTP header, so other characters are rejected at startup. Must be unset or empty in `proxy` mode. |
| `CHRIEK_SIGNING_SECRET` | required | HMAC secret for download URLs, at least 32 characters. Changing it invalidates every issued URL (polling the job again mints new ones). |
| `CHRIEK_URL_TTL_SECONDS` | `900` | Lifetime of a download URL, 1–604800 seconds. |
| `CHRIEK_DB_PATH` | `chriek-data/chriek.db` | SQLite database file, used literally (`?` and `%20` are ordinary filename characters); parent directories are created. Put it on a volume to keep jobs across restarts. The exact value `:memory:` (or `./:memory:`) selects an ephemeral in-memory database instead; see [In-memory database](#in-memory-database). |
| `CHRIEK_STORAGE_DIR` | `chriek-data/images` | Directory for generated PNGs, one subdirectory per job. |
| `CHRIEK_HOST` | `127.0.0.1` | Listen address. |
| `CHRIEK_PORT` | `8000` | Listen port. |
| `CHRIEK_TLS_CERTFILE` | unset | PEM certificate (chain) for [native TLS](#native-tls). Set together with `CHRIEK_TLS_KEYFILE`; leave both unset for plain HTTP. |
| `CHRIEK_TLS_KEYFILE` | unset | Unencrypted PEM private key matching `CHRIEK_TLS_CERTFILE`. |
| `CHRIEK_MAX_QUEUED_JOBS` | `100` | Maximum `queued` jobs per owner (1–100000); see [Queue limit](#queue-limit). |
| `CHRIEK_MAX_IMAGES` | `4` | Maximum `image_count` per job (configurable 1–64). |
| `CHRIEK_MAX_DIMENSION` | `2048` | Maximum `width` and `height`; a multiple of 16 (16–8192). |
| `CHRIEK_MAX_PIXELS` | `1048576` | Maximum `width * height` (256–67108864). |
| `CHRIEK_MAX_PROMPT_CHARS` | `2000` | Maximum prompt length in characters (1–100000). |

Relative paths resolve against the working directory. The database schema is created on first start; there are no migrations yet.

### In-memory database

`CHRIEK_DB_PATH=:memory:` keeps the job database in memory, meant for tests and local development. No database file is written. The database lives exactly as long as one `chriek serve` process: submissions, the worker and polling share it while the process runs, and every restart starts empty, so job IDs from an earlier run return `404` and their download URLs stop working. The restart guarantees in [Job behaviour](#job-behaviour) (queued jobs survive, running jobs fail as interrupted) hold only for a database file. PNGs still go to `CHRIEK_STORAGE_DIR` on disk and aren't deleted on restart.

## API

In `api-key` mode (the default) every `/jobs` request needs `Authorization: Bearer <key>`. A missing or invalid key returns `401` before the request body is read, so even a malformed body gets `401`. A job created with another key returns `404`, exactly like a job that doesn't exist. In `proxy` mode `/jobs` needs no key and never returns `401`; every request acts as the one shared owner ([Behind a reverse proxy](#behind-a-reverse-proxy-chriek_auth_modeproxy)). `GET /health` and image downloads need no key in either mode.

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

Types are strict (`"1024"` or `1024.0` is not an integer) and unknown fields are rejected. Nonfinite numbers (`NaN`, `Infinity`, `1e999`) and prompts with unpaired UTF-16 surrogates (`"\ud800"`) are rejected too; error entries echo such input as text (`"inf"`, `"\\ud800"`). Requests that fail validation, including malformed JSON syntax such as `{`, return `422` with FastAPI's `{"detail": [...]}` error list. Other body decoding failures, such as invalid UTF-8 or an integer literal thousands of digits long, return `400 {"detail": "There was an error parsing the body"}` instead. Neither kind of rejected request enters the queue. A valid request returns `202`, a `Location: /jobs/<id>` header and the job (below) in status `queued`. Job IDs are UUIDv7. When `seed` is omitted, chriek picks one at random and reports it in `params.seed`; image `i` uses seed `seed + i`.

A valid request whose owner already has `CHRIEK_MAX_QUEUED_JOBS` queued jobs returns `429` instead and is not queued; see [Queue limit](#queue-limit). Validation comes first, so an invalid request gets `422` or `400` even when the queue is full.

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
- `error` explains a `failed` job, for example `image 2 of 3 failed: …`.

### `GET /images/{id}/{index}.png?expires=…&sig=…`

Downloads one PNG without any API key: the HMAC signature covers exactly that job, image index and expiry. A wrong or missing signature returns `403 invalid signature`; an expired URL returns `403 link expired; poll the job again`. URLs are built from the host the client used to reach chriek.

```sh
curl -s -o image0.png "$URL"
```

### `GET /health`

Returns `200 {"status": "ok"}`. If the worker could not save a job's result (succeeded or failed) to the database, it stops and this returns `503 {"status": "error", "detail": "worker stopped: a job result could not be saved; restart chriek"}`; see [Job behaviour](#job-behaviour).

## Job behaviour

- **Queue.** Jobs are stored in SQLite and run one at a time, oldest first, by a single worker thread in the `chriek serve` process.
- **All or nothing.** If any image of a job fails, the job fails and the images it already rendered are discarded. A job is never partly successful.
- **Restarts.** With a database file, queued jobs survive a restart and run afterwards. A job that was `running` when the process stopped is marked `failed` on the next start (`interrupted: the server stopped while this job was running`) and isn't retried. An [in-memory database](#in-memory-database) starts empty instead.
- **Unsaved results stop the worker.** If saving a finished job's result to the database fails, the worker logs the error and stops instead of moving on: that job stays `running`, no later job runs, new submissions stay `queued` until the [queue limit](#queue-limit) answers `429`, and `GET /health` returns `503`. Fix the database or disk problem and restart `chriek serve`; restart recovery then fails the stuck job as interrupted and the queue resumes.
- **AI disclosure in every PNG**, always on: PNG text chunks `AI-Generated: true`, `Model`, `Model-Revision`, `Seed` and `Description`, plus XMP (`XML:com.adobe.xmp`) with IPTC `Iptc4xmpExt:DigitalSourceType` = `http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia`. With the fake backend, `Model` is `chriek-fake-backend` and `Model-Revision` is `fake-v1`.
- No retention: generated files stay in `CHRIEK_STORAGE_DIR` until you delete them.

### Queue limit

Each owner may have at most `CHRIEK_MAX_QUEUED_JOBS` jobs (default `100`, configurable 1–100000) in status `queued` at once. Only `queued` jobs count: once the worker starts a job (`running`) it frees its slot, and `succeeded` or `failed` jobs never count. A submission that would exceed the limit returns `429 {"detail": "queue full: at most <n> queued jobs per owner; retry after some of them have started"}`; the rejected job is not stored and never runs. Each API key has its own limit; in `proxy` mode all clients share one. The check and the insert are one SQLite statement, so concurrent submissions, also from several `chriek serve` processes on the same database file, can't exceed the limit.

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
uv sync --locked && uv run --locked pre-commit run jscpd --all-files && uv run ruff check . && uv run ruff format --check . && uv run pytest
```

The TLS tests generate throwaway certificates with the `openssl` command, so it must be on `PATH` (OpenSSL or LibreSSL).

Tests drive chriek only through its HTTP API and the installed `chriek` command, with the fake backend.
