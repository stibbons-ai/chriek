# chriek

Self-hostable image generation server for FLUX 3 Image Klein, with an asynchronous job API: submit a job, poll it, download the PNGs through signed, expiring URLs. Architecture: [ADR-0001](docs/adr/0001-mvp-architecture.md).

> **Status: walking skeleton.** Only **home mode** with a **fake image backend** exists. The fake backend draws deterministic placeholder PNGs (two coloured rectangles derived from prompt, size, seed and uploaded reference images). It does **not** run FLUX or any other model, needs no GPU and downloads no weights. The real MLX (Apple Silicon) and CUDA backends come later; the HTTP contract below stays the same.

## Install and run (uv)

Requires [uv](https://docs.astral.sh/uv/); uv installs Python 3.14 itself if needed.

```sh
git clone https://github.com/stibbons-ai/chriek.git
cd chriek
uv sync --locked

export CHRIEK_API_KEYS="consumer:$(openssl rand -hex 32),ikonograf:$(openssl rand -hex 32)"
export CHRIEK_SIGNING_SECRET="$(openssl rand -hex 32)"
uv run chriek serve
```

`chriek serve` runs the API and the in-process worker in one process (home mode). It listens on `127.0.0.1:8000` by default. A setting rejected by configuration validation (see the ranges and formats in [Configuration](#configuration)) stops it before it opens the database or a port, with exit code 2 and a message that names the variable but never prints secret values. Values that aren't validated, such as an unresolvable `CHRIEK_HOST`, fail later, when the server binds.

Logs go to standard output/error: startup, shutdown and errors. There is no HTTP access log, because download URLs carry their signature, which works like a password until it expires; put any access logging you need in a reverse proxy and strip the query string there.

chriek has no TLS of its own. Expose it only on a private network or behind a TLS-terminating reverse proxy: API keys and download signatures travel in requests.

## Configuration

All configuration comes from environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `CHRIEK_API_KEYS` | required | Named static API keys, `name:key,name:key`. The name is the owner of the jobs created with that key. Names: 1–64 characters from `A-Z a-z 0-9 _ . -`, unique. Keys: at least 32 printable ASCII characters (`!` to `~`; no spaces, control or non-ASCII characters, no commas), unique. Keys travel in an HTTP header, so other characters are rejected at startup. |
| `CHRIEK_SIGNING_SECRET` | required | HMAC secret for download URLs, at least 32 characters. Changing it invalidates every issued URL (polling the job again mints new ones). |
| `CHRIEK_URL_TTL_SECONDS` | `900` | Lifetime of a download URL, 1–604800 seconds. |
| `CHRIEK_DB_PATH` | `chriek-data/chriek.db` | SQLite database file, used literally (`?` and `%20` are ordinary filename characters); parent directories are created. Put it on a volume to keep jobs across restarts. The exact value `:memory:` (or `./:memory:`) selects an ephemeral in-memory database instead; see [In-memory database](#in-memory-database). |
| `CHRIEK_STORAGE_DIR` | `chriek-data/images` | Directory for generated PNGs (`<id>/<index>.png`) and uploaded reference images (`<id>/references/<index>.png`), one subdirectory per job. |
| `CHRIEK_HOST` | `127.0.0.1` | Listen address. |
| `CHRIEK_PORT` | `8000` | Listen port. |
| `CHRIEK_MAX_IMAGES` | `4` | Maximum `image_count` per job (configurable 1–64). |
| `CHRIEK_MAX_DIMENSION` | `2048` | Maximum `width` and `height`; a multiple of 16 (16–8192). |
| `CHRIEK_MAX_PIXELS` | `1048576` | Maximum `width * height` for generated outputs and uploaded reference images (256–67108864). |
| `CHRIEK_MAX_PROMPT_CHARS` | `2000` | Maximum prompt length in characters (1–100000). |
| `CHRIEK_MAX_REFERENCES` | `4` | Maximum uploaded reference images per job (0–16; `0` disables edit jobs). |
| `CHRIEK_MAX_UPLOAD_BYTES` | `20971520` | Maximum raw HTTP request body size in bytes across the entire `POST /jobs` request, enforced while streaming without trusting `Content-Length` (1024–536870912). |

Relative paths resolve against the working directory. The database schema is created on first start; there are no migrations yet.

### In-memory database

`CHRIEK_DB_PATH=:memory:` keeps the job database in memory, meant for tests and local development. No database file is written. The database lives exactly as long as one `chriek serve` process: submissions, the worker and polling share it while the process runs, and every restart starts empty, so job IDs from an earlier run return `404` and their download URLs stop working. The restart guarantees in [Job behaviour](#job-behaviour) (queued jobs survive, running jobs fail as interrupted) hold only for a database file. PNGs still go to `CHRIEK_STORAGE_DIR` on disk and aren't deleted on restart.

## API

Every `/jobs` request needs `Authorization: Bearer <key>`. A missing or invalid key returns `401` before the request body is read, so even a malformed body gets `401`. A job created with another key returns `404`, exactly like a job that doesn't exist. `GET /health` and image downloads need no key.

### `POST /jobs`

`POST /jobs` accepts either `application/json` (text-to-image) or `multipart/form-data` (zero or more uploaded `references` image files: zero selects text-to-image, one or more selects image-conditioned edit). The server never fetches reference URLs.

```sh
# Text-to-image (JSON)
curl -s -X POST http://127.0.0.1:8000/jobs \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "a lighthouse at dusk", "image_count": 2, "seed": 41, "width": 1024, "height": 768}'

# Edit job with reference images (multipart/form-data)
curl -s -X POST http://127.0.0.1:8000/jobs \
  -H "Authorization: Bearer $KEY" \
  -F "prompt=keep both characters consistent in a stormy harbor" \
  -F "image_count=2" -F "seed=41" -F "width=1024" -F "height=768" \
  -F "references=@char1.png" -F "references=@char2.jpg"
```

| Field | Type | Default | Rule |
|---|---|---|---|
| `prompt` | string | required | not blank, at most `CHRIEK_MAX_PROMPT_CHARS` |
| `image_count` | integer | `1` | 1 to `CHRIEK_MAX_IMAGES` |
| `seed` | integer or null | random | `0 ≤ seed` and `seed + image_count - 1 ≤ 4294967295` |
| `width`, `height` | integer | `1024` | multiples of 16, 16 to `CHRIEK_MAX_DIMENSION`, `width * height ≤ CHRIEK_MAX_PIXELS` |
| `references` | 0..N image files (multipart only) | none | at most `CHRIEK_MAX_REFERENCES` files; each decoded as an image with `width * height ≤ CHRIEK_MAX_PIXELS`; total raw request body at most `CHRIEK_MAX_UPLOAD_BYTES` |

In JSON requests, types are strict (`"1024"` or `1024.0` is not an integer). In `multipart/form-data`, scalar fields use the same names with decimal integer strings (`seed=` empty string or omitted chooses a random seed), and reference images are uploaded as repeated `references` file parts in order; client filenames are ignored and stored as `references/<index>.png` inside the job's storage directory before the job enters the queue. Unknown fields, non-file `references` parts (such as URLs), unreadable images, and count or byte limit violations return `422` with FastAPI's `{"detail": [...]}` error list. Nonfinite numbers (`NaN`, `Infinity`, `1e999`) and prompts with unpaired UTF-16 surrogates (`"\ud800"`) are rejected too; error entries echo such input as text (`"inf"`, `"\\ud800"`). Other JSON body decoding failures, such as invalid UTF-8 or an integer literal thousands of digits long, return `400 {"detail": "There was an error parsing the body"}` instead. Neither kind of rejected request enters the queue. A valid request returns `202`, a `Location: /jobs/<id>` header and the job (below) in status `queued`. Job IDs are UUIDv7. When `seed` is omitted, chriek picks one at random and reports it in `params.seed`; image `i` uses seed `seed + i`.

### `GET /jobs/{id}`

```sh
curl -s http://127.0.0.1:8000/jobs/$JOB_ID -H "Authorization: Bearer $KEY"
```

```json
{
  "id": "019a7c1e-4b2a-7c3d-9e8f-0123456789ab",
  "status": "succeeded",
  "params": {"prompt": "a lighthouse at dusk", "image_count": 2, "seed": 41, "width": 1024, "height": 768, "reference_count": 0},
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
- **Unsaved results stop the worker.** If saving a finished job's result to the database fails, the worker logs the error and stops instead of moving on: that job stays `running`, no later job runs, new submissions stay `queued`, and `GET /health` returns `503`. Fix the database or disk problem and restart `chriek serve`; restart recovery then fails the stuck job as interrupted and the queue resumes.
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
uv sync --locked && uv run --locked pre-commit run jscpd --all-files && uv run ruff check . && uv run ruff format --check . && uv run pytest
```

Tests drive chriek only through its HTTP API and the installed `chriek` command, with the fake backend.
