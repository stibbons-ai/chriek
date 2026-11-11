# chriek — design

Standalone, self-hostable image generation server for FLUX 3 Image Klein. Extracted from the fabula-mia PoC runtime (`runtime/flux-local/` at tag `poc-final-20261110T105632Z`). Repo: `stibbons-ai/chriek` (public, MIT). Named after Otto Chriek, Discworld's iconographer.

Drafted in the fabula-mia design session on 2026-11-11 and moved here. Spec: #1. Decisions: ADR-0001.

## Intent

- fabula-mia (first consumer): character reference sheets (text-to-image) and story panels (edit conditioned on the reference images of all characters in a scene), as in the PoC.
- Ikonograf (second consumer): LinkedIn profile picture from a few reference photos of a real person.
- Runs on Apple Silicon (Metal, as in the PoC) and NVIDIA GPUs: serverless hosted GPUs with scale-to-zero for fabula-mia production (idle GPU cost is unacceptable), consumer cards (3090/4090/5090) for home/hobby use.
- Reachable over the network and safe without a reverse proxy; also deployable behind one.
- MVP, not a big project.

## Goals

- REST API: submit a generation job, poll its result, download images via authenticated URLs.
- Text-to-image and reference-image editing from day one.
- One prompt → 1..N images per job.
- Storage: local directory works out of the box; S3-compatible bucket as alternative.
- Jobs/queue persisted in SQLite (configurable path, e.g. on a named volume); Postgres supported.
- `uv` everywhere; the README's recommended path is uv.

## Non-goals (MVP)

- Batch processing (multiple prompts per request).
- Webhooks.
- Built-in content filter / safety checker.
- fp8 / nvfp4 / quantization; 12 GB cards (e.g. 4070).
- Automatic hardware detection.
- Retention / cleanup of stored files (S3 lifecycle rules cover the bucket case).
- Other models. Code doesn't need to be model-agnostic beyond keeping the backend boundary clean.
- User accounts, key management API.
- Provider-specific deployment tooling (belongs to consuming projects).

## Anti-goals

- Synchronous long-running requests (2–5 min per edit image breaks proxies/clients).
- Spawning a Python process per image (PoC pattern): the model stays loaded while a worker lives.
- A GPU that runs (and bills) while idle in serverless mode, e.g. because a worker polls for work.
- Any key or job ID being the only thing protecting an image.
- Vendoring or redistributing FLUX weights.

## Decided

### Stack

- Python, FastAPI, SQLAlchemy (SQLite + Postgres), boto3 for S3.
- uv for dependency management; `uv.lock` committed.
- Mac: runs natively via `uv run`. NVIDIA: Docker image (CUDA); uv used in the build stage (`uv sync --frozen`), runtime image ships only the venv.
- One codebase, two roles, two deployment modes (see Deployment modes).

### Deployment modes

- **Home** (`chriek serve`): API + worker in one process. SQLite on a configurable path, local storage dir. In-process dispatch.
- **Serverless** (`chriek api` + `chriek worker`):
  - `chriek api`: always-on, CPU only (cheap). REST API, auth, jobs, signed URLs. Postgres + S3 required (api and worker run on different machines).
  - `chriek worker`: GPU container scaled to zero by the provider. Handles exactly one job per invocation: loads job from DB, reads references from S3, renders, writes outputs + status, exits the request.
  - Dispatch is push, not poll: on `POST /jobs` the api calls the provider's serverless endpoint with the job ID; the provider's own queue/autoscaler starts and stops GPUs.
  - Worker's invocation interface is a plain HTTP endpoint (job ID in, status out), so the image stays provider-neutral; provider-specific bits are limited to dispatch config (endpoint URL, credentials, extra headers).
  - Deploying to a specific provider is the consuming project's job, not chriek's. Worker exposes `GET /health`; weights cached on a persistent volume via `HF_HOME`.
  - A job killed mid-run (e.g. scale-down SIGTERM) ends up `failed` via the running-timeout.
  - Reference consumer setup (fabula-mia, guidance only): Verda Serverless Containers, continuous deployment, min replicas 0, async requests (`Prefer: respond-async`), not Batch jobs (a fresh replica per job = cold load + 10-minute billing chunk per panel). Docs checked 2026-11-11. Open with Verda: max request TTL, result retention, FIN-only pinning, cold-start time.
  - Cold start (loading ~20 GB weights) adds latency to the first job after idle; accepted. Weights come from the image or a provider volume, never downloaded from HF per start.

### API

- All endpoints authenticated (see Auth).
- `POST /jobs`: prompt, optional reference images, image count, optional seed, width/height. Returns `202` + job ID immediately.
- `GET /jobs/{id}`: status `queued | running | succeeded | failed`; on success signed download URLs per image; on failure an error message.
- Job IDs: UUIDv7.
- Jobs are scoped to the API key that created them; other keys get `404`.

### Auth

- Mode `api-key`: static API keys from config; each key has a name (owner). Constant-time comparison.
- Mode `proxy`: no key check in the app (reverse proxy authenticates); all jobs belong to one shared owner.
- Download URLs: signed and time-limited. Local storage: app signs and serves them (no header needed to download). S3: presigned URLs.

### Generation

- Model: FLUX 3 Image Klein, `black-forest-labs/flux-3-image-klein`, pinned revision (PoC: `09561231f44e41381c461caf53d5452005a60ac9`), BF16, unquantized.
- Weights are not in the repo. Operators download them with their own HF token (gated repo) and accept the FLUX Kommunity License themselves.
- PoC parameters as baseline: 16 steps fixed, no guidance/CFG, no prompt upsampling, width/height multiples of 16, ~1 MP. References → edit mode; none → text-to-image.
- Same request contract on both backends.
- Mac backend: flux3-mlx on MLX (as in PoC).
- NVIDIA backend: PyTorch/CUDA, BF16.
- Config `offload`: `none` (hosted ≥48 GB, Mac) or `sequential` (24 GB cards: text encoder encodes, is unloaded, then transformer runs).
- Minimum hardware (README): NVIDIA 24 GB VRAM; Apple Silicon ≥32 GB unified memory (guess, from 22.6 GB PoC peak).
- Every PNG carries metadata: "AI generated", model, revision, seed. Always on, not configurable.

### Queue

- Jobs stored in the DB; survive restart.
- Home: one worker per GPU, FIFO. A job that was `running` at startup is marked `failed` (no auto-retry).
- Serverless: ordering and concurrency come from the provider's queue. A job stuck in `running` past a timeout is marked `failed`.
- Reference images uploaded with the job are stored in the same storage as outputs.

### Calls made by the designer (not discussed; check)

- Home: one GPU per chriek instance; single in-process consumer. Multi-GPU home host = multiple instances.
- Transport: optional TLS cert/key paths passed to the ASGI server; no ACME. Without TLS, README says: only on a private network or behind a TLS terminator.
- Abuse limit: per-key cap on queued jobs (configurable); no rate limiter beyond that.
- N images per job: seeds `seed + i`; any image failure fails the whole job (generation errors are systemic, not per-image). Images already produced are discarded.
- AI disclosure: PNG text chunk plus XMP with IPTC `DigitalSourceType = trainedAlgorithmicMedia`, written via Pillow (no extra dependency). No C2PA in MVP.
- Test hardware: RTX 3090 (24 GB, `offload=sequential`) and M4 Max 128 GB. 24 GB fit of multi-reference BF16 edits is unproven; the plan builds and checks the NVIDIA backend on the 3090 early, before the rest depends on it.

### Docs

- README: uv path first; hardware presets per card class; operator responsibility for content filtering and the FLUX license (incl. revenue cap, AI disclosure, biometric clause).

## Known risk

- FLUX Kommunity License bans "biometric processing". Ikonograf (portraits of real people from their photos) may conflict. Needs a careful read of the license before Ikonograf goes live. Not a blocker for chriek.

## Left to the builder

- Python version (3.14 if MLX/torch/flux3-mlx support it, for stdlib `uuid.uuid7`; otherwise a small UUIDv7 library).
- Exact NVIDIA inference library (BFL reference code, diffusers, …): verify against current docs; the HF repo ships diffusers-format copies.
- Upload format for reference images (multipart proposed; no server-side fetching of URLs, to avoid SSRF).
- Config mechanism and names (env vars recommended), signed-URL TTL, request limits (max images per job, max references, max upload size).
- Migrations approach, health endpoint, logging, per-job run record (PoC fields: timings, peak memory, hardware) in the job response.
- Repo layout, tests, CI.
