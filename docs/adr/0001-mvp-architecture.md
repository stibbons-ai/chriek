---
status: accepted
---

# MVP architecture: async job API, two deployment modes, push dispatch

chriek is a self-hostable FLUX 3 Image Klein server, extracted from the fabula-mia PoC runtime. Its first consumers are fabula-mia (character reference sheets and story panels) and Ikonograf (profile pictures from reference photos). Full spec: #1. Decisions and rationale below were made with Daniel in the design session of 2026-11-11 (source design: [docs/superpowers/specs/2026-11-11-chriek-design.md](../superpowers/specs/2026-11-11-chriek-design.md)).

## Decisions

- **Async job API, not synchronous generation.** `POST /jobs` returns `202` and a job ID right away; clients poll `GET /jobs/{id}` and receive signed, time-limited download URLs. Why: one generation takes about 1 min (text-to-image) to 5 min (edit with 3 references) on an M4 Max, and requests that long break proxies and clients. Daniel confirmed "job annehmen und ergebnis abfragen". Batch processing (several prompts per request) was explicitly excluded from the MVP.
- **Two deployment modes, one codebase.**
  - **Home:** a single process holds API and worker, using SQLite on a configurable path (so it can live on a named volume) and a local directory.
  - **Serverless:** an always-on, CPU-only API plus a GPU worker the provider scales to zero, sharing Postgres and an S3-compatible bucket.
  - Why: fabula-mia production must scale to zero ("sonst haben wir enorm hohe, unnötige kosten für idle runtime", Daniel), and home use on a Mac or a 3090/4090/5090 must work without infrastructure. Daniel accepted keeping the small API always-on.
- **Push dispatch, not polling workers.** In serverless mode the API sends each job ID to the provider's endpoint, and the worker handles one job per HTTP invocation. Why: a polling worker never scales to zero. The dispatcher is provider-neutral (URL, credentials, extra headers are config). Deploying to a specific provider (e.g. Verda for fabula-mia) is the consuming project's job, not chriek's. Daniel: "das ist doch eher eine Anforderung für das nutzende Projekt".
- **Auth built in, or delegated to a reverse proxy.** `api-key` mode uses named static keys. A missing or invalid key returns `401`, and a job owned by another key returns `404`. `proxy` mode skips the check. Why: chriek has to be safe on a network without a reverse proxy, while still allowing one (Daniel). Each consuming app gets its own key, and keys don't see each other's jobs (Daniel: "API keys getrennt").
- **Model: FLUX 3 Image Klein, BF16, pinned revision, on MLX (Metal) and PyTorch (CUDA).** Text-to-image and edits with reference images are both in the MVP, since edits are what fabula-mia uses (Daniel). Operators download the gated weights with their own HF token, and chriek never redistributes them. The minimum NVIDIA card is 24 GB (3090 class) with sequential offload. Smaller cards (e.g. 4070) and quantization are out of scope (Daniel: "lass die 4070 raus").
- **Every PNG is marked as AI-generated:** a text chunk plus XMP with IPTC `trainedAlgorithmicMedia`, always on. Content filtering stays with the operator. Why: the FLUX Kommunity License requires AI disclosure and content review before distribution. Daniel agreed to build in disclosure and document filtering as the operator's job.
- **Stack: Python, FastAPI, SQLAlchemy, boto3, uv.** Why: the model runtimes are Python, and Daniel requires uv as the recommended install path.

## Consequences

- Serverless mode requires Postgres and S3. SQLite and local storage only work in home mode.
- The first job after an idle period pays a cold model load (about 20 GB of weights). That is accepted.
- CI has no GPU. Tests go through the HTTP API with a fake image backend, and the real backends are verified with a smoke-test command on real hardware.
- License risk, not blocking: the FLUX Kommunity License bans "biometric processing", and Ikonograf may conflict with that. It needs a license review before Ikonograf goes live.