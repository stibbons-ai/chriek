# chriek

Self-hostable image generation server for FLUX 3 Image Klein, with an asynchronous job API: submit a job, poll it, download the PNGs through signed, expiring URLs. Architecture: [ADR-0001](docs/adr/0001-mvp-architecture.md).

> **Status: home mode.** The default fake backend draws deterministic placeholder PNGs (two coloured rectangles derived from prompt, size, seed and uploaded reference images) without a GPU or model weights; it does not run FLUX or any other model. On Apple Silicon, the optional MLX backend runs FLUX 3 Image Klein in BF16, including edits with reference images uploaded through the API or the local smoke command. CUDA is not available yet.

## Install and run (uv)

Requires [uv](https://docs.astral.sh/uv/); uv installs Python 3.14 itself if needed.

```sh
git clone https://github.com/stibbons-ai/chriek.git
cd chriek
uv sync --locked

export CHRIEK_API_KEYS="consumer:$(openssl rand -hex 32),ikonograf:$(openssl rand -hex 32)"
export CHRIEK_SIGNING_SECRET="$(openssl rand -hex 32)"
export CHRIEK_PUBLIC_URL="http://127.0.0.1:8000"
uv run chriek serve
```

Generate every API key and the signing secret independently from a cryptographically secure random source, as above. chriek only checks lengths, characters and that no value is reused; it can't tell a guessable value from a random one.

chriek runs only on POSIX hosts and filesystems (Linux, macOS) and refuses to start when a directory above its data is group- or other-writable without the sticky bit, such as a `0775` checkout under umask `002`; see [Storage permissions](#storage-permissions).

`chriek serve` runs the API and the in-process worker in one process (home mode). It listens on `127.0.0.1:8000` by default. A missing setting that has no default, such as `CHRIEK_PUBLIC_URL`, or a setting rejected by configuration validation (see the ranges and formats in [Configuration](#configuration)) stops it before it opens the database or a port, with exit code 2 and a message that names the variable but never prints secret values or the rejected `CHRIEK_PUBLIC_URL`. A database the driver can't open or reach (a Postgres server that refuses the connection or a host name it can't resolve, a SQLite file it can't open) stops it with exit code 1 and `chriek: database error: ` followed by the driver's message, with the database password masked as `***`. Values that aren't validated, such as an unresolvable `CHRIEK_HOST`, fail later, when the server binds.

Every environment variable whose name starts with `CHRIEK_` (in any letter case) must be one of the settings in [Configuration](#configuration); any other, such as a misspelt `CHRIEK_TLS_CERTFLE`, stops startup the same way, naming the variable but never its value. Earlier versions ignored unknown `CHRIEK_` names, so a leftover or misspelt one in an existing deployment now stops startup until you remove or correct it. On Kubernetes, a Service named `chriek` or starting with `chriek-` (such as `chriek-web`) can inject variables like `CHRIEK_WEB_SERVICE_HOST` into pods in its namespace; set `enableServiceLinks: false` in the pod spec, or rename the Service. Other variables, such as uvicorn's `FORWARDED_ALLOW_IPS`, are unaffected.

Logs go to standard output/error: startup, shutdown and errors. At startup chriek logs its authentication mode, as a `WARNING` in `proxy` mode because chriek itself then authenticates nobody; it never logs keys or secrets. There is no HTTP access log, because download URLs carry their signature, which works like a password until it expires; put any access logging you need in a reverse proxy and strip the query string there.

> **Warning: without TLS, use chriek only on a private network or behind a TLS-terminating reverse proxy.** API keys travel in `/jobs` requests in `api-key` mode; download signatures travel in image URLs in both modes. Over plain HTTP anyone on the path can read and reuse these credentials. Either let a reverse proxy terminate TLS, or turn on [native TLS](#native-tls).

### Native TLS

chriek can serve HTTPS itself, without a proxy. Point it at a PEM certificate (chain) and its unencrypted PEM private key:

```sh
export CHRIEK_TLS_CERTFILE=/etc/chriek/fullchain.pem
export CHRIEK_TLS_KEYFILE=/etc/chriek/privkey.pem
export CHRIEK_PUBLIC_URL="https://127.0.0.1:8000"
uv run chriek serve   # logs "Uvicorn running on https://127.0.0.1:8000"
```

Set both variables or neither. With TLS configured, chriek never falls back to plain HTTP. Configuration validation loads the pair once: if only one variable is set, either is set but empty, or the files can't be loaded as a matching certificate and key (missing, unreadable, not PEM, mismatched, or an encrypted key), startup stops with exit code 2. A path that is missing, not a regular file or not readable is reported with the variable that names it; an encrypted key names `CHRIEK_TLS_KEYFILE`; other load failures, such as a mismatched pair, name both variables. The server loads the files again when it starts. If that load fails, startup stops with the server's own load error and no listener opens. Replacing the files between validation and server startup can change the outcome: another valid pair serves HTTPS, and an encrypted key can prompt for a pass phrase on a terminal. Neither case enables plain HTTP. A plain-HTTP request to a TLS port gets no HTTP answer. chriek doesn't obtain or renew certificates (no ACME): renew them yourself and restart `chriek serve` to load the new files. TLS doesn't change download URLs: they always start with `CHRIEK_PUBLIC_URL`, so set it to the server's `https://` origin (see [Public URL and reverse proxies](#public-url-and-reverse-proxies)).

The exit-2, before-database guarantee applies only to configuration validation. A TLS load failure during server startup can instead exit 1 with a traceback after the database has been created; it still opens no listener.

### Behind a reverse proxy (`CHRIEK_AUTH_MODE=proxy`)

In `proxy` mode chriek does no authentication of its own: the reverse proxy in front of it must authenticate every client before forwarding a request. chriek ignores `Authorization` and identity headers such as `X-Forwarded-User`; it never derives an owner from headers. Every request belongs to one fixed, shared owner, so **all clients share all jobs**: any client can poll any job, and all of them share one queue capacity (`CHRIEK_MAX_QUEUED_JOBS`). Run separate chriek instances, or use `api-key` mode, when clients must not see each other's jobs.

Anyone who reaches chriek's port directly bypasses the proxy and gets full access. Restrict access to trusted peers: with the default `CHRIEK_HOST=127.0.0.1`, every local process and user can reach it; on a private network, every host with access to the port can reach it. Neither binding alone restricts access to the proxy. Use host isolation, a dedicated container network or firewall rules when other peers must not have access. `CHRIEK_API_KEYS` must be unset, empty or whitespace only in proxy mode; `CHRIEK_SIGNING_SECRET` is still required with at least 32 characters, and downloads still need valid signed URLs. Polling is separate per mode: jobs created in one mode can't be polled in the other, even on the same database. Signed download URLs are bearer links and don't depend on the mode: one issued before a mode change keeps working until it expires, as long as the database, image directory and `CHRIEK_SIGNING_SECRET` stay the same.

In `proxy` mode chriek answers only requests whose `Host` header names one of `CHRIEK_ALLOWED_HOSTS`, by default `localhost`, `127.0.0.1` and `[::1]`; every other request, on every path, gets `400 Invalid host header` before chriek reads or stores anything. Requests with no, several or a malformed `Host` header get `400` too, with nothing read or stored, though the server's HTTP parser may reject some of them first with a different body. Entries are exact host names or IP addresses, matched case-insensitively; the request's `Host` may add any port. IPv6 addresses match as written, not by value: `[::1]` and `[0:0:0:0:0:0:0:1]` are different entries. This stops DNS rebinding: a web page on another site can't make a visitor's browser drive chriek through its own host name. It is not authentication: every local process and every peer that reaches the port and sends an allowed `Host` still has full access, so the network isolation above stays necessary. Behind a public reverse proxy, set `CHRIEK_ALLOWED_HOSTS` to the public host name (for example `img.example.com`) and have the proxy forward the original `Host`; the replacement list drops the defaults, so add `127.0.0.1` too if local health checks use it. chriek never derives allowed names from the listen address. Only `Host` counts; `X-Forwarded-Host` is ignored.

Download URLs always start with `CHRIEK_PUBLIC_URL` (see [Public URL and reverse proxies](#public-url-and-reverse-proxies)): set it to the proxy's public origin, for example `https://img.example.com`. Neither `Host` nor `X-Forwarded-Proto` changes them, whatever `FORWARDED_ALLOW_IPS` trusts.

```sh
export CHRIEK_AUTH_MODE=proxy
unset CHRIEK_API_KEYS
export CHRIEK_SIGNING_SECRET="$(openssl rand -hex 32)"
export CHRIEK_PUBLIC_URL="https://img.example.com"
export CHRIEK_ALLOWED_HOSTS="img.example.com"
uv run chriek serve
```

### Backend dependencies

`uv sync --locked` installs the default fake backend, with no GPU runtime or model weights. `uv sync --locked --extra mlx` adds `flux3-mlx` and its inference dependencies on Apple Silicon. Installing the extra does not select the backend. There is no `cuda` extra yet.

Keep the committed `uv.lock` as the single lockfile; do not introduce hand-maintained `requirements_mlx.txt` or `requirements_cuda.txt` copies.

### Apple Silicon

Preset: Apple Silicon with **at least 32 GB of unified memory**, `CHRIEK_BACKEND=mlx`, `CHRIEK_OFFLOAD=none` (the pipeline keeps all components loaded). Leave headroom for other apps; larger images and references can raise the peak.

```sh
uv sync --locked --extra mlx
export CHRIEK_BACKEND=mlx CHRIEK_OFFLOAD=none
export HF_HOME="$HOME/.cache/huggingface"
```

Before the first render, accept the [model license](https://huggingface.co/black-forest-labs/flux-3-image-klein) using your Hugging Face account and supply your own `HF_TOKEN` through the environment or secret store. The first render downloads the pinned BF16 weights if they are missing from `HF_HOME`. With the snapshot cached, the backend resolves it locally before trying the network; set `HF_HUB_OFFLINE=1` to prohibit online lookups.

To serve MLX jobs, also set the server settings from [Install and run](#install-and-run-uv), then run `uv run --locked --extra mlx chriek serve`.

### Smoke test

Generate a labeled PNG and print a JSON run record to stdout. The command uses the configured backend, defaults to 1024×1024 and seed 7, and needs no server settings. Dimensions must be positive multiples of 16 and obey the default server limits for each side and for total pixels. `--reference` is repeatable. An existing output is replaced only after the new PNG is fully written. Server jobs take references as multipart uploads; see [`POST /jobs`](#post-jobs).

```sh
uv run --locked --extra mlx chriek smoke --prompt "A red bicycle in the rain" --out bicycle.png
uv run --locked --extra mlx chriek smoke --prompt "Move the bicycle into a sunny garden" \
  --reference bicycle.png --out garden.png
```

MLX uses 16 steps, without guidance or prompt upsampling. The smoke record adds `prompt`, `seed`, `width`, `height`, `references` (count), `output` and `steps` (`null` for fake) to the normal run record. If MLX cannot read the GPU core count, `hardware.gpu_cores` is `null` and rendering continues.

Measured offline on an Apple M4 Max (40 GPU cores, 128 GB, macOS 27.2.1), with `CHRIEK_OFFLOAD=none`, MLX 0.32.4, `flux3-mlx` 0.1.0, 1024×1024, seed 7:

| Mode | References | Load (s) | Wall (s) | MLX peak (bytes) |
|---|---:|---:|---:|---:|
| Text-to-image | 0 | 3.879 | 57.895 | 22,610,518,196 |
| Edit | 1 | 0.542 | 112.444 | 22,615,761,072 |
| Edit | 2 | 0.640 | 197.514 | 22,621,003,960 |

Two offline 512×512 jobs in one server process recorded 3.662 s and 0.0 s of load time, respectively.

## Configuration

All configuration comes from environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `CHRIEK_AUTH_MODE` | `api-key` | `api-key`: every `/jobs` request needs a key from `CHRIEK_API_KEYS`, and each key owns its jobs. `proxy`: no key check; a trusted reverse proxy authenticates, and all clients are one shared owner; see [Behind a reverse proxy](#behind-a-reverse-proxy-chriek_auth_modeproxy). Empty, differently cased or other values stop startup. |
| `CHRIEK_API_KEYS` | required in `api-key` mode | Named static API keys, `name:key,name:key`. The name is the owner of the jobs created with that key. Names: 1–64 characters from `A-Z a-z 0-9 _ . -`, unique. Names are not secret: each job stores its owner's name in plain text, so never use a key or the signing secret as a name. Keys: at least 32 printable ASCII characters (`!` to `~`; no spaces, control or non-ASCII characters, no commas), unique. Keys travel in an HTTP header, so other characters are rejected at startup. Startup errors identify an invalid entry by its position, never by its name or key. Must be unset, empty or whitespace only in `proxy` mode. |
| `CHRIEK_BACKEND` | `fake` | `fake`: deterministic placeholders. `mlx`: BF16 FLUX 3 Image Klein on Apple Silicon; requires the `mlx` extra. Exact lowercase values only. |
| `CHRIEK_OFFLOAD` | `none` | Only exact lowercase `none` is supported today. MLX keeps the text encoder, DiT and VAE loaded. |
| `CHRIEK_SIGNING_SECRET` | required | HMAC secret for download URLs, at least 32 characters. Changing it invalidates every issued URL (polling the job again mints new ones). Anyone who knows it can sign download URLs, so it must not be one of the API keys: startup rejects a signing secret equal to any key or name in `CHRIEK_API_KEYS`. |
| `CHRIEK_PUBLIC_URL` | required | Public base URL of download links, `http(s)://host[:port][/prefix]`, for example `https://images.example/chriek`. Final `/` characters are optional and ignored. The host is a DNS name, a dotted-decimal IPv4 address such as `192.0.2.10` or a bracketed IPv6 address; a port must be 1–65535. Hosts that browsers would read as a different IPv4 address (`017.0.0.1`, `2130706433`, `0x7f.1`) or as an invalid one (`1.2.3.4.5`) are rejected. The prefix must not contain `\`, `.` or `..` segments (also percent-encoded as `%2e`), which clients rewrite. Only printable ASCII is accepted (no whitespace or control characters; write internationalized host names in punycode), with no user name or password, query (`?`) or fragment (`#`). See [Public URL and reverse proxies](#public-url-and-reverse-proxies). |
| `CHRIEK_URL_TTL_SECONDS` | `900` | Lifetime of a download URL, 1–604800 seconds. |
| `CHRIEK_JOB_TIMEOUT_SECONDS` | `3600` | Longest a job may stay `running`, 1–604800 seconds, counted from `started_at` (time spent `queued` doesn't count). A job running longer fails as timed out; see [Job behaviour](#job-behaviour). |
| `CHRIEK_DATABASE_URL` | unset | Postgres database, `postgresql://user:password@host:5432/dbname` (or `postgresql+psycopg://…`; chriek always uses the psycopg 3 driver). When set, it takes precedence and `CHRIEK_DB_PATH` is ignored. Set but empty, unparsable, with another scheme, with an unencoded `@` other than the one before the host (more than one, or one after the first `/` or `?` following `://`; write every other `@` as `%40`, and a `/` or `?` in the user name or password as `%2F` or `%3F`), or containing a NUL character (also as `%00`) or raw invalid UTF-8 is a configuration error. A parsable URL with parts left out, such as bare `postgresql://`, is accepted: libpq fills them from its defaults and `PG*` variables, and a connection that then fails is a database error. See [Postgres](#postgres). |
| `CHRIEK_DB_PATH` | `chriek-data/chriek.db` | SQLite database file, used when `CHRIEK_DATABASE_URL` is unset. Used literally (`?` and `%20` are ordinary filename characters); missing parent directories are created private. Its parent must be a dedicated directory; see [Storage permissions](#storage-permissions). Put it on a volume to keep jobs across restarts. The exact value `:memory:` (or `./:memory:`) selects an ephemeral in-memory database instead; see [In-memory database](#in-memory-database). |
| `CHRIEK_STORAGE` | `local` | Where reference images and generated PNGs live: `local` (`CHRIEK_STORAGE_DIR`) or `s3` (an existing private bucket, `CHRIEK_S3_*`); see [S3 storage](#s3-storage). Exact lowercase values only. |
| `CHRIEK_STORAGE_DIR` | `chriek-data/images` | `local` storage only (setting it with `s3` stops startup). Dedicated directory for generated PNGs (`<id>/<index>.png`) and uploaded reference images (`<id>/references/<index>.png`), one subdirectory per job; see [Storage permissions](#storage-permissions). |
| `CHRIEK_S3_BUCKET` | required with `s3` | Name of an existing, private bucket (3–63 characters from `a-z 0-9 . -`). chriek never creates buckets or changes their policy. |
| `CHRIEK_S3_ACCESS_KEY_ID` | required with `s3` | Access key ID of the S3 credentials (printable ASCII without whitespace). |
| `CHRIEK_S3_SECRET_ACCESS_KEY` | required with `s3` | Secret access key (printable ASCII without whitespace). A secret: errors and logs never contain it. |
| `CHRIEK_S3_ENDPOINT_URL` | unset (AWS) | Endpoint of an S3-compatible service, `http(s)://host[:port]` without path, user name, password, query or fragment, validated like `CHRIEK_PUBLIC_URL`. Download links point here, so clients must reach it too. When set, chriek uses path-style addressing (`<endpoint>/<bucket>/<key>`); unset, the AWS endpoint of `CHRIEK_S3_REGION`. Set but empty stops startup. |
| `CHRIEK_S3_REGION` | `us-east-1` | Signing region (1–32 characters from `a-z 0-9 -`). |
| `CHRIEK_S3_PREFIX` | unset | Key prefix for every object, `/`-separated segments from `A-Z a-z 0-9 _ . -` without empty, `.` or `..` segments, at most 256 characters, for example `chriek/prod`. Unset stores at the bucket root. |
| `CHRIEK_HOST` | `127.0.0.1` | Listen address. |
| `CHRIEK_ALLOWED_HOSTS` | `localhost,127.0.0.1,[::1]` in `proxy` mode | `proxy` mode only: comma-separated host names and IP addresses (IPv6 in brackets, as in `[::1]`) that requests must name in their `Host` header; see [Behind a reverse proxy](#behind-a-reverse-proxy-chriek_auth_modeproxy). Exact names only: no wildcards, empty entries, schemes, ports or paths. Setting it in `api-key` mode stops startup. |
| `CHRIEK_PORT` | `8000` | Listen port. |
| `CHRIEK_TLS_CERTFILE` | unset | PEM certificate (chain) for [native TLS](#native-tls). Set together with `CHRIEK_TLS_KEYFILE`; leave both unset for plain HTTP. |
| `CHRIEK_TLS_KEYFILE` | unset | Unencrypted PEM private key matching `CHRIEK_TLS_CERTFILE`. |
| `CHRIEK_MAX_QUEUED_JOBS` | `100` | Maximum `queued` jobs per owner (1–100000); see [Queue limit](#queue-limit). |
| `CHRIEK_MAX_IMAGES` | `4` | Maximum `image_count` per job (configurable 1–64). |
| `CHRIEK_MAX_DIMENSION` | `2048` | Maximum `width` and `height`; a multiple of 16 (16–8192). |
| `CHRIEK_MAX_PIXELS` | `1048576` | Maximum `width * height` for generated outputs and uploaded reference images (256–67108864). |
| `CHRIEK_MAX_PROMPT_CHARS` | `2000` | Maximum prompt length in characters (1–100000). |
| `CHRIEK_MAX_REFERENCES` | `4` | Maximum uploaded reference images per job (0–10; `0` disables edit jobs). chriek returns `422` at submission if the count exceeds this setting. Configurations from earlier drafts that set `11`–`16` must lower the value to start. |
| `CHRIEK_MAX_UPLOAD_BYTES` | `20971520` | Maximum raw HTTP request body size in bytes of a `multipart/form-data` `POST /jobs` request (including framing), enforced while streaming without trusting `Content-Length` (1024–536870912). A small value can reject a request before `CHRIEK_MAX_PROMPT_CHARS` is reached. JSON bodies have the fixed 2 MiB budget instead; see [`POST /jobs`](#post-jobs). |

Relative paths resolve against the working directory. The database schema (SQLite or Postgres) is created on first start, and indexes added by newer chriek versions are created in an existing database on start, keeping its jobs. Tables added by newer versions are created the same way: an existing database gets the `job_references` table on its first start with this version, so a Postgres role then still needs the right to create tables in its schema (`CREATE` on it). The only other migration renames the draft SQLite `references` table to `job_references`; see [Reference table upgrade](#job-behaviour). There are no others yet.

### Postgres

Set `CHRIEK_DATABASE_URL` to keep jobs in Postgres instead of a SQLite file:

```sh
export CHRIEK_DATABASE_URL="postgresql://chriek:<percent-encoded password>@db.internal:5432/chriek?sslmode=verify-full"
```

- **Credentials.** The URL carries the database password, so treat the whole variable as a secret: inject it from your secret store or an environment file readable only by the service, not from a command line or a committed file. Percent-encode reserved characters in the user name and password (`%` as `%25`, `@` as `%40`, `:` as `%3A`, `/` as `%2F`, `?` as `%3F`) before putting them into the URL; don't interpolate a raw password. A literal `@` anywhere else (database name, query value) must be `%40` too. Query parameters are percent-decoded and passed to libpq as connection options, so `sslmode`, `sslrootcert` and the like work as in any Postgres client; in query values, unlike in the user name and password, `+` decodes to a space (write `%2B`). Give each parameter only once, and don't use `plugin`, which SQLAlchemy reserves. chriek always connects with client encoding UTF-8, overriding `client_encoding` and `PGCLIENTENCODING`. Use TLS (`sslmode=verify-full`) when the database isn't on the same host or a private network. chriek never logs the variable.
- **Database and role.** Create an empty database with `UTF8` encoding (the default of the official images and of `initdb` in a UTF-8 locale) and a role that owns it (or may create tables in its default schema); chriek creates its `jobs`, `images` and `job_references` tables on first start (`job_references` also in an existing database; see [Configuration](#configuration)) and never drops anything. Other database encodings are unsupported: a prompt the encoding can't represent gets `500`.
- **Limits of this release.** Postgres replaces only the job database. Run exactly **one** `chriek serve` per database: on startup, each process fails every job still `running` (see [Job behaviour](#job-behaviour)), so a second process would fail the first one's job in progress, and both would run queued jobs at once. PNGs (outputs and uploaded references) live in the configured image storage: `CHRIEK_STORAGE_DIR` on the local disk of that one process, or an [S3 bucket](#s3-storage). There is no tool to copy existing jobs from a SQLite file into Postgres. Connections that the server dropped while idle are replaced before use; a database still unreachable when a result is saved, or a connection lost in the middle of saving it, stops the worker (see [Job behaviour](#job-behaviour)).

### In-memory database

`CHRIEK_DB_PATH=:memory:` keeps the job database in memory, meant for tests and local development. No database file is written. The database lives exactly as long as one `chriek serve` process: submissions, the worker and polling share it while the process runs, and every restart starts empty, so job IDs from an earlier run return `404` and their download URLs stop working. The restart guarantees in [Job behaviour](#job-behaviour) (queued jobs survive, running jobs fail as interrupted) hold only for a database file. PNGs still go to the configured image storage and aren't deleted on restart.

### S3 storage

With `CHRIEK_STORAGE=s3`, uploaded references and generated PNGs are objects in an existing bucket of AWS S3 or an S3-compatible service, instead of files below `CHRIEK_STORAGE_DIR`:

```sh
export CHRIEK_STORAGE=s3
export CHRIEK_S3_ENDPOINT_URL=https://s3.example.internal:9000   # omit for AWS
export CHRIEK_S3_REGION=us-east-1
export CHRIEK_S3_BUCKET=chriek-images
export CHRIEK_S3_PREFIX=chriek                                   # optional
export CHRIEK_S3_ACCESS_KEY_ID=<access key id>
export CHRIEK_S3_SECRET_ACCESS_KEY=<secret access key>
```

- **Layout.** Keys mirror the local layout below the optional prefix: `<prefix>/<id>/references/<index>.png` for normalized reference uploads, `<prefix>/<id>/<index>.png` for outputs.
- **Private bucket.** The bucket must exist before chriek starts and stay private: no public-read policy or ACL, no anonymous listing. chriek never creates, configures or deletes buckets; it checks at startup that the bucket exists and the credentials reach it (`HeadBucket`). The credentials need `s3:ListBucket` on the bucket (for that check) and `s3:GetObject`, `s3:PutObject` and `s3:DeleteObject` on the prefix. Only static access keys are supported; treat both `CHRIEK_S3_*_KEY*` variables as secrets.
- **Direct, presigned downloads.** For a succeeded job, every poll returns for each image a presigned AWS Signature Version 4 `GET` URL that points straight at the bucket (`<endpoint>/<bucket>/<key>?X-Amz-…` with an endpoint, the AWS virtual-host or path form otherwise); image bytes never pass through chriek, and there is no fallback to chriek's `/images/…` route, which returns `404` in S3 mode. The link expires `CHRIEK_URL_TTL_SECONDS` (at most 604800, the SigV4 maximum) after it was signed, and `expires_at` is that signing time plus the TTL. The service enforces the signature and expiry: a changed key (another image or a reference), signature or expiry, an expired link or an unsigned request gets the service's `403`. Links are bearer credentials until they expire; chriek can't revoke one early except by rotating the access key. References are never linked.
- **Publication.** Outputs are written to their final keys while the job renders, but stay unreachable: links are only presigned once the whole batch is written and the job's success is stored. A failed, timed-out or interrupted-by-a-late-result job has its output objects deleted; references stay with their job. A submission rejected after its references were written (queue full, database failure) has its reference objects deleted before the error is returned.
- **Failures.** A missing bucket, wrong credentials or an unreachable endpoint stops startup (`chriek: storage error: CHRIEK_S3_BUCKET <bucket> cannot be used (<S3 error code or connection error>); …`, exit status 2), never printing credentials. While running, a failed reference write returns `500` and stores no job; a failed output write fails the job (`image N of M failed`); a reference that can no longer be read fails the job (`reference N of M could not be loaded`). Object deletions that fail are logged and leave orphans, never published images.
- **Switching storage.** Changing `CHRIEK_STORAGE`, the bucket, endpoint or prefix does not migrate anything: earlier jobs keep their database rows, but their images stay where they were written, so their downloads and queued edit jobs fail (`404` from the service or `reference N of M could not be loaded`). Copy the objects or files yourself (same layout) before switching, or let those jobs finish first.
- `CHRIEK_PUBLIC_URL` and `CHRIEK_SIGNING_SECRET` are still required and validated in S3 mode but unused for downloads.
- [Storage permissions](#storage-permissions) below apply to the local database and to `local` image storage only; in S3 mode chriek checks and creates only the database side on disk.

### Public URL and reverse proxies

Every download URL chriek returns is `CHRIEK_PUBLIC_URL` followed by `/images/{id}/{index}.png?expires=…&sig=…`; `https://images.example/chriek/` yields `https://images.example/chriek/images/…`. Nothing from the request goes into it: not the `Host` header, `Forwarded`, `X-Forwarded-Host`, `X-Forwarded-Proto`, nor the scheme or port the client connected with. The server (uvicorn) does honour `X-Forwarded-For` and `X-Forwarded-Proto` from trusted proxy addresses, by default only `127.0.0.1` and `::1`, a list the `FORWARDED_ALLOW_IPS` environment variable replaces; that changes the client address and request scheme the server sees, never the link origin.

- **Upgrading.** `CHRIEK_PUBLIC_URL` is new and has no default. Earlier versions built links from the host the client used; a deployment upgraded without setting it no longer starts (exit 2, `CHRIEK_PUBLIC_URL is required`). There is no fallback to the request host. Set it to the address clients use to reach chriek, for example the `http://127.0.0.1:8000` of the [install example](#install-and-run-uv) for local use. No data migration is needed.
- **Scheme.** `http://` suits local use on one machine. For anything reached over a network, use an `https://` URL served by native TLS or a TLS-terminating proxy: download URLs are bearer credentials until they expire.
- **Path prefix.** chriek always serves downloads at `/images/…` on its own listener; it does not strip or know about any prefix. With a prefix such as `/chriek`, your proxy must route `https://images.example/chriek/images/…` to chriek's `/images/…`, removing `/chriek` and keeping the query string unchanged. Without that mapping the links fail instead of returning the image: with the proxy's error, or with chriek's `404 {"detail": "Not Found"}` when the proxy forwards the path without removing the prefix. The `Location: /jobs/<id>` header of a submission carries no prefix either; a prefix proxy must rewrite it for clients that follow that header.

### Storage permissions

Jobs and images are private to the user chriek runs as (the service user) on POSIX filesystems, by mode bits. The database's directory (the parent of `CHRIEK_DB_PATH`) and `CHRIEK_STORAGE_DIR` must be dedicated to chriek and owned by the service user; the defaults (`chriek-data/` and `chriek-data/images/`) are. chriek creates missing directories with mode `0700`, the database with `0600` (SQLite gives its `-wal` and `-shm` sidecars the database's mode, also when it recreates them) and every PNG with `0600` from its first byte, including while a job is still rendering. These creation modes exclude group and others whatever the process umask is, for `chriek serve` and for code that embeds chriek's app, store or worker; ACLs and deleting a live database are separate limits (see below). Working additionally needs a umask that leaves the owner read, write and search rights (such as `022` or `077`): a umask like `0277` makes a first start fail and, on an existing installation, makes every job fail.

With `CHRIEK_DATABASE_URL` set, the database lives on the Postgres server and `CHRIEK_DB_PATH` is ignored: chriek checks and creates only `CHRIEK_STORAGE_DIR` on disk. Protect the database itself with Postgres roles, `pg_hba.conf` and TLS.

Before the database is opened, any job runs or any missing directory is created, `chriek serve` and `create_app` check both configured storage sides and refuse to start if any check fails (`chriek serve` prints `chriek: storage error: <path> <problem>. <guidance>` and exits with status 2 without listening). Used directly, `JobStore` checks only the database side and `chriek.images.LocalImages` (which a `Worker` is given) checks only the image side:

- The database's directory and `CHRIEK_STORAGE_DIR`, if they exist: a directory, not a symbolic link, owned by the service user (effective UID), no group or other permission bits (`mode & 077 == 0`).
- The database file and its `-wal`, `-shm` and `-journal` sidecars, if they exist: a regular file, not a symbolic link, owned by the service user, no group or other permission bits. Other files in the database's directory are not checked.
- Every directory and symbolic link followed to reach those two directories, including link targets and chains of links: each directory owned by root or the service user and not writable by group or others unless the sticky bit is set (as on `/tmp`), so dedicated private directories inside `/tmp` pass the checks (but see below on directories that are cleaned automatically); each link owned by root or the service user and inside a directory that is not writable by group or others, sticky or not. A link's owner alone proves nothing: in a sticky directory any user can hard-link a link root owns (as macOS allows), and in a writable one anyone can re-point it. Links such as macOS's `/var` (in the root-owned `/`) are resolved once at startup and chriek uses the real path from then on: re-pointing a link while chriek runs doesn't move its storage before a restart.
- Every file and directory below `CHRIEK_STORAGE_DIR`: not a symbolic link, owned by the service user, no group or other permission bits. A directory chriek can't list fails the check. This includes files chriek didn't write, such as a `.DS_Store`: with chriek stopped, delete that file or remove its group and other permissions (`chmod go= <file>`).

A path chriek can't check (for example below a regular file, inside a directory it can't search, through a symbolic link loop, or through a link whose target continues with `..` after a missing directory) is reported the same way. If a missing directory appears while chriek is creating it, chriek refuses it rather than using a directory it didn't create. The checks cover what already exists, not every later failure: creating the missing parts or opening the database can still fail, for example in a directory chriek can't write, with a file name too long for the filesystem, or with a database SQLite can't open or write. Such an error is reported as a storage error or, for some database errors, as a traceback; nothing is rolled back, so private directories and an empty private database file chriek created so far stay. Stop chriek and correct the cause before restarting; a corrected configuration or umask doesn't repair leftovers. Remove only newly created empty files/directories, or restore owner permissions on chriek's dedicated paths (directories need owner read, write and search; files need owner read and write). Never remove existing job data or change a shared parent.

chriek creates `CHRIEK_STORAGE_DIR` only at startup. If it is deleted while chriek runs, jobs fail for as long as it stays absent, until a restart checks and creates it again: a job without reference images is accepted and then fails (`image 1 of N failed`), a job already queued with reference images fails (`reference N of M could not be loaded`), and a new submission with reference images gets a server error (`500`) because its references can't be stored; no job is created for it. That promise ends if someone re-creates the directory meanwhile: chriek checks only at startup, so it would write later jobs into, and serve downloads from, a directory it never checked; whoever re-created it can make downloads return any file the service user can read. Keep chriek's data in a persistent location, not in a temporary directory that the system cleans automatically (as it may in `/tmp`).

Deleting the database or its sidecars while chriek runs is unsupported. SQLite may re-create them with the process umask rather than the private creation modes above; the existing `0700` database directory still blocks other users. A re-created database has no schema, requests can fail, and the next startup refuses any group- or other-accessible file. chriek may also keep using the deleted file without any error: its jobs, including ones accepted after the deletion, are lost when it stops, and the next start begins with an empty database.

chriek never changes permissions or ownership of anything that already exists: not your data, not the directories above it. A bare filename such as `CHRIEK_DB_PATH=chriek.db` puts the database in the working directory, which is usually shared (`0755`) and is refused; point it into a dedicated directory such as `chriek-data/chriek.db` instead. With `CHRIEK_DB_PATH=:memory:` there is no database file, and `CHRIEK_STORAGE_DIR` is checked and protected the same way.

A group-writable ancestor is refused even when the group is private to you: mode bits can't tell a private group from a shared one. Under umask `002` (the default for user-private groups on some Linux distributions) your own project directory is `0775`, so chriek refuses the default `chriek-data/` inside it. Configure dedicated paths under an ancestor only you or root can write, such as a directory in your home created with `mkdir -m 700`, or remove group write from your own directory (`chmod g-w <dir>`) if no one else relies on it.

**Migrating an existing installation.** Earlier versions created directories and files with the process umask, so under the common umask `022` directories were `0755` and files `0644`, and chriek refuses to start on their data. Stop chriek, then as the service user make only chriek's dedicated directories private and restart:

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

Changing modes or moving a file doesn't revoke descriptors opened earlier: a process of another user that opened `chriek.db` or a sidecar before the migration can keep reading it, including jobs stored afterwards, until it closes the file. Only new private files keep future writes away from such a process; chriek doesn't detect old descriptors or create those files for you.

If old descriptors are a concern, after the permission steps above and before restarting, keep every process using the database stopped and, as the service user, create a new private directory (`mkdir -m 700 <new-directory>`). Copy the database and each existing sidecar named in the loop above into it as new files (not moves or hard links), then remove group and other permissions from those copies before setting `CHRIEK_DB_PATH` to the copied database. Copy the images the same way into a new private image root and make the copies private, never a shared parent:

```sh
mkdir -m 700 <new-image-root>
cp -R <old-image-root>/. <new-image-root>/
chmod -R go= <new-image-root>
```

Then set `CHRIEK_STORAGE_DIR` to the new root and restart. Never resume writes to the old files. This keeps WAL-only data with its database while future writes use new inodes; it cannot retract data already read. Once chriek has started on the copies and your jobs and downloads are there, delete the old database files (by their exact names, as in the loop above) and the old image root to remove the obsolete paths. Deleting them does not revoke descriptors already open on the old files.

A configured path that is a symbolic link is refused, also when its target doesn't exist yet; configure its real target instead. Data owned by another user must be copied or `chown`ed by an administrator before chriek will use it.

chriek removes `..` from `CHRIEK_DB_PATH` and `CHRIEK_STORAGE_DIR` before following links in them, so `alias/../images` names `images` beside `alias`, while earlier versions followed `alias` first and used `images` beside its target; the database file name itself is still used literally. Before upgrading with such a path, set `CHRIEK_DB_PATH` to the real path of the existing database file and `CHRIEK_STORAGE_DIR` to the real directory the existing images are in, so they stay where they are; otherwise chriek starts on a new, empty database.

**Limits.** Mode bits keep other local users out; they don't protect against root, against other processes running as the same user, or against access granted through ACLs, including ACLs a new file inherits from its directory (chriek neither inspects nor manages ACLs, and mode bits don't revoke them). Checks run only at startup; chriek doesn't watch for permission changes or replaced directories while it runs. Only POSIX filesystems with real ownership and modes are supported: on FAT/exFAT, some network or FUSE mounts, or other filesystems that synthesize modes, the checks can pass without protecting anything.

## API

In `api-key` mode (the default) every `/jobs` request needs `Authorization: Bearer <key>`. A missing or invalid key returns `401` before the request body is read, so even a malformed body gets `401`. A job created with another key returns `404`, exactly like a job that doesn't exist. In `proxy` mode `/jobs` needs no key and never returns `401`; every request acts as the one shared owner ([Behind a reverse proxy](#behind-a-reverse-proxy-chriek_auth_modeproxy)). `GET /health` and image downloads need no key in either mode.

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
| `prompt` | string | required | not blank, no NUL character (`\u0000`), at most `CHRIEK_MAX_PROMPT_CHARS` |
| `image_count` | integer | `1` | 1 to `CHRIEK_MAX_IMAGES` |
| `seed` | integer or null | random | `0 ≤ seed` and `seed + image_count - 1 ≤ 4294967295` |
| `width`, `height` | integer | `1024` | multiples of 16, 16 to `CHRIEK_MAX_DIMENSION`, `width * height ≤ CHRIEK_MAX_PIXELS` |
| `references` | 0..N image files (multipart only) | none | at most `CHRIEK_MAX_REFERENCES` files (PNG, JPEG or WebP); each decoded as an image with `width * height ≤ CHRIEK_MAX_PIXELS`; total raw request body at most `CHRIEK_MAX_UPLOAD_BYTES` |

Types are strict (`"1024"` or `1024.0` is not an integer) and unknown fields are rejected. Nonfinite numbers (`NaN`, `Infinity`, `1e999`) and prompts with unpaired UTF-16 surrogates (`"\ud800"`) are rejected too. Requests that fail validation, including malformed JSON syntax such as `{`, return `422` with a `{"detail": [{"type": ..., "loc": [...], "msg": ...}]}` error list. Other JSON decoding failures, such as a byte `0xFF` inside an otherwise UTF-8 JSON string or an integer literal thousands of digits long, return `400 {"detail": "There was an error parsing the body"}` instead; chriek JSON-decodes only bodies sent as `application/json` or `application/*+json`. The JSON decoder also accepts UTF-16 and UTF-32; the request byte cap below applies to those encodings too. A body without a `Content-Type` or of any other type, such as `text/plain` (even invalid bytes), is not decoded and fails validation with `422`. Neither kind of rejected request enters the queue. A valid request returns `202`, a `Location: /jobs/<id>` header and the job (below) in status `queued`. Job IDs are UUIDv7. When `seed` is omitted, chriek picks one at random and reports it in `params.seed`; image `i` uses seed `seed + i`.

In `multipart/form-data`, scalar fields use the same names with decimal integer strings (`seed=` empty string or omitted chooses a random seed), and reference images are uploaded as repeated `references` parts in order, with or without the optional `filename` parameter: every `references` part is decoded as an image whatever its filename or `Content-Type`; only PNG, JPEG and WebP images are accepted, and each is re-encoded as PNG (not stored as the original uploaded bytes) at `references/<index>.png` inside the job's storage directory before the job enters the queue, while client filenames are ignored. Normalization: only one picture is kept: for PNG the default image (an APNG's first frame, or its separate default image when it has one), for animated WebP the first frame, and for JPEG the first picture (Pillow's JPEG decoder also reads multi-picture MPO files from cameras; later pictures are ignored). The image is rotated or flipped by its EXIF orientation tag, or by an XMP `tiff:Orientation` when EXIF has no orientation (when JFIF gives no DPI, Pillow's JPEG reader also reads the EXIF resolution tags; their values are dropped), and converted to RGB without colour management: pixel values pass through unchanged (CMYK and greyscale sources use Pillow's plain conversion; 16-bit greyscale is scaled to 8 bits), so an embedded ICC profile is dropped, not applied, even though it described those values. Any alpha channel and any PNG `tRNS` transparency is discarded (transparent pixels keep their colour values and become opaque). The stored PNG carries no metadata: ICC profiles, EXIF (including the orientation already applied), XMP, comments, text and DPI are all dropped. JPEG uploads are checked before decoding, because their structure can cost far more memory and CPU than their pixels: between SOI and the first EOI at most 256 markers (restart markers inside scan data don't count), at most 64 scans and at most 256 `0xFF` fill bytes in total before markers outside scan data, that is after SOI, a marker segment or a bare RST or TEM marker (fill bytes between scan data and the next marker are skipped with the scan data and not counted); exceeding a bound returns `422 reference image <i> has too complex a JPEG structure …`. Stray bytes between segments, a nested SOI, a reserved JPG or JPGn marker (`0xFFC8`, `0xFFF0` to `0xFFFD`, which encoders don't write), a segment length below 2, an EOI before the first scan and a file ending before EOI return `422 … is not a valid image`. Data after the first EOI (further MPO pictures) may be buffered but is never parsed or decoded. PNG uploads are checked the same way: up to IEND, before or after the image data, at most 256 chunks in total, not counting `IDAT` chunks of at least 1024 bytes (so `IHDR`, `IEND` and every `IDAT` chunk shorter than 1024 bytes count, and so do an APNG's `acTL`, every `fcTL` and every `fdAT` chunk, so about 127 frames is an upper ceiling for APNGs, not a guarantee), and of those at most 8 `iCCP`, `zTXt` or `iTXt` chunks with a nonzero compression flag, whatever their keyword or content. Encoders writing the usual 8 KiB or larger `IDAT` chunks (plus a shorter final one) are fine; encoders splitting the pixel data into many chunks under 1024 bytes are rejected. Exceeding a bound returns `422 reference image <i> has too many PNG chunks …`; the bounds are not configurable. They limit how many chunks Pillow reads and inflates, not what each costs: every counted compressed chunk may still inflate up to Pillow's own 1 MiB per-chunk ceiling while decoding. WebP uploads are walked too: at most 256 chunks in total, whatever their type or size, counting every top-level chunk, every `ANMF` animation frame and every chunk inside a frame. An animation spends at least two chunks per frame (its `ANMF` and its `VP8 ` or `VP8L` image data, plus `ALPH` for lossy frames with alpha) besides `VP8X` and `ANIM`, so about 127 frames is an upper ceiling for animated WebPs, not a guarantee, and longer animations are rejected. Exceeding the bound returns `422 reference image <i> has too many WebP chunks …`; it is not configurable. Before decoding, each reference's raw EXIF payload is also capped at 65,536 bytes (64 KiB), counted over every occurrence, not only the one Pillow uses: in JPEG the APP1 `Exif` segments up to the first EOI, without their 6-byte `Exif\0\0` headers; in PNG every `eXIf` chunk, before or after the image data, plus the text of `tEXt`, `zTXt` and `iTXt` chunks keyed `exif` or `Raw profile type exif` (decompressed for `zTXt` and for `iTXt` with any nonzero compression flag, so hex-encoded raw profiles count at their text size); in WebP every `EXIF` chunk, top-level or inside an animation frame (`ANMF`), where libwebp reads it too. A larger total returns `422 reference image <i> has too much EXIF metadata …`, so images with unusually large EXIF blocks (for example big maker notes or embedded thumbnails beyond 64 KiB) must have their metadata stripped before upload; the cap is not configurable. Multipart requests that fail validation, including unknown fields, `references` parts that are not images (a URL sent as a `references` value is just bytes that fail decoding, never fetched), unsupported or unreadable images, and count or byte limit violations, return `422` with the same error list and never enter the queue.

Multipart text parts have fixed bounds too, not configurable: at most 32 per request, counting every part not named `references` (unknown and repeated names included; `references` parts count against `CHRIEK_MAX_REFERENCES` instead, and a part with a `filename` under any other name is rejected), each at most 1,048,576 bytes (1 MiB). Exceeding either returns `422` while the request is still being received.

chriek parses multipart bodies itself rather than through Starlette's `Request.form()`, which spools file parts to temporary files and answers malformed bodies with `400`: its own parser reads the body only after authentication, stops reading at the first chunk that crosses `CHRIEK_MAX_UPLOAD_BYTES`, the reference count or a text-part bound, and answers malformed or over-limit bodies with `422`. The error's `loc` is `["body"]` for multipart framing errors and the `CHRIEK_MAX_UPLOAD_BYTES` cap, or names the field or the reference index where one applies (`["body", "references", 1]`).

The multipart body in flight is held in RAM, not spooled to disk: each part accumulates in a buffer and is copied once when it ends, so a request's upload buffering peaks at roughly twice `CHRIEK_MAX_UPLOAD_BYTES`, before any image decoding allocates more. That figure is per request and adds up across concurrent uploads; budget memory for it, especially before raising `CHRIEK_MAX_UPLOAD_BYTES`. It is not a ceiling on the process's total memory.

Two fixed byte budgets apply; neither is configurable:

- **Request body: at most 2,097,152 bytes (2 MiB).** After the key check, both authenticated routes (`POST /jobs` and `GET /jobs/{id}`) reject a `Content-Length` above the limit with `413 {"detail": "request body must be at most 2097152 bytes"}` before any body is read. When `POST /jobs` reads a body, chriek also counts the bytes it actually receives from the server, for example a chunked (`Transfer-Encoding: chunked`) body, and answers `413` at the first chunk that crosses the limit, without reading on. `GET /jobs/{id}` does not read a body. HTTP framing is the server's job: with a declared `Content-Length`, the server hands chriek at most that many bytes, and anything sent after them is not part of the request. A `POST /jobs` body of exactly 2 MiB is validated normally. The budget is a byte limit, independent of `CHRIEK_MAX_PROMPT_CHARS` (which counts characters): a UTF-8 JSON body with a prompt at the highest allowed setting, 100,000 characters, fits even when every character is written as `\uXXXX` escapes. `multipart/form-data` bodies of `POST /jobs` are exempt: `CHRIEK_MAX_UPLOAD_BYTES` bounds them instead, counting received bytes and answering `422 … upload exceeds maximum size of <n> bytes`; it covers the whole body including multipart framing, so a small setting can reject a prompt below `CHRIEK_MAX_PROMPT_CHARS`. A `GET /jobs/{id}` declaring a `multipart/form-data` body keeps the 2 MiB `Content-Length` check.
- **`422` response: at most 16,384 bytes (16 KiB).** Error entries carry only `type`, `loc` and `msg`; the rejected input and the validation error's `ctx` are never echoed. Each text value, including field names you sent that show up in `loc`, is first clipped to 200 characters; unpaired surrogates in it are then written as escape text such as `\ud800`, which may make it longer. Locations retain at most eight parts. Errors that no longer fit are dropped from the end, so the list may be incomplete.

An early rejection can leave the request body unread. Clients that finish uploading before reading the response may see a broken pipe or connection reset instead of the `401` or `413` response.

Once the worker has [stopped on an unsaved result](#job-behaviour), a request that passes authentication, the body-size limit and validation returns `503 {"detail": "worker stopped: a job result could not be saved; restart chriek"}` instead of `202`: no `Location` header, no job stored, nothing run. Authentication (`401`), the body-size limit (`413`) and validation (`422`, `400`) are checked first and keep their answers. The check sees the worker state at the time of the request, so a job accepted just before a result fails to save stays `queued` until restart. This `503` takes precedence over a full-queue `429`.

A valid request whose owner already has `CHRIEK_MAX_QUEUED_JOBS` queued jobs returns `429` instead and is not queued; see [Queue limit](#queue-limit). Validation comes first, so an invalid request gets `422` or `400` even when the queue is full. This holds for every transport: JSON, multipart without references and edit jobs with references alike. A rejected edit job's references are decoded and validated, then removed again: no job, no association and no reference files remain.

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
- `params.reference_count` is present in submit and poll responses at every status. It is response-only and is `0` for text-to-image jobs, including jobs created before reference uploads were supported.
- `run` records `backend`, `model`, `revision` and `render_seconds` (wall time including model load and PNG writes) when the worker finishes the job, whether it succeeded, an image failed or a stored reference could not be loaded. MLX also reports `load_seconds` (the initial pipeline load paid by this job; `0.0` after loading), `peak_memory_bytes` (MLX allocation peak for this job, not system RAM use; `null` if the counter could not be reset or read), `hardware` (`chip`, `gpu_cores`, `memory_gb`, `macos_version`), `runtime` (`flux3_mlx`, `mlx` versions), `precision` (`BF16`) and `steps` (`16`), except after a reference load failure, when the backend never ran. A job failed by restart recovery or by the running-time limit has `run: null`; see [Job behaviour](#job-behaviour).
- `images` is filled only when `succeeded`. Every poll mints fresh URLs that expire `CHRIEK_URL_TTL_SECONDS` later: chriek's signed `/images/…` URLs with local storage, presigned bucket URLs with [S3 storage](#s3-storage).
- `error` is `null` unless the job `failed`. A failed image is reported as exactly `image N of M failed` (N is the one-based image that failed, M the requested `image_count`); a stored reference image the worker can't load (for example one deleted from `CHRIEK_STORAGE_DIR`) as exactly `reference N of M could not be loaded`; a job failed by restart recovery or by the running-time limit carries the interrupted or timed-out message below. The error never includes exception text, classes or other backend details: the server log keeps the full traceback with the job ID and the zero-based image index, or for a reference that can't be loaded the zero-based reference index.

### `GET /images/{id}/{index}.png?expires=…&sig=…`

Local storage only; with [S3 storage](#s3-storage) download URLs point at the bucket and this route returns `404`. Downloads one PNG without any API key: the HMAC signature covers exactly that job, image index and expiry. A wrong or missing signature returns `403 invalid signature`; an expired URL returns `403 link expired; poll the job again`. URLs start with `CHRIEK_PUBLIC_URL`, never with the host the request came in on; see [Public URL and reverse proxies](#public-url-and-reverse-proxies).

```sh
curl -s -o image0.png "$URL"
```

### `GET /health`

Returns `200 {"status": "ok"}`. If the worker could not save a job's result (succeeded or failed) to the database, it stops and this returns `503 {"status": "error", "detail": "worker stopped: a job result could not be saved; restart chriek"}`; see [Job behaviour](#job-behaviour).

## Job behaviour

- **Queue.** Jobs are stored in the database (SQLite or Postgres) and run one at a time, oldest first, by a single worker thread in the `chriek serve` process. There is one global queue across all owners: no round-robin or per-owner fairness. If owner A submits A1 and A2 before owner B submits B1, both A jobs run before B1; an A3 submitted after B1 runs after B1, even if A1 and A2 finish first. An admitted job is delayed only by jobs submitted before it, never by later ones, but chriek promises no maximum wait: a long or stalled earlier job holds up everything behind it.
- **Admission cap.** Each owner may have at most `CHRIEK_MAX_QUEUED_JOBS` queued jobs; see [Queue limit](#queue-limit). The cap does not bound how long jobs run or how long another owner's job waits, and is no SLA.
- **All or nothing.** If any image of a job fails, the job fails and the images it already rendered are discarded. A job is never partly successful.
- **Restarts.** With a database file or Postgres, queued jobs survive a restart and run afterwards, still `queued` until the worker claims them. A job that was `running` when the process stopped is marked `failed` on the next start (`interrupted: the server stopped while this job was running`) and isn't retried. Jobs already `succeeded` or `failed`, including those failed by the running-time limit, keep their state and error. An [in-memory database](#in-memory-database) starts empty instead.
- **Running-time limit.** Once `started_at` is more than `CHRIEK_JOB_TIMEOUT_SECONDS` (default 3600) in the past, the owner's `GET /jobs/{id}` or the worker's completion path can mark a still-`running` job `failed` (`timed out: the job ran longer than CHRIEK_JOB_TIMEOUT_SECONDS`, `run: null`, no images). A timed-out job isn't retried. The worker checks the time after writing the PNGs on success, including when nobody has polled, and likewise when an image or reference fails. The exact limit does not count as overdue. The first terminal write wins: an overdue poll can record a timeout before the worker saves its result. Timed-out images are never downloadable; if the worker's database write fails, files may remain on disk. A timed-out edit job keeps its `reference_count` and its stored reference PNGs. The timeout does not cancel the render, so later jobs stay `queued` until the worker can continue or the service restarts. The same limit applies to SQLite (file or in-memory) and Postgres.
- **Polling during write failures.** If an overdue owner poll reads the job but SQLite's write lock prevents it from recording the timeout, the poll logs the error and returns `200` with the last read status (`running`). A later overdue poll can retry. Not all write failures receive this fallback: an overdue poll can return `500` if Postgres becomes read-only while chriek is running.
- **Durability covers a crash of the chriek process only.** Reference and output files are not `fsync`ed, so after a power loss or operating system crash a recorded job can lack files or have truncated ones: a queued edit job then fails (`reference N of M could not be loaded`). chriek makes no power-loss guarantee in home mode. If an error occurs during database `COMMIT` (for example, a connection is lost) or while closing the session after `COMMIT`, chriek returns `500` and keeps the uploaded references: the job may already be committed and can still run. If no job was committed, the `<id>/references/` directory can remain with no job until you delete it (see retention below). chriek removes a submission's references on definite pre-commit failures, such as a reference write failure, a queue (`429`) rejection or a database error before `COMMIT`. There is no retry or reconciliation of unknown outcomes.
- **Reference table upgrade.** Reference associations live in the table `job_references`. SQLite databases from earlier drafts of this feature, whose table was named `references`, are renamed at startup, keeping every job, association and order. A SQLite database that has both tables does not start (storage error, exit status 2); merge or remove one of them first. Postgres databases are never renamed: those drafts predate Postgres support.
- **Unsaved results stop the worker.** If saving a finished job's result to the database fails, the worker logs the error and stops instead of moving on: that job stays `running` (until a poll after the running-time limit fails it as timed out), no later job runs, jobs accepted before the failure stay `queued`, new submissions are rejected with `503` (see [`POST /jobs`](#post-jobs)), and `GET /health` returns `503`. Fix the database or disk problem and restart `chriek serve`; restart recovery then fails the stuck job as interrupted if it is still `running`, the queued jobs run, and submissions are accepted again. Rejected submissions were never stored, so they don't run after the restart.
- **AI disclosure in every generated PNG**, always on: PNG text chunks `AI-Generated: true`, `Model`, `Model-Revision`, `Seed` and `Description`, plus XMP (`XML:com.adobe.xmp`) with IPTC `Iptc4xmpExt:DigitalSourceType` = `http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia`. With the fake backend, `Model` is `chriek-fake-backend` and `Model-Revision` is `fake-v1`. Stored reference PNGs are uploads, not outputs, and carry no metadata (see [`POST /jobs`](#post-jobs)).
- No retention: generated files and normalized reference PNGs (including references of failed, timed-out or interrupted jobs) stay in `CHRIEK_STORAGE_DIR` or the bucket until you delete them. Reference PNGs are written before the job row is created, so a crash between the two can leave a `<id>/references/` directory or objects with no job; they also stay until you delete them.

### Queue limit

Each owner may have at most `CHRIEK_MAX_QUEUED_JOBS` jobs (default `100`, configurable 1–100000; there is no unlimited setting) in status `queued` at once. Earlier versions had no limit: after upgrading, an owner who already has 100 or more queued jobs gets `429` for new submissions until the queue drains below the limit, and existing queued jobs are kept and still run. Raise `CHRIEK_MAX_QUEUED_JOBS` if 100 is too low. Only `queued` jobs count: once the worker starts a job (`running`) it frees its slot, and `succeeded` or `failed` jobs never count. A submission that would exceed the limit returns `429 {"detail": "queue full: at most <n> queued jobs per owner; new jobs are accepted only while fewer are queued"}`; the rejected job is not stored and never runs. A full queue frees up only while the worker runs; if it has stopped (`GET /health` returns `503`), retrying doesn't help until chriek restarts. Each API key has its own limit; in `proxy` mode all clients share one. With a database file, a queue that is already full is rejected by a read alone, without waiting for SQLite's write lock. An in-memory database shares one pooled connection, so that read can wait for another operation to release the connection. Submissions that pass the read take the write lock like any other write, and some are then rejected there: admission counts again inside the same SQLite statement that inserts the job, so concurrent submissions, even through separate database connections, can't exceed the limit. With Postgres, the read that rejects a full queue takes no lock either. Submissions that pass it first take a transaction-level advisory lock for their owner, which only that owner's other submissions wait for, and are then counted again inside the statement that inserts the job, so they can't exceed the limit either; chriek runs its Postgres transactions at `READ COMMITTED` for this, whatever the server's default. Run one `chriek serve` per database: on startup each server fails every `running` job as interrupted, including jobs another live server is rendering.

## Operator obligations under the FLUX Kommunity License

The fake backend uses no FLUX model. chriek is MIT-licensed, but the FLUX 3 Image Klein weights are licensed by Black Forest Labs under the **FLUX Kommunity License v1.0**. chriek never ships or redistributes the weights: you download them with your own Hugging Face account and accept that license yourself. Read the [license text](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/09561231f44e41381c461caf53d5452005a60ac9/LICENSE.md); this summary is not legal advice.

- **Revenue cap.** Commercial use of outputs, and running the model in production to generate them, is allowed only for a "Qualifying User": you together with your affiliates have gross annualized revenue below US$5,000,000 (§1.e, §2.a). Otherwise the license covers only non-commercial purposes, and commercial use needs a separate license from Black Forest Labs (§2.b).
- **Content filtering.** You must implement and maintain content filters that prevent unlawful, unsafe or infringing content, or review outputs for unlawful or infringing content before distributing or displaying them (§2.e(i)). chriek has no built-in filter; this is the operator's job.
- **AI disclosure.** Outputs must carry disclosures that they were generated or modified with AI to the extent applicable law requires (§2.e(ii)). chriek writes the PNG text chunk and IPTC XMP marker above into every generated image; keep them intact and add any disclosure your distribution channel needs.
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

The TLS tests generate throwaway certificates with the `openssl` command, so it must be on `PATH` (OpenSSL or LibreSSL).

Application tests drive chriek through its HTTP API and the installed `chriek` command, with the fake backend. Some check what HTTP can't reach directly: they parse configuration, call the job store, inject store write failures, pause one store's commit while another admits a job, or open the database to read back stored jobs and their owners; SQLite-only ones read its indexes, hold its write lock from another connection, or prepare a database file without a newer index; direct store, worker, app and `chriek.storage` checks cover where storage privacy must hold without the CLI. Another user's ownership is simulated by patching `os.stat`/`os.lstat`, so the suite runs as one user. Proxy tests use chriek's internal shared-owner name to check the owner stored for proxy-mode jobs and that no API key name can own them. Some installed-command tests send raw HTTP requests over a socket for `Host` headers an HTTP client won't send. `tests/test_dependency_audit.py` checks the dependency-audit gate's boundaries offline: injected pip-audit reports, and the real pinned pip-audit with only its HTTP transport answering canned PyPI responses.

The suite runs on SQLite by default. To run it against Postgres as CI does, point `CHRIEK_TEST_POSTGRES_URL` at a disposable server with a superuser (or a role allowed to create databases). Each test creates its own `chriek_test_<random>` database and drops it afterwards; the fixtures drop no other database. Tests marked `sqlite` (in-memory, file-path, SQLite file privacy, write-lock, index-catalog and pre-Postgres schema behaviour) stay on SQLite. Postgres-only cases (a server-dropped connection during a render, client-encoding overrides and a stricter isolation default) are skipped on SQLite. With Docker:

```sh
docker run --rm -d --name chriek-test-postgres -e POSTGRES_PASSWORD=chriek -p 127.0.0.1:55432:5432 postgres:18
until docker exec chriek-test-postgres pg_isready -h 127.0.0.1 -U postgres; do sleep 1; done
uv run pytest && CHRIEK_TEST_POSTGRES_URL=postgresql://postgres:chriek@127.0.0.1:55432/postgres uv run pytest
docker stop chriek-test-postgres   # --rm deletes the container and its data
```

Images go to local storage by default. To run the suite against an S3-compatible service as CI does, set `CHRIEK_TEST_S3_URL=http://<access key>:<secret>@<host>:<port>/<bucket>` (the bucket is created if missing). Each test stores below its own `chriek-test/<random>` prefix and deletes only that prefix's objects afterwards. Tests marked `local_storage` (file privacy, chriek's own download route, local paths) keep local images; `tests/test_s3.py` covers presigned links, their expiry and tampering, and object cleanup, and is skipped without the variable. CI uses RustFS:

```sh
docker run --rm -d --name chriek-test-s3 -e RUSTFS_ACCESS_KEY=chriek -e RUSTFS_SECRET_KEY=chriek-secret-0123456789 -p 127.0.0.1:59000:9000 rustfs/rustfs:1.0.6
until curl -fsS http://127.0.0.1:59000/health; do sleep 1; done
CHRIEK_TEST_S3_URL=http://chriek:chriek-secret-0123456789@127.0.0.1:59000/chriek-test uv run pytest
docker stop chriek-test-s3
```

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

The MLX lock was audited on 2026-11-18 on macOS 27.2.1 arm64 with Python 3.14.6: 73 locked third-party packages, no known vulnerabilities (exit 0).
