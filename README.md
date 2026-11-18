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

Today's supported installation remains `uv sync --locked`: only the fake backend is available, with no GPU runtimes or model weights.

Shared runtime dependencies stay in `pyproject.toml` under `[project].dependencies`; development tools stay in `[dependency-groups].dev`. Real backend implementations will add `mlx` and `cuda` extras under `[project.optional-dependencies]`, containing their actual inference dependencies, not just a tensor library.

The future commands `uv sync --locked --extra mlx` and `uv sync --locked --extra cuda` are **not available today**: neither extra exists yet. Extras install dependencies; they neither detect hardware nor select a runtime backend. CUDA wheel source and version configuration belongs with the real CUDA implementation; naming an extra `cuda` does not select a CUDA build.

Keep the committed `uv.lock` as the single lockfile; do not introduce hand-maintained `requirements_mlx.txt` or `requirements_cuda.txt` copies.

## Configuration

All configuration comes from environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `CHRIEK_AUTH_MODE` | `api-key` | `api-key`: every `/jobs` request needs a key from `CHRIEK_API_KEYS`, and each key owns its jobs. `proxy`: no key check; a trusted reverse proxy authenticates, and all clients are one shared owner; see [Behind a reverse proxy](#behind-a-reverse-proxy-chriek_auth_modeproxy). Empty, differently cased or other values stop startup. |
| `CHRIEK_API_KEYS` | required in `api-key` mode | Named static API keys, `name:key,name:key`. The name is the owner of the jobs created with that key. Names: 1–64 characters from `A-Z a-z 0-9 _ . -`, unique. Names are not secret: each job stores its owner's name in plain text, so never use a key or the signing secret as a name. Keys: at least 32 printable ASCII characters (`!` to `~`; no spaces, control or non-ASCII characters, no commas), unique. Keys travel in an HTTP header, so other characters are rejected at startup. Startup errors identify an invalid entry by its position, never by its name or key. Must be unset, empty or whitespace only in `proxy` mode. |
| `CHRIEK_SIGNING_SECRET` | required | HMAC secret for download URLs, at least 32 characters. Changing it invalidates every issued URL (polling the job again mints new ones). Anyone who knows it can sign download URLs, so it must not be one of the API keys: startup rejects a signing secret equal to any key or name in `CHRIEK_API_KEYS`. |
| `CHRIEK_PUBLIC_URL` | required | Public base URL of download links, `http(s)://host[:port][/prefix]`, for example `https://images.example/chriek`. Final `/` characters are optional and ignored. The host is a DNS name, a dotted-decimal IPv4 address such as `192.0.2.10` or a bracketed IPv6 address; a port must be 1–65535. Hosts that browsers would read as a different IPv4 address (`017.0.0.1`, `2130706433`, `0x7f.1`) or as an invalid one (`1.2.3.4.5`) are rejected. The prefix must not contain `\`, `.` or `..` segments (also percent-encoded as `%2e`), which clients rewrite. Only printable ASCII is accepted (no whitespace or control characters; write internationalized host names in punycode), with no user name or password, query (`?`) or fragment (`#`). See [Public URL and reverse proxies](#public-url-and-reverse-proxies). |
| `CHRIEK_URL_TTL_SECONDS` | `900` | Lifetime of a download URL, 1–604800 seconds. |
| `CHRIEK_JOB_TIMEOUT_SECONDS` | `3600` | Longest a job may stay `running`, 1–604800 seconds, counted from `started_at` (time spent `queued` doesn't count). A job running longer fails as timed out; see [Job behaviour](#job-behaviour). |
| `CHRIEK_DATABASE_URL` | unset | Postgres database, `postgresql://user:password@host:5432/dbname` (or `postgresql+psycopg://…`; chriek always uses the psycopg 3 driver). When set, it takes precedence and `CHRIEK_DB_PATH` is ignored. Set but empty, unparsable, with another scheme, with an unencoded `@` other than the one before the host (more than one, or one after the first `/` or `?` following `://`; write every other `@` as `%40`, and a `/` or `?` in the user name or password as `%2F` or `%3F`), or containing a NUL character (also as `%00`) or raw invalid UTF-8 is a configuration error. A parsable URL with parts left out, such as bare `postgresql://`, is accepted: libpq fills them from its defaults and `PG*` variables, and a connection that then fails is a database error. See [Postgres](#postgres). |
| `CHRIEK_DB_PATH` | `chriek-data/chriek.db` | SQLite database file, used when `CHRIEK_DATABASE_URL` is unset. Used literally (`?` and `%20` are ordinary filename characters); missing parent directories are created private. Its parent must be a dedicated directory; see [Storage permissions](#storage-permissions). Put it on a volume to keep jobs across restarts. The exact value `:memory:` (or `./:memory:`) selects an ephemeral in-memory database instead; see [In-memory database](#in-memory-database). |
| `CHRIEK_STORAGE_DIR` | `chriek-data/images` | Dedicated directory for generated PNGs, one subdirectory per job; see [Storage permissions](#storage-permissions). |
| `CHRIEK_HOST` | `127.0.0.1` | Listen address. |
| `CHRIEK_ALLOWED_HOSTS` | `localhost,127.0.0.1,[::1]` in `proxy` mode | `proxy` mode only: comma-separated host names and IP addresses (IPv6 in brackets, as in `[::1]`) that requests must name in their `Host` header; see [Behind a reverse proxy](#behind-a-reverse-proxy-chriek_auth_modeproxy). Exact names only: no wildcards, empty entries, schemes, ports or paths. Setting it in `api-key` mode stops startup. |
| `CHRIEK_PORT` | `8000` | Listen port. |
| `CHRIEK_TLS_CERTFILE` | unset | PEM certificate (chain) for [native TLS](#native-tls). Set together with `CHRIEK_TLS_KEYFILE`; leave both unset for plain HTTP. |
| `CHRIEK_TLS_KEYFILE` | unset | Unencrypted PEM private key matching `CHRIEK_TLS_CERTFILE`. |
| `CHRIEK_MAX_QUEUED_JOBS` | `100` | Maximum `queued` jobs per owner (1–100000); see [Queue limit](#queue-limit). |
| `CHRIEK_MAX_IMAGES` | `4` | Maximum `image_count` per job (configurable 1–64). |
| `CHRIEK_MAX_DIMENSION` | `2048` | Maximum `width` and `height`; a multiple of 16 (16–8192). |
| `CHRIEK_MAX_PIXELS` | `1048576` | Maximum `width * height` (256–67108864). |
| `CHRIEK_MAX_PROMPT_CHARS` | `2000` | Maximum prompt length in characters (1–100000). |

Relative paths resolve against the working directory. The database schema (SQLite or Postgres) is created on first start, and indexes added by newer chriek versions are created in an existing database on start, keeping its jobs; there are no other migrations yet.

### Postgres

Set `CHRIEK_DATABASE_URL` to keep jobs in Postgres instead of a SQLite file:

```sh
export CHRIEK_DATABASE_URL="postgresql://chriek:<percent-encoded password>@db.internal:5432/chriek?sslmode=verify-full"
```

- **Credentials.** The URL carries the database password, so treat the whole variable as a secret: inject it from your secret store or an environment file readable only by the service, not from a command line or a committed file. Percent-encode reserved characters in the user name and password (`%` as `%25`, `@` as `%40`, `:` as `%3A`, `/` as `%2F`, `?` as `%3F`) before putting them into the URL; don't interpolate a raw password. A literal `@` anywhere else (database name, query value) must be `%40` too. Query parameters are percent-decoded and passed to libpq as connection options, so `sslmode`, `sslrootcert` and the like work as in any Postgres client; in query values, unlike in the user name and password, `+` decodes to a space (write `%2B`). Give each parameter only once, and don't use `plugin`, which SQLAlchemy reserves. chriek always connects with client encoding UTF-8, overriding `client_encoding` and `PGCLIENTENCODING`. Use TLS (`sslmode=verify-full`) when the database isn't on the same host or a private network. chriek never logs the variable.
- **Database and role.** Create an empty database with `UTF8` encoding (the default of the official images and of `initdb` in a UTF-8 locale) and a role that owns it (or may create tables in its default schema); chriek creates its `jobs` and `images` tables on first start and never drops anything. Other database encodings are unsupported: a prompt the encoding can't represent gets `500`.
- **Limits of this release.** Postgres replaces only the job database. Run exactly **one** `chriek serve` per database: on startup, each process fails every job still `running` (see [Job behaviour](#job-behaviour)), so a second process would fail the first one's job in progress, and both would run queued jobs at once. PNGs still live in `CHRIEK_STORAGE_DIR` on the local disk of that one process. There is no tool to copy existing jobs from a SQLite file into Postgres. Connections that the server dropped while idle are replaced before use; a database still unreachable when a result is saved, or a connection lost in the middle of saving it, stops the worker (see [Job behaviour](#job-behaviour)).

### In-memory database

`CHRIEK_DB_PATH=:memory:` keeps the job database in memory, meant for tests and local development. No database file is written. The database lives exactly as long as one `chriek serve` process: submissions, the worker and polling share it while the process runs, and every restart starts empty, so job IDs from an earlier run return `404` and their download URLs stop working. The restart guarantees in [Job behaviour](#job-behaviour) (queued jobs survive, running jobs fail as interrupted) hold only for a database file. PNGs still go to `CHRIEK_STORAGE_DIR` on disk and aren't deleted on restart.

### Public URL and reverse proxies

Every download URL chriek returns is `CHRIEK_PUBLIC_URL` followed by `/images/{id}/{index}.png?expires=…&sig=…`; `https://images.example/chriek/` yields `https://images.example/chriek/images/…`. Nothing from the request goes into it: not the `Host` header, `Forwarded`, `X-Forwarded-Host`, `X-Forwarded-Proto`, nor the scheme or port the client connected with. The server (uvicorn) does honour `X-Forwarded-For` and `X-Forwarded-Proto` from trusted proxy addresses, by default only `127.0.0.1` and `::1`, a list the `FORWARDED_ALLOW_IPS` environment variable replaces; that changes the client address and request scheme the server sees, never the link origin.

- **Upgrading.** `CHRIEK_PUBLIC_URL` is new and has no default. Earlier versions built links from the host the client used; a deployment upgraded without setting it no longer starts (exit 2, `CHRIEK_PUBLIC_URL is required`). There is no fallback to the request host. Set it to the address clients use to reach chriek, for example the `http://127.0.0.1:8000` of the [install example](#install-and-run-uv) for local use. No data migration is needed.
- **Scheme.** `http://` suits local use on one machine. For anything reached over a network, use an `https://` URL served by native TLS or a TLS-terminating proxy: download URLs are bearer credentials until they expire.
- **Path prefix.** chriek always serves downloads at `/images/…` on its own listener; it does not strip or know about any prefix. With a prefix such as `/chriek`, your proxy must route `https://images.example/chriek/images/…` to chriek's `/images/…`, removing `/chriek` and keeping the query string unchanged. Without that mapping the links fail instead of returning the image: with the proxy's error, or with chriek's `404 {"detail": "Not Found"}` when the proxy forwards the path without removing the prefix. The `Location: /jobs/<id>` header of a submission carries no prefix either; a prefix proxy must rewrite it for clients that follow that header.

### Storage permissions

Jobs and images are private to the user chriek runs as (the service user) on POSIX filesystems, by mode bits. The database's directory (the parent of `CHRIEK_DB_PATH`) and `CHRIEK_STORAGE_DIR` must be dedicated to chriek and owned by the service user; the defaults (`chriek-data/` and `chriek-data/images/`) are. chriek creates missing directories with mode `0700`, the database with `0600` (SQLite gives its `-wal` and `-shm` sidecars the database's mode, also when it recreates them) and every PNG with `0600` from its first byte, including while a job is still rendering. These creation modes exclude group and others whatever the process umask is, for `chriek serve` and for code that embeds chriek's app, store or worker; ACLs and deleting a live database are separate limits (see below). Working additionally needs a umask that leaves the owner read, write and search rights (such as `022` or `077`): a umask like `0277` makes a first start fail and, on an existing installation, makes every job fail.

With `CHRIEK_DATABASE_URL` set, the database lives on the Postgres server and `CHRIEK_DB_PATH` is ignored: chriek checks and creates only `CHRIEK_STORAGE_DIR` on disk. Protect the database itself with Postgres roles, `pg_hba.conf` and TLS.

Before the database is opened, any job runs or any missing directory is created, `chriek serve` and `create_app` check both configured storage sides and refuse to start if any check fails (`chriek serve` prints `chriek: storage error: <path> <problem>. <guidance>` and exits with status 2 without listening). Used directly, `JobStore` checks only the database side and `Worker` checks only the image side:

- The database's directory and `CHRIEK_STORAGE_DIR`, if they exist: a directory, not a symbolic link, owned by the service user (effective UID), no group or other permission bits (`mode & 077 == 0`).
- The database file and its `-wal`, `-shm` and `-journal` sidecars, if they exist: a regular file, not a symbolic link, owned by the service user, no group or other permission bits. Other files in the database's directory are not checked.
- Every directory and symbolic link followed to reach those two directories, including link targets and chains of links: each directory owned by root or the service user and not writable by group or others unless the sticky bit is set (as on `/tmp`), so dedicated private directories inside `/tmp` pass the checks (but see below on directories that are cleaned automatically); each link owned by root or the service user and inside a directory that is not writable by group or others, sticky or not. A link's owner alone proves nothing: in a sticky directory any user can hard-link a link root owns (as macOS allows), and in a writable one anyone can re-point it. Links such as macOS's `/var` (in the root-owned `/`) are resolved once at startup and chriek uses the real path from then on: re-pointing a link while chriek runs doesn't move its storage before a restart.
- Every file and directory below `CHRIEK_STORAGE_DIR`: not a symbolic link, owned by the service user, no group or other permission bits. A directory chriek can't list fails the check. This includes files chriek didn't write, such as a `.DS_Store`: with chriek stopped, delete that file or remove its group and other permissions (`chmod go= <file>`).

A path chriek can't check (for example below a regular file, inside a directory it can't search, through a symbolic link loop, or through a link whose target continues with `..` after a missing directory) is reported the same way. If a missing directory appears while chriek is creating it, chriek refuses it rather than using a directory it didn't create. The checks cover what already exists, not every later failure: creating the missing parts or opening the database can still fail, for example in a directory chriek can't write, with a file name too long for the filesystem, or with a database SQLite can't open or write. Such an error is reported as a storage error or, for some database errors, as a traceback; nothing is rolled back, so private directories and an empty private database file chriek created so far stay. Stop chriek and correct the cause before restarting; a corrected configuration or umask doesn't repair leftovers. Remove only newly created empty files/directories, or restore owner permissions on chriek's dedicated paths (directories need owner read, write and search; files need owner read and write). Never remove existing job data or change a shared parent.

chriek creates `CHRIEK_STORAGE_DIR` only at startup. If it is deleted while chriek runs, every later job fails (`image 1 of N failed`) for as long as it stays absent, until a restart checks and creates it again. That promise ends if someone re-creates the directory meanwhile: chriek checks only at startup, so it would write later jobs into, and serve downloads from, a directory it never checked; whoever re-created it can make downloads return any file the service user can read. Keep chriek's data in a persistent location, not in a temporary directory that the system cleans automatically (as it may in `/tmp`).

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

```sh
curl -s -X POST http://127.0.0.1:8000/jobs \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "a lighthouse at dusk", "image_count": 2, "seed": 41, "width": 1024, "height": 768}'
```

| Field | Type | Default | Rule |
|---|---|---|---|
| `prompt` | string | required | not blank, no NUL character (`\u0000`), at most `CHRIEK_MAX_PROMPT_CHARS` |
| `image_count` | integer | `1` | 1 to `CHRIEK_MAX_IMAGES` |
| `seed` | integer or null | random | `0 ≤ seed` and `seed + image_count - 1 ≤ 4294967295` |
| `width`, `height` | integer | `1024` | multiples of 16, 16 to `CHRIEK_MAX_DIMENSION`, `width * height ≤ CHRIEK_MAX_PIXELS` |

Types are strict (`"1024"` or `1024.0` is not an integer) and unknown fields are rejected. Nonfinite numbers (`NaN`, `Infinity`, `1e999`) and prompts with unpaired UTF-16 surrogates (`"\ud800"`) are rejected too. Requests that fail validation, including malformed JSON syntax such as `{`, return `422` with a `{"detail": [{"type": ..., "loc": [...], "msg": ...}]}` error list. Other JSON decoding failures, such as a byte `0xFF` inside an otherwise UTF-8 JSON string or an integer literal thousands of digits long, return `400 {"detail": "There was an error parsing the body"}` instead; chriek JSON-decodes only bodies sent as `application/json` or `application/*+json`. The JSON decoder also accepts UTF-16 and UTF-32; the request byte cap below applies to those encodings too. A body without a `Content-Type` or of any other type, such as `text/plain` (even invalid bytes), is not decoded and fails validation with `422`. Neither kind of rejected request enters the queue. A valid request returns `202`, a `Location: /jobs/<id>` header and the job (below) in status `queued`. Job IDs are UUIDv7. When `seed` is omitted, chriek picks one at random and reports it in `params.seed`; image `i` uses seed `seed + i`.

Two fixed byte budgets apply; neither is configurable:

- **Request body: at most 2,097,152 bytes (2 MiB).** After the key check, both authenticated routes (`POST /jobs` and `GET /jobs/{id}`) reject a `Content-Length` above the limit with `413 {"detail": "request body must be at most 2097152 bytes"}` before any body is read. When `POST /jobs` reads a body, chriek also counts the bytes it actually receives from the server, for example a chunked (`Transfer-Encoding: chunked`) body, and answers `413` at the first chunk that crosses the limit, without reading on. `GET /jobs/{id}` does not read a body. HTTP framing is the server's job: with a declared `Content-Length`, the server hands chriek at most that many bytes, and anything sent after them is not part of the request. A `POST /jobs` body of exactly 2 MiB is validated normally. The budget is a byte limit, independent of `CHRIEK_MAX_PROMPT_CHARS` (which counts characters): a UTF-8 JSON body with a prompt at the highest allowed setting, 100,000 characters, fits even when every character is written as `\uXXXX` escapes. Multipart uploads ([#5](https://github.com/stibbons-ai/chriek/issues/5)) will need their own size policy.
- **`422` response: at most 16,384 bytes (16 KiB).** Error entries carry only `type`, `loc` and `msg`; the rejected input and the validation error's `ctx` are never echoed. Each text value, including field names you sent that show up in `loc`, is first clipped to 200 characters; unpaired surrogates in it are then written as escape text such as `\ud800`, which may make it longer. Locations retain at most eight parts. Errors that no longer fit are dropped from the end, so the list may be incomplete.

An early rejection can leave the request body unread. Clients that finish uploading before reading the response may see a broken pipe or connection reset instead of the `401` or `413` response.

Once the worker has [stopped on an unsaved result](#job-behaviour), a request that passes authentication, the body-size limit and validation returns `503 {"detail": "worker stopped: a job result could not be saved; restart chriek"}` instead of `202`: no `Location` header, no job stored, nothing run. Authentication (`401`), the body-size limit (`413`) and validation (`422`, `400`) are checked first and keep their answers. The check sees the worker state at the time of the request, so a job accepted just before a result fails to save stays `queued` until restart. This `503` takes precedence over a full-queue `429`.

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
- `run` is the run record, set once the worker finishes the job: which backend, model and revision rendered it and how long rendering took. The fake backend reports no hardware or memory figures because it measures none. A job failed by restart recovery or by the running-time limit (see [Job behaviour](#job-behaviour)) has `run: null`, because no run record was persisted.
- `images` is filled only when `succeeded`. Every poll mints fresh URLs that expire `CHRIEK_URL_TTL_SECONDS` later.
- `error` is `null` unless the job `failed`. A failed image is reported as exactly `image N of M failed` (N is the one-based image that failed, M the requested `image_count`); a job failed by restart recovery or by the running-time limit carries the interrupted or timed-out message below. The error never includes exception text, classes or other backend details: the server log keeps the full traceback with the job ID and the zero-based image index.

### `GET /images/{id}/{index}.png?expires=…&sig=…`

Downloads one PNG without any API key: the HMAC signature covers exactly that job, image index and expiry. A wrong or missing signature returns `403 invalid signature`; an expired URL returns `403 link expired; poll the job again`. URLs start with `CHRIEK_PUBLIC_URL`, never with the host the request came in on; see [Public URL and reverse proxies](#public-url-and-reverse-proxies).

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
- **Running-time limit.** Once `started_at` is more than `CHRIEK_JOB_TIMEOUT_SECONDS` (default 3600) in the past, the owner's `GET /jobs/{id}` or the worker's completion path can mark a still-`running` job `failed` (`timed out: the job ran longer than CHRIEK_JOB_TIMEOUT_SECONDS`, `run: null`, no images). A timed-out job isn't retried. The worker checks the time after writing the PNGs on success, including when nobody has polled. The exact limit does not count as overdue. The first terminal write wins: an overdue poll can record a timeout before the worker saves its result. Timed-out images are never downloadable; if the worker's database write fails, files may remain on disk. The timeout does not cancel the render, so later jobs stay `queued` until the worker can continue or the service restarts. The same limit applies to SQLite (file or in-memory) and Postgres.
- **Polling during write failures.** If an overdue owner poll reads the job but SQLite's write lock prevents it from recording the timeout, the poll logs the error and returns `200` with the last read status (`running`). A later overdue poll can retry. Not all write failures receive this fallback: an overdue poll can return `500` if Postgres becomes read-only while chriek is running.
- **Unsaved results stop the worker.** If saving a finished job's result to the database fails, the worker logs the error and stops instead of moving on: that job stays `running` (until a poll after the running-time limit fails it as timed out), no later job runs, jobs accepted before the failure stay `queued`, new submissions are rejected with `503` (see [`POST /jobs`](#post-jobs)), and `GET /health` returns `503`. Fix the database or disk problem and restart `chriek serve`; restart recovery then fails the stuck job as interrupted if it is still `running`, the queued jobs run, and submissions are accepted again. Rejected submissions were never stored, so they don't run after the restart.
- **AI disclosure in every PNG**, always on: PNG text chunks `AI-Generated: true`, `Model`, `Model-Revision`, `Seed` and `Description`, plus XMP (`XML:com.adobe.xmp`) with IPTC `Iptc4xmpExt:DigitalSourceType` = `http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia`. With the fake backend, `Model` is `chriek-fake-backend` and `Model-Revision` is `fake-v1`.
- No retention: generated files stay in `CHRIEK_STORAGE_DIR` until you delete them.

### Queue limit

Each owner may have at most `CHRIEK_MAX_QUEUED_JOBS` jobs (default `100`, configurable 1–100000; there is no unlimited setting) in status `queued` at once. Earlier versions had no limit: after upgrading, an owner who already has 100 or more queued jobs gets `429` for new submissions until the queue drains below the limit, and existing queued jobs are kept and still run. Raise `CHRIEK_MAX_QUEUED_JOBS` if 100 is too low. Only `queued` jobs count: once the worker starts a job (`running`) it frees its slot, and `succeeded` or `failed` jobs never count. A submission that would exceed the limit returns `429 {"detail": "queue full: at most <n> queued jobs per owner; new jobs are accepted only while fewer are queued"}`; the rejected job is not stored and never runs. A full queue frees up only while the worker runs; if it has stopped (`GET /health` returns `503`), retrying doesn't help until chriek restarts. Each API key has its own limit; in `proxy` mode all clients share one. With a database file, a queue that is already full is rejected by a read alone, without waiting for SQLite's write lock. An in-memory database shares one pooled connection, so that read can wait for another operation to release the connection. Submissions that pass the read take the write lock like any other write, and some are then rejected there: admission counts again inside the same SQLite statement that inserts the job, so concurrent submissions, even through separate database connections, can't exceed the limit. With Postgres, the read that rejects a full queue takes no lock either. Submissions that pass it first take a transaction-level advisory lock for their owner, which only that owner's other submissions wait for, and are then counted again inside the statement that inserts the job, so they can't exceed the limit either; chriek runs its Postgres transactions at `READ COMMITTED` for this, whatever the server's default. Run one `chriek serve` per database: on startup each server fails every `running` job as interrupted, including jobs another live server is rendering.

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

The TLS tests generate throwaway certificates with the `openssl` command, so it must be on `PATH` (OpenSSL or LibreSSL).

Application tests drive chriek through its HTTP API and the installed `chriek` command, with the fake backend. Some check what HTTP can't reach directly: they parse configuration, call the job store, inject store write failures, pause one store's commit while another admits a job, or open the database to read back stored jobs and their owners; SQLite-only ones read its indexes, hold its write lock from another connection, or prepare a database file without a newer index; direct store, worker, app and `chriek.storage` checks cover where storage privacy must hold without the CLI. Another user's ownership is simulated by patching `os.stat`/`os.lstat`, so the suite runs as one user. Proxy tests use chriek's internal shared-owner name to check the owner stored for proxy-mode jobs and that no API key name can own them. Some installed-command tests send raw HTTP requests over a socket for `Host` headers an HTTP client won't send. `tests/test_dependency_audit.py` checks the dependency-audit gate's boundaries offline: injected pip-audit reports, and the real pinned pip-audit with only its HTTP transport answering canned PyPI responses.

The suite runs on SQLite by default. To run it against Postgres as CI does, point `CHRIEK_TEST_POSTGRES_URL` at a disposable server with a superuser (or a role allowed to create databases). Each test creates its own `chriek_test_<random>` database and drops it afterwards; the fixtures drop no other database. Tests marked `sqlite` (in-memory, file-path, SQLite file privacy, write-lock, index-catalog and pre-Postgres schema behaviour) stay on SQLite. Postgres-only cases (a server-dropped connection during a render, client-encoding overrides and a stricter isolation default) are skipped on SQLite. With Docker:

```sh
docker run --rm -d --name chriek-test-postgres -e POSTGRES_PASSWORD=chriek -p 127.0.0.1:55432:5432 postgres:18
until docker exec chriek-test-postgres pg_isready -h 127.0.0.1 -U postgres; do sleep 1; done
uv run pytest && CHRIEK_TEST_POSTGRES_URL=postgresql://postgres:chriek@127.0.0.1:55432/postgres uv run pytest
docker stop chriek-test-postgres   # --rm deletes the container and its data
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
