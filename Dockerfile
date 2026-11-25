# CUDA image: chriek with the `cuda` extra. Build stage installs from the committed lockfile;
# the runtime stage gets only the virtual environment. Model weights are mounted at run time.
FROM ghcr.io/astral-sh/uv:0.11.19-python3.14-trixie-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --extra cuda --no-install-project
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra cuda --no-editable

# Same Python as the build stage: the venv's interpreter links to /usr/local/bin/python3.
FROM python:3.14-slim-trixie
RUN useradd --system --uid 10001 --create-home chriek
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CHRIEK_BACKEND=cuda \
    CHRIEK_HOST=0.0.0.0
USER chriek
WORKDIR /home/chriek
ENTRYPOINT ["chriek"]
CMD ["serve"]
