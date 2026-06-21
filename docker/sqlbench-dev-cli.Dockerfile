FROM python:3.12-slim AS runtime

ARG INSTALL_GROUPS="mlops"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY datasets ./datasets
COPY experiments ./experiments
COPY flows ./flows
COPY schemas ./schemas

RUN set -eux; \
    if [ -n "${INSTALL_GROUPS}" ]; then \
        sync_args=""; \
        for group in ${INSTALL_GROUPS}; do \
            sync_args="${sync_args} --group ${group}"; \
        done; \
        uv sync --frozen --no-dev ${sync_args}; \
    else \
        uv sync --frozen --no-dev; \
    fi

ENTRYPOINT ["python", "-m", "sqlbench_lab.cli"]
CMD ["--help"]
