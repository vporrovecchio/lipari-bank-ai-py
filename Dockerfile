FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Solo i manifest: questo layer si invalida solo se cambiano le dipendenze
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Il codice dopo: cambia a ogni commit, e invalida solo da qui in giù
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN uv sync --frozen --no-dev

# ── Fase 2: esecuzione ───────────────────────────────────────────
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 lipari
WORKDIR /app

COPY --from=builder --chown=lipari:lipari /app /app
ENV PATH="/app/.venv/bin:$PATH"

USER lipari
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["gunicorn", "src.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "2", "--bind", "0.0.0.0:8000", \
     "--max-requests", "1000", "--max-requests-jitter", "100", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-"]