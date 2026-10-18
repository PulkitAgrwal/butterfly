FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.24 /uv /usr/local/bin/uv
WORKDIR /app

# Optional: behind a TLS-inspecting proxy, pass its CA bundle as the `extra_ca` build secret
# (see EXTRA_CA_FILE in docker-compose.yml). On a normal network this does nothing.
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then cat /run/secrets/extra_ca >> /etc/ssl/certs/ca-certificates.crt; fi
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt UV_NATIVE_TLS=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
# The benchmark data is public (MIT); bake it into the image so startup needs no network.
RUN uv run --no-sync python scripts/fetch_data.py

ENV FALKORDB_HOST=falkordb FALKORDB_PORT=6379 PYTHONPATH=/app
EXPOSE 8000
# Import is idempotent (MERGE), so every container start rebuilds the graphs safely.
CMD ["sh", "-c", "uv run --no-sync python scripts/import_runs.py && uv run --no-sync uvicorn butterfly.api:app --host 0.0.0.0 --port 8000"]
