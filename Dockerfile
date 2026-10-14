# Serving image: API over the built artifacts. Hansen 30 m tiles are range-read from
# Google Cloud Storage at request time unless HANSEN_DIR points at local copies.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .
COPY artifacts ./artifacts
COPY data/processed ./data/processed
ENV CONGO_EWS_ROOT=/app
RUN useradd --uid 10001 app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "congo_ews.serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
