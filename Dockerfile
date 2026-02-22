FROM python:3.10-slim AS base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY med_proj/ med_proj/
COPY artifacts/ artifacts/
COPY config.yaml .

ENV ARTIFACT_DIR=artifacts
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "med_proj.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
