FROM python:3.11-slim AS runner

ENV PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM wheels often require OpenMP runtime; keep build tools for arm64/source builds.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Repo layout inside the container mirrors the monorepo:
# /app/prediction_app (code) and /app/raw_data + /app/models (mounted by docker-compose).
WORKDIR /app/prediction_app

COPY prediction_app/api/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# App code + bundled resources (CSV + model assets).
COPY prediction_app /app/prediction_app
COPY raw_data /app/raw_data
COPY models /app/models

# Create mount points (safe even if compose binds volumes over them).
RUN mkdir -p /app/raw_data /app/models /app/uploads

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
