# Backend API image (FastAPI + uvicorn).
FROM python:3.13-slim

# libgomp1 is the OpenMP runtime scikit-learn links against at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and migration assets (.dockerignore keeps runtime data out,
# but lets the builtin seed CSVs under data/ through).
COPY roughapp/ ./roughapp/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY data/ ./data/

EXPOSE 8000

# Migrations run at app startup (init_db → alembic upgrade head), so the
# container only needs to launch the server.
CMD ["uvicorn", "roughapp.api:app", "--host", "0.0.0.0", "--port", "8000"]
