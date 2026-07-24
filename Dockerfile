# Reproducible CPU-only image for ilmarinen.
#   docker build -t ilmarinen .
#   docker run --rm ilmarinen                      # runs the integrity self-check
#   docker run --rm ilmarinen python -m pytest tests_unit/ -m "not smoke"
#
# Dependencies are pinned via requirements.lock (Linux + CPU torch); the package
# itself is installed with --no-deps so nothing drifts off the lock. Dataset
# extras (rdkit, torchvision, ...) are intentionally NOT included — this is the
# core image (import ilmarinen, build an AllGraph, run a fit on in-memory data).
FROM python:3.12-slim

WORKDIR /app

# 1) Pinned dependencies first, for cacheable layers.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# 2) The package (deps already satisfied by the lock).
COPY . .
RUN pip install --no-cache-dir --no-deps .

CMD ["python", "-m", "ilmarinen._selfcheck"]
