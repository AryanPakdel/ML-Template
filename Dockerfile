# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Serving image for the ML pipeline template.
#
# Two-layer copy strategy:
#   Layer 1 copies ONLY requirements.txt and installs third-party deps.
#     This layer is expensive (torch, xgboost, ...) but changes rarely,
#     so Docker's build cache keeps it across code edits.
#   Layer 2 copies the project source and installs the package itself.
#     Code changes invalidate only this cheap layer, never the deps above.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# libgomp1: OpenMP runtime needed by the lightgbm/xgboost manylinux wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 1: dependencies (cached until requirements.txt changes) ---------
# requirements.txt already carries the torch CPU --extra-index-url line.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Layer 2: project code --------------------------------------------------
# README.md is required: pyproject.toml declares it as the package readme.
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Serve the most recent trained bundle. Mount ./artifacts to /app/artifacts
# (see docker-compose.yml) so bundles live outside the image.
CMD ["mlpipe", "serve", "--bundle", "latest", "--host", "0.0.0.0", "--port", "8000"]
