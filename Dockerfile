# Multi-stage Dockerfile for the LLM deploy demo.
#
# Stage 1 (builder): install Python deps into a venv inside the image
# Stage 2 (runtime): copy only the venv + app code; no compilers, no build tools
#
# Why multi-stage?  A torch install is ~1 GB and pulls in compilers.  By splitting
# the build, the final image carries only what we need to run the app.

# ============================================================
# Stage 1: builder
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps needed by some Python packages (e.g. tokenizers, safetensors).
# In stage 2 we drop these, keeping the final image small.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create a venv.  Copying a venv between stages is a common pattern for
# self-contained Python apps because venvs are fully relocatable.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python deps first.  By doing this before copying app code,
# Docker can cache the install layer: if only app code changes, we skip the
# whole torch/transformers install on rebuild.
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip cache purge

# Now copy the app code
COPY app/ ./app/

# ============================================================
# Stage 2: runtime
# ============================================================
FROM python:3.12-slim AS runtime

# Don't run as root.  Create a dedicated user and own the app dir.
RUN groupadd --system --gid 1001 appuser \
 && useradd  --system --uid 1001 --gid appuser --no-create-home appuser \
 && mkdir -p /app \
 && chown -R appuser:appuser /app
USER appuser
WORKDIR /app

# Copy the built venv from the builder stage.
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
# Copy the app code (also from builder, so we don't pay the cost of a second COPY context).
COPY --from=builder --chown=appuser:appuser /build/app /app/app

# Make the venv the default Python environment.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# HuggingFace caches model weights here.  We point HF_HOME at a directory
# /app owns and create it ahead of time so the non-root appuser can write to it.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chown -R appuser:appuser /app/.cache

EXPOSE 8000

# Healthcheck is informational; Kubernetes uses its own probes (we'll see those
# in k8s/deployment.yaml).  This just helps when running the container bare.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
