"""
FastAPI app exposing a small LLM over HTTP.

Endpoints:
  GET  /health    -> liveness/readiness probe (used by Kubernetes)
  POST /generate  -> runs the model on a prompt, returns generated text

The model is loaded ONCE during the FastAPI lifespan startup. This avoids
paying the model-load cost (seconds, plus 60MB download) on every request.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import MODEL_NAME, generate, get_model

# Standard logging config so we see model-load and access logs in `kubectl logs`.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: runs once at startup, once at shutdown.

    Loading the model here means: by the time the first HTTP request arrives,
    the model is already in memory and ready to serve.
    """
    logger.info("App starting up; loading model...")
    start = time.time()
    get_model()  # Forces model load. Cached for the process lifetime.
    logger.info("Model ready in %.2fs", time.time() - start)
    yield  # <-- App serves requests here.
    logger.info("App shutting down.")


app = FastAPI(
    title="LLM Deploy Demo",
    description="A small language model served via FastAPI, containerized, deployed on Kubernetes.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---- Request / response shapes (Pydantic gives us validation + OpenAPI docs) ----

class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The instruction or question to send to the model.",
        examples=["Translate to French: hello world"],
    )
    max_new_tokens: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Upper bound on generated tokens (also bounds latency).",
    )


class GenerateResponse(BaseModel):
    response: str
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str


# ---- Endpoints ----

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness/readiness probe.

    Kubernetes calls this every few seconds. If it returns 200, the pod is
    considered healthy and gets traffic. If it fails, K8s restarts the pod.
    """
    return HealthResponse(status="ok", model=MODEL_NAME)


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
def generate_text(req: GenerateRequest) -> GenerateResponse:
    """Run the model on a prompt and return the generated text."""
    try:
        start = time.time()
        output = generate(req.prompt, max_new_tokens=req.max_new_tokens)
        elapsed = time.time() - start
        logger.info(
            "prompt=%r max_new_tokens=%d elapsed=%.2fs response=%r",
            req.prompt[:60], req.max_new_tokens, elapsed, output[:60],
        )
        return GenerateResponse(response=output, model=MODEL_NAME)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
