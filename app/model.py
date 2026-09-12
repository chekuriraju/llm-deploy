"""
Model loader and inference wrapper.

Concept: We load the model + tokenizer once when the module is imported, and keep
them in module-level singletons. The model is 60 MB and takes ~2-5 seconds to load
on first import; we don't want to pay that cost on every HTTP request.

Supports loading from:
1. Model Registry (versioned artifacts) — via MODEL_VERSION env var
2. HuggingFace Hub (direct download) — fallback/default

`flan-t5-small` is an instruction-tuned text-to-text model. You give it a string
prompt, it generates a string response. Examples:
  prompt="Translate to French: hello"  -> "Bonjour"
  prompt="Summarize: <long text>"       -> "<short summary>"
"""

from __future__ import annotations

import logging
import os
from threading import Lock

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.registry import get_registry, ensure_model_registered

logger = logging.getLogger(__name__)

# Model identifier on the HuggingFace Hub. Public, no auth needed.
MODEL_NAME = os.environ.get("MODEL_NAME", "google/flan-t5-small")

# Model version to load from registry. Options:
#   - "stable": load currently promoted version
#   - "latest": load most recently registered version
#   - "v1", "v2", etc.: load specific version
#   - unset/empty (default): load from HF Hub directly (backward compatible)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "")

# Inference defaults — overridable per request.
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", "64"))

# Module-level singletons. Initialized once via `get_model()`.
_tokenizer = None
_model = None
_loaded_version: str | None = None
_load_lock = Lock()


def _load_from_registry(model_name: str, version: str):
    """Load model artifacts from registry."""
    registry = get_registry()

    if version == "stable":
        mv = registry.get_stable_version(model_name)
        if not mv:
            raise RuntimeError(f"No stable version found for {model_name}. Run registration first.")
        version = mv.version
        logger.info("Loading stable version: %s", version)
    elif version == "latest":
        versions = registry.list_versions(model_name)
        if not versions:
            raise RuntimeError(f"No versions registered for {model_name}")
        version = versions[0].version
        logger.info("Loading latest version: %s", version)
    else:
        mv = registry.get_version(model_name, version)
        if not mv:
            raise RuntimeError(f"Version {version} not found for {model_name}")
        logger.info("Loading specific version: %s", version)

    artifacts_path = registry.get_artifacts_path(model_name, version)
    if not artifacts_path or not artifacts_path.exists():
        raise RuntimeError(f"Artifacts not found for {model_name}@{version}")

    logger.info("Loading model from registry: %s", artifacts_path)
    tokenizer = AutoTokenizer.from_pretrained(str(artifacts_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(artifacts_path))
    model.eval()
    return tokenizer, model, version


def _load_from_hf_hub(model_name: str):
    """Load model directly from HuggingFace Hub (original behavior)."""
    logger.info("Loading model from HF Hub: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model, "hf_hub"


def get_model():
    """Lazy, thread-safe model + tokenizer loader with version support.

    Loading priority:
    1. If MODEL_VERSION set: load from registry (stable/latest/specific)
    2. Else: load from HF Hub directly (backward compatible)

    First call downloads/copies weights and instantiates the model in memory.
    Subsequent calls return the cached objects instantly.
    """
    global _tokenizer, _model, _loaded_version

    # Check if already loaded with correct version
    if _model is not None and _tokenizer is not None:
        if MODEL_VERSION and _loaded_version != MODEL_VERSION:
            logger.warning(
                "Model already loaded with version %s, but MODEL_VERSION=%s. "
                "Restart process to change version.",
                _loaded_version, MODEL_VERSION
            )
        return _tokenizer, _model

    with _load_lock:
        # Double-checked locking
        if _model is not None and _tokenizer is not None:
            return _tokenizer, _model

        try:
            if MODEL_VERSION:
                # Load from registry
                _tokenizer, _model, _loaded_version = _load_from_registry(MODEL_NAME, MODEL_VERSION)
            else:
                # Backward compatible: load from HF Hub
                _tokenizer, _model, _loaded_version = _load_from_hf_hub(MODEL_NAME)

            logger.info("Model %s loaded successfully (version: %s)", MODEL_NAME, _loaded_version)
            return _tokenizer, _model

        except Exception as e:
            logger.exception("Failed to load model")
            # Clear partial state on failure
            _tokenizer = None
            _model = None
            _loaded_version = None
            raise RuntimeError(f"Model loading failed: {e}") from e


def get_loaded_version() -> str | None:
    """Return the version string of the currently loaded model."""
    return _loaded_version


def generate(prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
    """Run inference on a single prompt. Returns the generated text.

    Args:
        prompt: The instruction or question to send to the model.
        max_new_tokens: Cap on how much the model can generate (bounds latency).

    Returns:
        The model's text response, with leading/trailing whitespace stripped.
    """
    tokenizer, model = get_model()

    # Tokenize: convert the string to the integer IDs the model operates on.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Generate: the model produces a sequence of token IDs.
    # `do_sample=False` makes output deterministic (greedy decoding).
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,  # Greedy. num_beams>1 would do beam search (slower, often better).
    )

    # Decode: convert the IDs back to a human-readable string.
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()
