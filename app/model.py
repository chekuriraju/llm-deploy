"""
Model loader and inference wrapper.

Concept: We load the model + tokenizer once when the module is imported, and keep
them in module-level singletons. The model is 60 MB and takes ~2-5 seconds to load
on first import; we don't want to pay that cost on every HTTP request.

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

logger = logging.getLogger(__name__)

# Model identifier on the HuggingFace Hub. Public, no auth needed.
MODEL_NAME = os.environ.get("MODEL_NAME", "google/flan-t5-small")

# Inference defaults — overridable per request.
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", "64"))

# Module-level singletons. Initialized once via `get_model()`.
_tokenizer = None
_model = None
_load_lock = Lock()


def get_model():
    """Lazy, thread-safe model + tokenizer loader.

    First call downloads weights (~60 MB) and instantiates the model in memory.
    Subsequent calls return the cached objects instantly.
    """
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model

    with _load_lock:
        # Double-checked locking: another thread may have loaded while we waited.
        if _model is not None and _tokenizer is not None:
            return _tokenizer, _model

        logger.info("Loading model %s ...", MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        # `eval()` disables dropout, which is what we want for inference.
        _model.eval()
        logger.info("Model %s loaded successfully", MODEL_NAME)
        return _tokenizer, _model


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
