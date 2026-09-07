"""Model loading tests.

Verifies the singleton loader returns a usable tokenizer + model and that
repeated calls return the cached instances (no reload).
"""

from app.model import generate, get_model


def test_model_loads():
    tokenizer, model = get_model()
    assert tokenizer is not None
    assert model is not None
    # eval() mode: dropout disabled for inference
    assert not model.training


def test_model_is_cached():
    tokenizer1, model1 = get_model()
    tokenizer2, model2 = get_model()
    assert tokenizer1 is tokenizer2
    assert model1 is model2


def test_generate_returns_text():
    output = generate("Translate to French: hello", max_new_tokens=16)
    assert isinstance(output, str)
    assert len(output) > 0
