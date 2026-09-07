"""Tests for the POST /generate endpoint."""

from fastapi.testclient import TestClient

from app.main import app


def test_generate_ok():
    with TestClient(app) as client:
        r = client.post(
            "/generate",
            json={"prompt": "Translate to French: hello", "max_new_tokens": 16},
        )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["response"], str)
    assert len(body["response"]) > 0
    assert "model" in body


def test_generate_rejects_empty_prompt():
    with TestClient(app) as client:
        r = client.post("/generate", json={"prompt": ""})
    assert r.status_code == 422  # Pydantic validation error


def test_generate_rejects_missing_prompt():
    with TestClient(app) as client:
        r = client.post("/generate", json={})
    assert r.status_code == 422
