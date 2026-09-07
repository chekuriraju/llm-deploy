"""Health endpoint tests.

Uses FastAPI's TestClient, which runs the app in-process (including the
lifespan startup that loads the model) — no live server needed.
"""

from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body
