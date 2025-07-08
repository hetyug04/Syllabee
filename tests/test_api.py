# tests/test_api.py
from fastapi.testclient import TestClient
from planner import app

client = TestClient(app)

def test_root_returns_404():
    """
    Ensures the root path returns a 404 Not Found error,
    as no endpoint is defined for it.
    """
    response = client.get("/")
    assert response.status_code == 404
