"""Unit tests for server.py Flask API."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def client(monkeypatch):
    """Flask test client with answer() mocked out."""
    import server
    # Patch answer before app is used
    def _fake_answer(question, family_id=None, strategy="section_aware", top_k=5, **kwargs):
        return {
            "answer": "הפרנשייז הוא 3,000 ₪",
            "sources": ["## פרק 3\n\nהפרנשייז..."],
            "strategy": strategy,
            "question": question,
        }
    monkeypatch.setattr(server, "answer", _fake_answer)
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def test_ask_returns_answer(client):
    resp = client.post(
        "/ask",
        data=json.dumps({"question": "מה הפרנשייז?", "strategy": "section_aware"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["answer"] == "הפרנשייז הוא 3,000 ₪"
    assert "sources" in data


def test_ask_missing_question_returns_400(client):
    resp = client.post(
        "/ask",
        data=json.dumps({"strategy": "section_aware"}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_ask_defaults_strategy_to_section_aware(client):
    resp = client.post(
        "/ask",
        data=json.dumps({"question": "מה הפרנשייז?"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.get_json()["strategy"] == "section_aware"


def test_cors_header_present(client):
    resp = client.post(
        "/ask",
        data=json.dumps({"question": "שאלה"}),
        content_type="application/json",
    )
    assert "Access-Control-Allow-Origin" in resp.headers


def test_warmup_returns_warming_up(client, monkeypatch):
    import server
    monkeypatch.setattr(server, "_do_warmup", lambda: None)
    resp = client.post("/warmup")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "warmed_up"

