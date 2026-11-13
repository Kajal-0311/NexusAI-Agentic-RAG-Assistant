"""
Integration tests for FastAPI REST endpoints using TestClient.
"""
from fastapi.testclient import TestClient
import pytest
from backend.main import app
from tests.conftest import create_sample_pdf_bytes

client = TestClient(app)


def test_endpoint_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "NexusAI" in res.json()["app"]


def test_endpoint_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "ollama_status" in data
    assert "vector_store_status" in data


def test_endpoint_document_flow():
    # 1. Upload
    pdf_bytes = create_sample_pdf_bytes(num_pages=2)
    files = [("files", ("test_paper.pdf", pdf_bytes, "application/pdf"))]
    upload_res = client.post("/documents/upload", files=files)
    assert upload_res.status_code == 201
    docs = upload_res.json()["documents"]
    assert len(docs) == 1
    doc_id = docs[0]["document_id"]

    # 2. List
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    assert any(d["document_id"] == doc_id for d in list_res.json())

    # 3. Get single
    get_res = client.get(f"/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["filename"] == "test_paper.pdf"

    # 4. Delete
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200


def test_endpoint_sessions_flow():
    # 1. Create Session
    create_res = client.post("/sessions", json={"title": "Test Session", "document_ids": []})
    assert create_res.status_code == 201
    session_id = create_res.json()["session_id"]

    # 2. Get Session Detail
    get_res = client.get(f"/sessions/{session_id}")
    assert get_res.status_code == 200
    assert get_res.json()["session"]["title"] == "Test Session"

    # 3. Delete Session
    del_res = client.delete(f"/sessions/{session_id}")
    assert del_res.status_code == 200


def test_endpoint_chat():
    payload = {
        "query": "Hello, what is NexusAI?",
        "retrieval_mode": "hybrid",
        "top_k": 3,
    }
    res = client.post("/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "step_traces" in data
    assert len(data["step_traces"]) > 0


def test_endpoint_evaluation():
    # Dataset
    data_res = client.get("/evaluation/dataset")
    assert data_res.status_code == 200
    assert len(data_res.json()) > 0

    # Results list
    res_res = client.get("/evaluation/results")
    assert res_res.status_code == 200


def test_endpoint_settings():
    res = client.get("/settings")
    assert res.status_code == 200
    assert "chunk_size" in res.json()
