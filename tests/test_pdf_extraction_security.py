"""HTTP regression tests for bounded PDF extraction."""
import sys
from types import ModuleType, SimpleNamespace
from fastapi.testclient import TestClient
from rag_intelligence.vector_api import create_app

TENANT = "11111111-1111-4111-8111-111111111111"

class StubPipeline:
    def __init__(self): self.calls = []
    def ingest(self, tenant, source, text):
        self.calls.append((tenant, source, text))
        return {"ok": True}

def make_client(monkeypatch, texts):
    fake = ModuleType("pypdf")
    fake.PdfReader = lambda *_a, **_k: SimpleNamespace(pages=[
        SimpleNamespace(extract_text=lambda t=t: t) for t in texts])
    monkeypatch.setitem(sys.modules, "pypdf", fake)
    pipeline = StubPipeline()
    return TestClient(create_app(api_key="test", tenant_id=TENANT, pipeline=pipeline)), pipeline

def upload(client):
    return client.post("/v1/documents/pdf", headers={"x-api-key": "test"},
        files={"file": ("test.pdf", b"%PDF-1.7 test", "application/pdf")})

def test_pdf_uses_newlines_between_pages(monkeypatch):
    client, pipeline = make_client(monkeypatch, ["first", "second"])
    assert upload(client).status_code == 200
    assert pipeline.calls[0][2] == "first\nsecond"

def test_pdf_rejects_large_extraction(monkeypatch):
    client, pipeline = make_client(monkeypatch, ["x" * 200000, "y"])
    assert upload(client).status_code == 413
    assert not pipeline.calls

def test_pdf_rejects_page_count(monkeypatch):
    client, pipeline = make_client(monkeypatch, ["x"] * 101)
    assert upload(client).status_code == 413
    assert not pipeline.calls
