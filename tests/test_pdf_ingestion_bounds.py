"""PDF extraction limits and text separators through the real HTTP endpoint."""
import sys
from types import ModuleType, SimpleNamespace
from uuid import UUID
from fastapi.testclient import TestClient
from rag_intelligence.vector_api import create_app

TENANT = "11111111-1111-4111-8111-111111111111"

class Pipeline:
    def __init__(self):
        self.ingested = []
    def ingest(self, tenant, source, text):
        self.ingested.append((tenant, source, text))
        return {"ok": True}

def client_for(monkeypatch, pages):
    fake = ModuleType("pypdf")
    fake.PdfReader = lambda *_args, **_kwargs: SimpleNamespace(
        pages=[SimpleNamespace(extract_text=lambda value=value: value) for value in pages]
    )
    monkeypatch.setitem(sys.modules, "pypdf", fake)
    pipeline = Pipeline()
    client = TestClient(create_app(api_key="test-key", tenant_id=TENANT, pipeline=pipeline))
    return client, pipeline

def upload(client):
    return client.post("/v1/documents/pdf", headers={"x-api-key":"test-key","x-tenant-id":TENANT},
                       files={"file":("test.pdf", b"%PDF-1.7\nexample", "application/pdf")})

def test_pdf_pages_separated_by_real_newline(monkeypatch):
    client, pipeline = client_for(monkeypatch, ["first", "second"])
    assert upload(client).status_code == 200
    assert pipeline.ingested[0][2] == "first\nsecond"
    assert pipeline.ingested[0][0] == UUID(TENANT)

def test_pdf_rejects_excess_text_before_ingest(monkeypatch):
    client, pipeline = client_for(monkeypatch, ["x" * 199999, "extra"])
    assert upload(client).status_code == 413
    assert not pipeline.ingested

def test_pdf_rejects_more_than_100_pages(monkeypatch):
    client, pipeline = client_for(monkeypatch, ["x"] * 101)
    assert upload(client).status_code == 413
    assert not pipeline.ingested
