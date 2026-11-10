"""Prometheus metrics stay bounded and do not export RAG payloads or tenant IDs."""
from uuid import uuid4
from fastapi.testclient import TestClient
from rag_intelligence.vector_api import create_app
from rag_intelligence.vector_store import RetrievalHit


class FakePipeline:
    def ingest(self, tenant, source, text):
        return {"document_id": str(uuid4()), "chunks": 2}

    def search(self, tenant, question, limit):
        return [RetrievalHit("private-chunk-id", "secret.md", "private-document-text",
                             .02, "hybrid_rrf")]


def test_rag_metrics_scrapeable_and_payload_safe():
    tenant = str(uuid4())
    client = TestClient(create_app(api_key="test-key", tenant_id=tenant, pipeline=FakePipeline()))
    assert client.post("/v1/search", json={"question": "private-question"}).status_code == 401
    headers = {"x-api-key": "test-key", "x-tenant-id": tenant}
    response = client.post("/v1/search", json={"question": "private-question"}, headers=headers)
    assert response.status_code == 200
    assert client.post("/v1/documents", json={"source": "secret.md", "text": "private-document-text"},
                       headers=headers).status_code == 200
    metrics = client.get("/metrics/")
    assert metrics.status_code == 200
    body = metrics.text
    assert 'atlas_rag_http_requests_total{operation="search",status_class="2xx"}' in body
    assert "atlas_rag_search_hits_count" in body
    assert 'atlas_rag_ingested_chunks_total{format="text"}' in body
    for secret in (tenant, "test-key", "private-question", "secret.md",
                   "private-document-text", "private-chunk-id"):
        assert secret not in body


def test_dynamic_document_paths_never_become_metric_labels():
    from rag_intelligence.metrics import operation_for
    assert operation_for("GET", "/v1/documents/secret-id/content") == "read_document"
    assert operation_for("DELETE", "/v1/documents/secret-id") == "delete_document"
    assert "secret-id" not in operation_for("GET", "/v1/documents/secret-id/content")
