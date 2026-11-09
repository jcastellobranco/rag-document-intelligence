"""Exact-document reads must be tenant-bound and never return other records."""
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from rag_intelligence.vector_api import create_app


def test_document_content_requires_key_and_tenant_and_exact_uuid():
    tenant, other = str(uuid4()), str(uuid4())
    requested = uuid4()
    calls = []

    class Pipeline:
        def get_document(self, actual_tenant, document_id):
            calls.append((actual_tenant, document_id))
            return {"document_id": str(requested), "source": "target.pdf", "chunks": [
                {"chunk_id": str(uuid4()), "ordinal": 0, "text": "confidencialidade"}
            ]} if document_id == requested else None

    client = TestClient(create_app(api_key="secret", tenant_id=tenant, pipeline=Pipeline()))
    path = f"/v1/documents/{requested}/content"
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"x-api-key": "secret", "x-tenant-id": other}).status_code == 403
    assert calls == []
    headers = {"x-api-key": "secret", "x-tenant-id": tenant}
    ok = client.get(path, headers=headers)
    assert ok.status_code == 200
    assert ok.json()["source"] == "target.pdf"
    assert calls == [(UUID(tenant), requested)]
    assert client.get(f"/v1/documents/{uuid4()}/content", headers=headers).status_code == 404


def test_document_content_rejects_excessive_chunks():
    tenant = str(uuid4())
    class Pipeline:
        def get_document(self, actual_tenant, document_id):
            raise ValueError("too many chunks")
    client = TestClient(create_app(api_key="secret", tenant_id=tenant, pipeline=Pipeline()))
    response = client.get(f"/v1/documents/{uuid4()}/content",
                          headers={"x-api-key": "secret", "x-tenant-id": tenant})
    assert response.status_code == 413
