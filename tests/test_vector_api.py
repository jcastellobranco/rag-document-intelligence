from uuid import uuid4
from fastapi.testclient import TestClient
from rag_intelligence.vector_api import create_app
from rag_intelligence.vector_store import RetrievalHit

class FakePipeline:
    def __init__(self):
        self.calls=[]
    def ingest(self, tenant, source, text):
        self.calls.append(("ingest",tenant,source,text))
        return {"document_id":str(uuid4()),"chunks":1}
    def search(self, tenant, question, limit):
        self.calls.append(("search",tenant,question,limit))
        return [RetrievalHit("c1", "guide.md", "Kubernetes schedules pods", .02, "hybrid_rrf")]
    def list_documents(self, tenant):
        self.calls.append(("list", tenant))
        return [{"document_id":str(uuid4()),"source":"guide.md","chunks":1,"created_at":"2026-11-06T00:00:00+00:00"}]
    def delete_document(self, tenant, document_id):
        self.calls.append(("delete",tenant,document_id))
        return True

def test_pgvector_http_contract_and_tenant_guard():
    tenant=str(uuid4())
    other=str(uuid4())
    pipeline=FakePipeline()
    client=TestClient(create_app(api_key="secret",tenant_id=tenant,pipeline=pipeline))
    assert client.get("/health").json()["backend"]=="pgvector"
    assert client.post("/v1/search",json={"question":"pods"}).status_code==401
    assert client.post("/v1/search",headers={"x-api-key":"bad"},json={"question":"pods"}).status_code==401
    headers={"x-api-key":"secret","x-tenant-id":other}
    assert client.post("/v1/search",headers=headers,json={"question":"pods"}).status_code==403
    assert pipeline.calls==[]
    headers["x-tenant-id"]=tenant
    ingested=client.post("/v1/documents",headers=headers,json={"source":"guide.md","text":"Kubernetes schedules pods"})
    assert ingested.status_code==200
    response=client.post("/v1/search",headers=headers,json={"question":"pods","limit":2})
    assert response.status_code==200
    citation=response.json()["citations"][0]
    assert citation["source"]=="guide.md"
    assert citation["chunk_id"]=="c1"
    assert citation["method"]=="hybrid_rrf"
    assert all(call[1]==__import__("uuid").UUID(tenant) for call in pipeline.calls)

def test_vector_api_requires_server_identity():
    import pytest
    with pytest.raises(RuntimeError,match="RAG_TENANT_ID"):
        create_app(api_key="secret",tenant_id="invalid",pipeline=FakePipeline())

def test_pgvector_pdf_rejects_bad_mime_and_blank_pages():
    import io
    from pypdf import PdfWriter
    tenant = str(uuid4())
    client = TestClient(create_app(api_key="secret",tenant_id=tenant,pipeline=FakePipeline()))
    headers={"x-api-key":"secret","x-tenant-id":tenant}
    assert client.post("/v1/documents/pdf",headers=headers,
        files={"file":("not-pdf.txt",b"hello","text/plain")}).status_code == 415
    stream=io.BytesIO()
    writer=PdfWriter()
    writer.add_blank_page(width=200,height=200)
    writer.write(stream)
    response=client.post("/v1/documents/pdf",headers=headers,
        files={"file":("blank.pdf",stream.getvalue(),"application/pdf")})
    assert response.status_code == 422

def test_document_catalogue_is_authenticated_and_tenant_bound():
    tenant = str(uuid4())
    pipeline = FakePipeline()
    client = TestClient(create_app(api_key="secret", tenant_id=tenant, pipeline=pipeline))
    assert client.get("/v1/documents").status_code == 401
    other = str(uuid4())
    assert client.get("/v1/documents", headers={"x-api-key":"secret","x-tenant-id":other}).status_code == 403
    headers={"x-api-key":"secret","x-tenant-id":tenant}
    response=client.get("/v1/documents", headers=headers)
    assert response.status_code == 200
    assert response.json()["items"][0]["source"] == "guide.md"
    doc=str(uuid4())
    deleted=client.delete(f"/v1/documents/{doc}",headers=headers)
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert pipeline.calls[0][1] == __import__("uuid").UUID(tenant)
    assert pipeline.calls[1][1] == __import__("uuid").UUID(tenant)
