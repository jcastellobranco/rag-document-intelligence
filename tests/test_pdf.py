import io
import pytest
from fastapi.testclient import TestClient
from rag_intelligence.service import create_app

def test_pdf_ingestion_and_search(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter
    client = TestClient(create_app(path=str(tmp_path/"rag.db"), api_key="secret"))
    headers={"x-api-key":"secret"}
    writer=PdfWriter()
    writer.add_blank_page(width=200,height=200)
    stream=io.BytesIO()
    writer.write(stream)
    response=client.post("/v1/documents/pdf",headers=headers,
        files={"file":("empty.pdf",stream.getvalue(),"application/pdf")})
    assert response.status_code == 422
    assert client.post("/v1/documents/pdf",headers=headers,
        files={"file":("invalid.txt",b"hello","text/plain")}).status_code == 415
    assert client.post("/v1/documents/pdf",headers=headers,
        files={"file":("huge.pdf",b"x"*(10*1024*1024+1),"application/pdf")}).status_code == 413
