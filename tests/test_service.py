import pytest
from fastapi.testclient import TestClient
from rag_intelligence.service import create_app

def test_auth_persistence_and_tenant_guard(tmp_path):
    path=str(tmp_path/"rag.db")
    client=TestClient(create_app(path=path,api_key="secret"))
    assert client.post("/v1/documents",json={"source":"a","text":"hello"}).status_code==401
    headers={"x-api-key":"secret"}
    response=client.post("/v1/documents",headers=headers,json={"source":"manual.md","text":"Kubernetes schedules pods"})
    assert response.status_code==200
    again=TestClient(create_app(path=path,api_key="secret"))
    hits=again.post("/v1/search",headers=headers,json={"question":"Kubernetes"})
    assert hits.json()["citations"][0]["source"]=="manual.md"
    assert again.post("/v1/search",headers={**headers,"x-tenant-id":"wrong"},json={"question":"Kubernetes"}).status_code==403

def test_refuses_missing_key(tmp_path):
    with pytest.raises(RuntimeError):
        create_app(path=str(tmp_path/"db"),api_key="")
