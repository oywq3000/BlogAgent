# -*- coding: utf-8 -*-
"""lifespan 测试：启动触发向量全量重建且异常不阻断服务（sync_vectors 全程 mock，不加载真实 BGE）。"""
from fastapi.testclient import TestClient


def test_lifespan_syncs_vectors_on_startup(monkeypatch):
    called = {}

    async def fake_sync(settings):
        called["yes"] = True
        return {"articles": 0, "chunks": 0, "updated": 0, "failed": 0}

    monkeypatch.setattr("app.main.sync_vectors", fake_sync)
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/chat/stop", json={"conversationId": "x"})
        assert resp.status_code == 200
    assert called.get("yes") is True


def test_lifespan_tolerates_sync_failure(monkeypatch):
    async def bad_sync(settings):
        raise RuntimeError("ES down")

    monkeypatch.setattr("app.main.sync_vectors", bad_sync)
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/chat/stop", json={"conversationId": "x"})
        assert resp.status_code == 200  # 同步失败不阻断启动
