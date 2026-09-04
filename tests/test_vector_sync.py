# -*- coding: utf-8 -*-
"""向量同步单测：索引幂等、全量重建、chunk 写字段、单条失败继续（MockTransport 全程拦截）。"""
import json

import httpx
import pytest

from app.config import Settings
from app.vector_sync import ensure_index, rebuild_vectors, sync_vectors


def make_settings() -> Settings:
    return Settings(
        es_url="http://es.test:9200",
        es_username="elastic",
        es_password="pw123",
        hybrid_search=True,
        chunk_max_tokens=50,
        chunk_overlap_tokens=10,
        _env_file=None,
    )


def fake_embed_documents(texts):
    return [[float(len(t)), 0.0] for t in texts]


def run(coro):
    import asyncio
    return asyncio.run(coro)


def test_ensure_index_power_idempotent():
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.method == "PUT":
            return httpx.Response(400, json={}, text='{"error":{"type":"resource_already_exists_exception"}}')
        raise AssertionError(f"unexpected: {request.method} {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    run(ensure_index(make_settings(), client))  # 不抛异常
    assert calls == ["http://es.test/article_chunks"]


def test_sync_vectors_rebuilds_chunk_index():
    calls = []

    def make_handler():
        async def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            method = request.method
            body = json.loads(request.content) if request.content else None
            calls.append((method, url, body))
            if method == "DELETE" and url.endswith("/article_chunks"):
                return httpx.Response(200, json={"acknowledged": True})
            if method == "PUT" and url.endswith("/article_chunks"):
                return httpx.Response(200, json={"acknowledged": True})
            if url.endswith("/articles/_search"):
                return httpx.Response(200, json={"hits": {"hits": [
                    {"_source": {"id": "a1", "title": "SSE协议", "tags": [], "createdAt": "2026-08-15",
                                 "content": "第一段。\n\n第二段。"}},
                ]}})
            if method == "PUT" and "/article_chunks/_doc/" in url:
                return httpx.Response(200, json={"result": "created"})
            raise AssertionError(f"unexpected: {method} {url}")
        return handler

    client = httpx.AsyncClient(transport=httpx.MockTransport(make_handler()), base_url="http://es.test")
    result = run(sync_vectors(make_settings(), client, embed_documents=fake_embed_documents, count_tokens=len))
    assert result == {"articles": 1, "chunks": 1, "updated": 1, "failed": 0}
    # 删除旧索引 → 重建 → 读文章 → 写 chunk（4 个请求，顺序断言）
    assert calls[0][0] == "DELETE" and calls[0][1].endswith("/article_chunks")
    assert calls[1][0] == "PUT" and calls[1][1].endswith("/article_chunks")
    assert calls[2][1].endswith("/articles/_search")
    doc_call = calls[3]
    assert doc_call[1] == "http://es.test/article_chunks/_doc/a1-0"
    doc = doc_call[2]
    assert doc["article_id"] == "a1" and doc["chunk_index"] == 0
    assert doc["title"] == "SSE协议" and doc["content"] == "第一段。\n\n第二段。"
    assert doc["content_vector"] == [10.0, 0.0]  # fake: len("第一段。\n\n第二段。")=10


def test_sync_vectors_continues_on_single_failure():
    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "DELETE" or (request.method == "PUT" and url.endswith("/article_chunks")):
            return httpx.Response(200, json={"acknowledged": True})
        if url.endswith("/articles/_search"):
            return httpx.Response(200, json={"hits": {"hits": [
                {"_source": {"id": "a1", "title": "T", "tags": [], "createdAt": None, "content": "AAA"}},
                {"_source": {"id": "a2", "title": "U", "tags": [], "createdAt": None, "content": "BBBB"}},
            ]}})
        if "/article_chunks/_doc/" in url and url.endswith("/a1-0"):
            raise httpx.ConnectError("es down", request=request)
        return httpx.Response(200, json={"result": "created"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    result = run(sync_vectors(make_settings(), client, embed_documents=fake_embed_documents, count_tokens=len))
    assert result["updated"] == 1
    assert result["failed"] == 1


def test_sync_vectors_counts_http_error_status_as_failed():
    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "DELETE" or (request.method == "PUT" and url.endswith("/article_chunks")):
            return httpx.Response(200, json={"acknowledged": True})
        if url.endswith("/articles/_search"):
            return httpx.Response(200, json={"hits": {"hits": [
                {"_source": {"id": "a1", "title": "T", "tags": [], "createdAt": None, "content": "AAA"}},
                {"_source": {"id": "a2", "title": "U", "tags": [], "createdAt": None, "content": "BBBB"}},
            ]}})
        if "/article_chunks/_doc/" in url and url.endswith("/a1-0"):
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"result": "created"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    result = run(sync_vectors(make_settings(), client, embed_documents=fake_embed_documents, count_tokens=len))
    assert result["updated"] == 1
    assert result["failed"] == 1


def test_rebuild_vectors_equals_sync():
    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "DELETE" or (request.method == "PUT" and url.endswith("/article_chunks")):
            return httpx.Response(200, json={"acknowledged": True})
        if url.endswith("/articles/_search"):
            return httpx.Response(200, json={"hits": {"hits": []}})
        raise AssertionError(f"unexpected: {request.method} {url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    result = run(rebuild_vectors(make_settings(), client, embed_documents=fake_embed_documents, count_tokens=len))
    assert result == {"articles": 0, "chunks": 0, "updated": 0, "failed": 0}
