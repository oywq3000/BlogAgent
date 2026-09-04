import json

import httpx
import pytest

from app.agent.tools import build_es_client, build_tools
from app.agent.tools.search_articles import _fuse
from app.config import Settings


def fake_embed_query(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def make_settings() -> Settings:
    return Settings(
        es_url="http://es.test:9200",
        es_username="elastic",
        es_password="pw123",
        search_page_size=5,
        article_content_max_chars=100,
        _env_file=None,
    )


SEARCH_HIT = {
    "_source": {"id": "2088", "title": "SSE协议", "tags": [], "createdAt": "2026-08-15T21:52:52.000"},
    "highlight": {"content": ["<em>服务</em>端会一直主动推送消息"]},
}


def test_fuse_two_route_rrf_ranking():
    bm25_hits = [
        {"_source": {"id": "A", "title": "A文", "tags": [], "createdAt": "t"},
         "highlight": {"content": ["<em>微</em>服务命中"]}},
        {"_source": {"id": "B", "title": "B文", "tags": [], "createdAt": "t"}},
    ]
    knn_hits = [
        {"_source": {"article_id": "B", "title": "B文", "tags": [], "createdAt": "t", "content": "B 的 chunk"}},
        {"_source": {"article_id": "C", "title": "C文", "tags": [], "createdAt": "t", "content": "C 的 chunk"}},
    ]
    items = _fuse(bm25_hits, knn_hits, page_size=5)
    assert [i["id"] for i in items] == ["B", "A", "C"]  # B: 1/62+1/61 最高；A: 1/61；C: 1/61
    assert items[0]["snippet"] == "B 的 chunk"[:150]  # knn-only 命中用 chunk 内容截取


def test_fuse_uses_highlight_snippet_and_dedup_chunks():
    bm25_hits = [
        {"_source": {"id": "A", "title": "A文", "tags": [], "createdAt": "t"},
         "highlight": {"content": ["<em>SSE</em>协议"]}},
    ]
    knn_hits = [
        {"_source": {"article_id": "A", "title": "A文", "tags": [], "createdAt": "t", "content": "c1"}},
        {"_source": {"article_id": "A", "title": "A文", "tags": [], "createdAt": "t", "content": "c2"}},
    ]
    items = _fuse(bm25_hits, knn_hits, page_size=5)
    assert len(items) == 1  # 同文章多 chunk 去重
    assert items[0]["snippet"] == "SSE协议"  # BM25 路 highlight 优先，去 em 标签


@pytest.mark.asyncio
async def test_search_articles_query_shape_and_auth():
    """断言两路请求形状（BM25 body 原样、knn body 过滤一致）与 BasicAuth 头。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        captured.setdefault("urls", []).append(url)
        captured.setdefault("auths", []).append(request.headers.get("Authorization", ""))
        captured.setdefault("bodies", []).append(json.loads(request.content))
        if "/article_chunks/" in url:
            return httpx.Response(200, json={"hits": {"hits": []}})
        return httpx.Response(200, json={"hits": {"hits": [SEARCH_HIT]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client, embed_query=fake_embed_query)
    result = await tools[0].ainvoke({"keyword": "微服务"})

    assert captured["urls"] == [
        "http://es.test/articles/_search",
        "http://es.test/article_chunks/_search",
    ]
    assert all(a.startswith("Basic ") for a in captured["auths"])
    bm25, knn = captured["bodies"]
    assert bm25["query"]["bool"]["filter"] == {"term": {"status": "published"}}
    assert bm25["query"]["bool"]["must"]["multi_match"]["fields"] == ["title^3", "content", "summary"]
    assert bm25["size"] == 5
    assert "title" in bm25["highlight"]["fields"]
    assert knn["knn"]["field"] == "content_vector"
    assert knn["knn"]["query_vector"] == [0.1, 0.2, 0.3]
    assert knn["knn"]["filter"] == {"term": {"status": "published"}}  # 与 BM25 路过滤同域
    assert knn["knn"]["k"] == 25  # search_page_size * 5
    # 高亮片段去标签、结果可解析、含标题
    assert "<em>" not in result
    assert "SSE协议" in result
    assert json.loads(result)[0]["title"] == "SSE协议"


@pytest.mark.asyncio
async def test_get_article_content_truncates():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://es.test/articles/_doc/a1"
        return httpx.Response(200, json={
            "_source": {"id": "a1", "title": "响应式编程", "content": "x" * 300}
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client)
    result = await tools[1].ainvoke({"article_id": "a1"})

    assert "响应式编程" in result
    assert len(json.loads(result)["content"]) == 100  # article_content_max_chars=100


@pytest.mark.asyncio
async def test_get_article_content_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"found": False})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client)
    result = await tools[1].ainvoke({"article_id": "ghost"})
    assert "未找到" in result


@pytest.mark.asyncio
async def test_list_articles_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["sort"] == [{"createdAt": "desc"}]
        assert body["from"] == 10  # page=2, size=10
        return httpx.Response(200, json={"hits": {"hits": [SEARCH_HIT]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client)
    result = await tools[2].ainvoke({"page": 2, "page_size": 10})
    assert "SSE协议" in result


@pytest.mark.asyncio
async def test_tool_error_returns_text_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client, embed_query=fake_embed_query)
    result = await tools[0].ainvoke({"keyword": "任意"})
    assert "搜索失败" in result  # 错误文本回给模型兜底，不抛异常


@pytest.mark.asyncio
async def test_no_auth_when_username_empty():
    captured = {"auth": ""}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] += request.headers.get("Authorization", "")
        if "/article_chunks/" in str(request.url):
            return httpx.Response(200, json={"hits": {"hits": []}})
        return httpx.Response(200, json={"hits": {"hits": []}})

    settings = make_settings()
    settings.es_username = ""
    settings.es_password = ""
    client = build_es_client(settings)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(settings, client=client, embed_query=fake_embed_query)
    await tools[0].ainvoke({"keyword": "x"})
    assert captured["auth"] == ""


@pytest.mark.asyncio
async def test_hybrid_disabled_only_bm25_request():
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"hits": {"hits": []}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    settings = make_settings()
    settings.hybrid_search = False
    tools = build_tools(settings, client=client, embed_query=fake_embed_query)
    await tools[0].ainvoke({"keyword": "微服务"})
    assert urls == ["http://es.test/articles/_search"]  # 只有 BM25 一路


@pytest.mark.asyncio
async def test_hybrid_falls_back_when_embed_fails():
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"hits": {"hits": []}})

    def broken_embed(text):
        raise RuntimeError("model load failed")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client, embed_query=broken_embed)
    result = await tools[0].ainvoke({"keyword": "微服务"})
    assert urls == ["http://es.test/articles/_search"]  # embed 失败 → 仅 BM25 路
    assert json.loads(result) == []


@pytest.mark.asyncio
async def test_knn_400_falls_back_to_bm25_only():
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        urls.append(url)
        if "/article_chunks/" in url:
            return httpx.Response(400, json={"error": {"type": "search_phase_execution_exception"}})
        return httpx.Response(200, json={"hits": {"hits": [SEARCH_HIT]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client, embed_query=fake_embed_query)
    result = await tools[0].ainvoke({"keyword": "微服务"})
    assert len(urls) == 2  # 发了两路，knn 400 被吞
    assert json.loads(result)[0]["title"] == "SSE协议"  # BM25 路结果照常返回
