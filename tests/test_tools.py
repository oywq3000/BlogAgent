import json

import httpx
import pytest

from app.agent.tools import build_es_client, build_tools
from app.config import Settings


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


@pytest.mark.asyncio
async def test_search_articles_query_shape_and_auth():
    """断言查询体（status 过滤、highlight、多字段匹配）与 BasicAuth 头。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization", "")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"hits": {"hits": [SEARCH_HIT]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client)
    result = await tools[0].ainvoke({"keyword": "微服务"})

    body = captured["body"]
    assert captured["url"] == "http://es.test/articles/_search"
    assert captured["auth"].startswith("Basic ")
    assert body["query"]["bool"]["filter"] == {"term": {"status": "published"}}
    assert body["query"]["bool"]["must"]["multi_match"]["query"] == "微服务"
    assert body["query"]["bool"]["must"]["multi_match"]["fields"] == ["title^3", "content", "summary"]
    assert body["size"] == 5
    assert "title" in body["highlight"]["fields"]
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
    tools = build_tools(make_settings(), client=client)
    result = await tools[0].ainvoke({"keyword": "任意"})
    assert "搜索失败" in result  # 错误文本回给模型兜底，不抛异常


@pytest.mark.asyncio
async def test_no_auth_when_username_empty():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json={"hits": {"hits": []}})

    settings = make_settings()
    settings.es_username = ""
    settings.es_password = ""
    client = build_es_client(settings)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(settings, client=client)
    await tools[0].ainvoke({"keyword": "x"})
    assert captured["auth"] == ""
