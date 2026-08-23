# -*- coding: utf-8 -*-
"""博客检索三工具：直连 ES articles 索引（只读）。

查询逻辑是 search-service 的 Python 侧最小重实现（status 过滤 + highlight）。
工具执行失败不抛异常：错误文本回给模型，由模型兜底回答（规格 §6）。
"""
import json
import re

import httpx
from langchain_core.tools import tool

from app.config import Settings

_EM_TAG = re.compile(r"</?em>")


def build_es_client(settings: Settings) -> httpx.AsyncClient:
    auth = (settings.es_username, settings.es_password) if settings.es_username else None
    return httpx.AsyncClient(base_url=settings.es_url, auth=auth, timeout=10.0)


def _clean_highlight(text: str) -> str:
    return _EM_TAG.sub("", text)


def build_tools(settings: Settings, client: httpx.AsyncClient | None = None) -> list:
    client = client or build_es_client(settings)
    # 工具请求始终按 settings 的凭据认证 ES：外部传入的 client（如测试的 MockTransport）可能不含 auth
    auth = (settings.es_username, settings.es_password) if settings.es_username else None

    @tool
    async def search_articles(keyword: str) -> str:
        """按关键词搜索博客已发布文章。

        Args:
            keyword: 搜索关键词，如“微服务”“SSE”
        Returns:
            JSON 列表：每项含 id、title、tags、createdAt 与命中内容片段（highlight）
        """
        body = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": keyword,
                            "fields": ["title^3", "content", "summary"],
                        }
                    },
                    "filter": {"term": {"status": "published"}},
                }
            },
            "highlight": {
                "fields": {
                    "title": {"number_of_fragments": 0},
                    "content": {"fragment_size": 150, "number_of_fragments": 1},
                }
            },
            "size": settings.search_page_size,
            "_source": ["id", "title", "tags", "createdAt"],
        }
        try:
            resp = await client.post("/articles/_search", json=body, auth=auth)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return f"搜索失败：{e.__class__.__name__}，请告知用户博客检索暂不可用"
        hits = resp.json().get("hits", {}).get("hits", [])
        items = []
        for h in hits:
            src = h.get("_source", {})
            hl = h.get("highlight", {})
            item = {
                "id": src.get("id"),
                "title": src.get("title"),
                "tags": src.get("tags", []),
                "createdAt": src.get("createdAt"),
            }
            snippet = hl.get("content") or hl.get("title")
            if snippet:
                item["snippet"] = _clean_highlight(snippet[0])
            items.append(item)
        return json.dumps(items, ensure_ascii=False)

    @tool
    async def get_article_content(article_id: str) -> str:
        """获取指定文章的完整内容。

        Args:
            article_id: 文章 id（先通过 search_articles 获得）
        Returns:
            JSON：title 与 content（超出上限截断，并注明已截断）
        """
        try:
            resp = await client.get(f"/articles/_doc/{article_id}", auth=auth)
        except httpx.HTTPError as e:
            return f"读取文章失败：{e.__class__.__name__}，请告知用户文章读取暂不可用"
        if resp.status_code == 404:
            return "未找到该文章，请确认文章 id 是否正确"
        resp.raise_for_status()
        src = resp.json().get("_source", {})
        content = src.get("content") or ""
        truncated = len(content) > settings.article_content_max_chars
        if truncated:
            content = content[: settings.article_content_max_chars]
        return json.dumps(
            {
                "title": src.get("title"),
                "content": content,
                "truncated": truncated,
            },
            ensure_ascii=False,
        )

    @tool
    async def list_articles(page: int = 1, page_size: int = 10) -> str:
        """按发布时间倒序浏览已发布文章列表。

        Args:
            page: 页码（从 1 开始）
            page_size: 每页条数（默认 10）
        Returns:
            JSON 列表：每项含 id、title、summary、tags、createdAt
        """
        body = {
            "query": {"term": {"status": "published"}},
            "sort": [{"createdAt": "desc"}],
            "from": max(page - 1, 0) * page_size,
            "size": page_size,
            "_source": ["id", "title", "summary", "tags", "createdAt"],
        }
        try:
            resp = await client.post("/articles/_search", json=body, auth=auth)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return f"浏览文章失败：{e.__class__.__name__}，请告知用户文章列表暂不可用"
        hits = resp.json().get("hits", {}).get("hits", [])
        items = [h.get("_source", {}) for h in hits]
        return json.dumps(items, ensure_ascii=False)

    return [search_articles, get_article_content, list_articles]
