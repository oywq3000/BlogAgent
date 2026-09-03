# -*- coding: utf-8 -*-
"""工具：按关键词搜索已发布文章（search_articles）。"""
import json

import httpx
from langchain_core.tools import BaseTool, tool

from app.config import Settings

from ._client import _clean_highlight


def build_search_articles(settings: Settings, client: httpx.AsyncClient) -> BaseTool:
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

    return search_articles
