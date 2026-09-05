# -*- coding: utf-8 -*-
"""工具：按发布时间倒序分页浏览已发布文章（list_articles）。"""
import json

import httpx
from langchain_core.tools import BaseTool, tool

from app.config import Settings


def build_list_articles(settings: Settings, client: httpx.AsyncClient) -> BaseTool:
    # 工具请求始终按 settings 的凭据认证 ES：外部传入的 client（如测试的 MockTransport）可能不含 auth
    auth = (settings.es_username, settings.es_password) if settings.es_username else None

    @tool
    async def list_articles(page: int = 1, page_size: int = 10) -> str:
        """按发布时间倒序浏览已发布文章列表。

        Args:
            page: 页码（从 1 开始）
            page_size: 每页条数（默认 10）
        Returns:
            JSON 列表：每项含 id、title、summary、tags、createdAt 与文章外链（url）
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
        for it in items:
            it["url"] = f"{settings.site_url}/article/{it.get('id')}"
        return json.dumps(items, ensure_ascii=False)

    return list_articles
