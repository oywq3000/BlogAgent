# -*- coding: utf-8 -*-
"""工具：按 id 获取文章完整内容（get_article_content）。"""
import json

import httpx
from langchain_core.tools import BaseTool, tool

from app.config import Settings


def build_get_article_content(settings: Settings, client: httpx.AsyncClient) -> BaseTool:
    # 工具请求始终按 settings 的凭据认证 ES：外部传入的 client（如测试的 MockTransport）可能不含 auth
    auth = (settings.es_username, settings.es_password) if settings.es_username else None

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

    return get_article_content
