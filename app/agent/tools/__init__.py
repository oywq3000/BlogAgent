# -*- coding: utf-8 -*-
"""博客检索工具包：三个 ES 只读工具，各占一个文件。

查询逻辑是 search-service 的 Python 侧最小重实现（status 过滤 + highlight）。
工具执行失败不抛异常：错误文本回给模型，由模型兜底回答（规格 §6）。

对外保持原 tools 模块接口：build_tools(settings, client=None, embed_query=None) -> [search_articles, get_article_content, list_articles]。
"""
import httpx

from app.config import Settings

from ._client import build_es_client
from .get_article_content import build_get_article_content
from .list_articles import build_list_articles
from .search_articles import build_search_articles

__all__ = ["build_tools", "build_es_client"]


def build_tools(settings: Settings, client: httpx.AsyncClient | None = None, embed_query=None) -> list:
    client = client or build_es_client(settings)
    return [
        build_search_articles(settings, client, embed_query=embed_query),
        build_get_article_content(settings, client),
        build_list_articles(settings, client),
    ]
