# -*- coding: utf-8 -*-
"""工具：按关键词搜索已发布文章（search_articles），BM25+向量两路混合检索（规格 2026-09-04 §5）。"""
import json
import logging

import httpx
from langchain_core.tools import BaseTool, tool

from app.config import Settings

from ._client import _clean_highlight

logger = logging.getLogger(__name__)

_RRF_K = 60
_SNIPPET_CHARS = 150


def _bm25_body(keyword: str, page_size: int) -> dict:
    return {
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
        "size": page_size,
        "_source": ["id", "title", "tags", "createdAt"],
    }


def _knn_body(query_vector: list[float], knn_k: int) -> dict:
    return {
        "knn": {
            "field": "content_vector",
            "query_vector": query_vector,
            "k": knn_k,
            "filter": {"term": {"status": "published"}},
        },
        "_source": ["article_id", "content", "title", "tags", "createdAt", "chunk_index"],
    }


def _fuse(bm25_hits: list[dict], knn_hits: list[dict], page_size: int) -> list[dict]:
    """两路文章级 RRF 融合。bm25_hits 按序记 rank；knn_hits 是 chunk 级，按 article_id 归并取最小 rank。"""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for rank, h in enumerate(bm25_hits):
        src = h.get("_source", {})
        aid = src.get("id")
        if not aid:
            continue
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        hl = h.get("highlight", {})
        snippet = (hl.get("content") or hl.get("title") or [None])[0]
        items[aid] = {
            "id": aid,
            "title": src.get("title"),
            "tags": src.get("tags", []),
            "createdAt": src.get("createdAt"),
        }
        if snippet:
            items[aid]["snippet"] = _clean_highlight(snippet)
    first_rank: dict[str, int] = {}
    first_src: dict[str, dict] = {}
    for rank, h in enumerate(knn_hits):
        src = h.get("_source", {})
        aid = src.get("article_id")
        if not aid:
            continue
        if aid not in first_rank:  # 首次出现 rank 最小
            first_rank[aid] = rank
            first_src[aid] = src
    for aid, rank in first_rank.items():
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        src = first_src[aid]
        if aid not in items:  # knn-only 命中：chunk 冗余了展示元数据
            items[aid] = {
                "id": aid,
                "title": src.get("title"),
                "tags": src.get("tags", []),
                "createdAt": src.get("createdAt"),
            }
        if "snippet" not in items[aid]:  # BM25 路无 highlight 时，用 chunk 内容补 snippet
            snippet = (src.get("content") or "")[:_SNIPPET_CHARS]
            if snippet:
                items[aid]["snippet"] = snippet
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [items[aid] for aid, _ in ranked[:page_size]]


def build_search_articles(
    settings: Settings,
    client: httpx.AsyncClient,
    embed_query=None,
) -> BaseTool:
    # 工具请求始终按 settings 的凭据认证 ES：外部传入的 client（如测试的 MockTransport）可能不含 auth
    auth = (settings.es_username, settings.es_password) if settings.es_username else None
    injected_embed_query = embed_query  # 保留注入（测试/显式传入）；构建工具时不触碰 embedding 模块

    @tool
    async def search_articles(keyword: str) -> str:
        """按关键词搜索博客已发布文章（BM25+向量混合检索）。

        Args:
            keyword: 搜索关键词，如“微服务”“SSE”
        Returns:
            JSON 列表：每项含 id、title、tags、createdAt 与命中内容片段（snippet）
        """
        # 路1：BM25 查 articles（原逻辑）
        try:
            bm25_resp = await client.post("/articles/_search", json=_bm25_body(keyword, settings.search_page_size), auth=auth)
            if bm25_resp.status_code >= 400:
                return f"搜索失败：HTTP {bm25_resp.status_code}，请告知用户博客检索暂不可用"
        except httpx.HTTPError as e:
            return f"搜索失败：{e.__class__.__name__}，请告知用户博客检索暂不可用"
        bm25_hits = bm25_resp.json().get("hits", {}).get("hits", [])

        # 路2：knn 查 article_chunks（仅混合开启且 embedding 可用；失败不影响路1）
        knn_hits = []
        if settings.hybrid_search:
            effective_embed_query = injected_embed_query
            if effective_embed_query is None:
                # 惰性导入：构建工具时不触碰 embedding，失败即降级仅 BM25 路
                try:
                    from app.embedding import embed_query as _default_embed_query
                    effective_embed_query = _default_embed_query
                except Exception as e:
                    logger.warning("embedding 模块不可用，仅 BM25 路：%s", e)
            if effective_embed_query is not None:
                try:
                    query_vector = effective_embed_query(keyword)
                except Exception as e:
                    logger.warning("embedding 失败，仅 BM25 路：%s", e)
                    query_vector = None
            if query_vector is not None:
                try:
                    knn_resp = await client.post(
                        "/article_chunks/_search",
                        json=_knn_body(query_vector, settings.search_page_size * 5),
                        auth=auth,
                    )
                    if knn_resp.status_code == 400:
                        logger.warning("knn 查询失败（chunk 索引可能未建），仅 BM25 路")
                    else:
                        knn_resp.raise_for_status()
                        knn_hits = knn_resp.json().get("hits", {}).get("hits", [])
                except httpx.HTTPError as e:
                    logger.warning("knn 查询失败，仅 BM25 路：%s", e)

        items = _fuse(bm25_hits, knn_hits, settings.search_page_size)
        return json.dumps(items, ensure_ascii=False)

    return search_articles
