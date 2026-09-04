# -*- coding: utf-8 -*-
"""向量同步：article_chunks 索引全量重建（规格 2026-09-04 §4.3）。"""
import logging

import httpx

from app.agent.tools._client import build_es_client
from app.config import Settings

logger = logging.getLogger(__name__)

_INDEX_MAPPINGS = {
    "properties": {
        "article_id": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        "content": {"type": "text"},
        "title": {"type": "text"},
        "tags": {"type": "keyword"},
        "createdAt": {"type": "date"},
        "content_vector": {"type": "dense_vector", "dims": 1024, "index": True, "similarity": "l2_norm"},
    }
}


def _resolve_client(settings: Settings, client: httpx.AsyncClient | None) -> httpx.AsyncClient:
    return client or build_es_client(settings)


def _resolve_embed_documents(embed_documents):
    if embed_documents is None:
        from app.embedding import embed_documents as _impl
        return _impl
    return embed_documents


def _resolve_count_tokens(count_tokens):
    if count_tokens is None:
        def _impl(text: str) -> int:
            from app.embedding import get_embedder
            return len(get_embedder().tokenizer.encode(text))
        return _impl
    return count_tokens


def _auth(settings: Settings):
    return (settings.es_username, settings.es_password) if settings.es_username else None


async def ensure_index(settings: Settings, client: httpx.AsyncClient | None = None) -> None:
    """建 article_chunks 索引（幂等：resource_already_exists_exception 视为成功）。"""
    client = _resolve_client(settings, client)
    resp = await client.put("/article_chunks", json={"mappings": _INDEX_MAPPINGS}, auth=_auth(settings))
    if resp.status_code == 400 and "resource_already_exists_exception" in resp.text:
        return
    resp.raise_for_status()


async def sync_vectors(
    settings: Settings,
    client: httpx.AsyncClient | None = None,
    embed_documents=None,
    count_tokens=None,
) -> dict:
    """全量重建 chunk 索引：删旧建新 → 读文章 → 切块 → 嵌入 → 写入。"""
    from app.chunking import split_content
    client = _resolve_client(settings, client)
    embed_documents = _resolve_embed_documents(embed_documents)
    count_tokens = _resolve_count_tokens(count_tokens)
    auth = _auth(settings)
    # 1. 删旧索引（404 忽略）→ 重建
    resp = await client.delete("/article_chunks", auth=auth)
    if resp.status_code not in (200, 404):
        resp.raise_for_status()
    await ensure_index(settings, client)
    # 2. 读文章（含全文与展示元数据）
    resp = await client.post(
        "/articles/_search",
        json={
            "query": {"match_all": {}},
            "size": 10000,
            "_source": ["id", "title", "tags", "createdAt", "content"],
        },
        auth=auth,
    )
    resp.raise_for_status()
    articles = [h["_source"] for h in resp.json().get("hits", {}).get("hits", [])]
    # 3. 切块 + 嵌入 + 逐 chunk 写入
    chunks_total = updated = failed = 0
    for article in articles:
        pieces = split_content(
            article.get("content") or "",
            count_tokens,
            settings.chunk_max_tokens,
            settings.chunk_overlap_tokens,
        )
        if not pieces:
            continue
        vectors = embed_documents(pieces)
        for i, (piece, vec) in enumerate(zip(pieces, vectors)):
            doc_id = f"{article['id']}-{i}"
            try:
                await client.put(
                    f"/article_chunks/_doc/{doc_id}",
                    json={
                        "article_id": article["id"],
                        "chunk_index": i,
                        "content": piece,
                        "title": article.get("title", ""),
                        "tags": article.get("tags", []),
                        "createdAt": article.get("createdAt"),
                        "content_vector": vec,
                    },
                    auth=auth,
                )
                updated += 1
            except httpx.HTTPError as e:
                logger.warning("写回 chunk 失败 %s: %s", doc_id, e)
                failed += 1
        chunks_total += len(pieces)
    logger.info("向量同步：文章 %d，chunk %d，成功 %d，失败 %d", len(articles), chunks_total, updated, failed)
    return {"articles": len(articles), "chunks": chunks_total, "updated": updated, "failed": failed}


async def rebuild_vectors(
    settings: Settings,
    client: httpx.AsyncClient | None = None,
    embed_documents=None,
    count_tokens=None,
) -> dict:
    """手动全量重建（与 sync_vectors 等价，供脚本调用）。"""
    return await sync_vectors(settings, client, embed_documents, count_tokens)
