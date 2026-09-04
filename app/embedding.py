# -*- coding: utf-8 -*-
"""BGE 文本向量化：懒加载 + query/passage 指令前缀区分（规格 2026-09-04 §4.1）。"""
import logging

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

# bge-large-zh-v1.5 官方查询指令：查询侧必须加，文档侧不加
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """懒加载 + 缓存；指定设备失败自动回退 cpu。"""
    global _embedder
    if _embedder is None:
        settings = get_settings()
        try:
            _embedder = SentenceTransformer(settings.embedding_model_path, device=settings.embedding_device)
        except Exception:
            logger.warning("embedding 在 %s 加载失败，回退 cpu", settings.embedding_device)
            _embedder = SentenceTransformer(settings.embedding_model_path, device="cpu")
    return _embedder


def embed_query(text: str) -> list[float]:
    return get_embedder().encode(_QUERY_PREFIX + text, show_progress_bar=False).tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    # 空文本兜底：空串编码无意义，用单空格替代；超长文本由 encode 按模型 max_seq_length 截断
    texts = [t if t else " " for t in texts]
    return get_embedder().encode(texts, show_progress_bar=False).tolist()
