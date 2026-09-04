# -*- coding: utf-8 -*-
"""BGE 向量化单测：指令前缀区分、懒加载、设备回退、空文本兜底（全程 mock 模型，不加载真实 BGE）。"""
import numpy as np
import pytest

import app.embedding as emb


class FakeModel:
    instances = 0

    def __init__(self, path=None, device=None):
        self.path = path
        self.device = device
        self.calls = []
        FakeModel.instances += 1

    def encode(self, texts, **kwargs):
        self.calls.append(texts)
        if isinstance(texts, str):
            texts = [texts]
        return np.array([[float(len(t))] for t in texts], dtype="float32")


@pytest.fixture(autouse=True)
def _reset_embedder():
    emb._embedder = None
    FakeModel.instances = 0
    yield
    emb._embedder = None


def test_embed_query_has_prefix(monkeypatch):
    monkeypatch.setattr(emb, "SentenceTransformer", FakeModel)
    vec = emb.embed_query("微服务")
    assert FakeModel.instances == 1
    assert emb._embedder.calls[0] == "为这个句子生成表示以用于检索相关文章：微服务"
    assert isinstance(vec, list) and len(vec) == 1


def test_embed_documents_no_prefix(monkeypatch):
    monkeypatch.setattr(emb, "SentenceTransformer", FakeModel)
    vecs = emb.embed_documents(["SSE协议", "响应式编程"])
    assert FakeModel.instances == 1
    assert emb._embedder.calls[0] == ["SSE协议", "响应式编程"]
    assert len(vecs) == 2 and all(isinstance(v, list) for v in vecs)


def test_embedder_lazy_singleton(monkeypatch):
    monkeypatch.setattr(emb, "SentenceTransformer", FakeModel)
    emb.get_embedder()
    emb.get_embedder()
    assert FakeModel.instances == 1  # 第二次复用缓存


def test_embed_documents_empty_text_fallback(monkeypatch):
    monkeypatch.setattr(emb, "SentenceTransformer", FakeModel)
    emb.embed_documents(["", "正常文本"])
    assert emb._embedder.calls[0] == [" ", "正常文本"]  # 空串兜底为单空格


def test_device_fallback_to_cpu(monkeypatch):
    class FlakyModel(FakeModel):
        def __init__(self, path=None, device=None):
            if device == "cuda":
                raise RuntimeError("cuda unavailable")
            super().__init__(path, device)

    monkeypatch.setattr(emb, "SentenceTransformer", FlakyModel)
    embedder = emb.get_embedder()
    assert embedder.device == "cpu"
