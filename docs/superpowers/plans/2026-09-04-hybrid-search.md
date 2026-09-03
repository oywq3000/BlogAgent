# search_articles 混合检索（BM25+向量+RRF）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `search_articles` 升级为 BM25 关键词 + BGE 向量双路综合检索（ES 原生 RRF 融合），并配套向量同步基础设施。

**Architecture:** BGE（本地 GPU）对查询与文章生成 1024 维向量；向量写回 ES articles 索引新增的 `content_vector` 字段（`dense_vector`/`l2_norm`）；`search_articles` 一个 `_search` 请求同时发 `knn` + BM25 query + `rank.rrf` 融合。启动时增量同步补缺失向量，`scripts/rebuild_vectors.py` 手动全量重建。任何环节失败降级回纯 BM25，不崩服务。

**Tech Stack:** Python 3.12 / FastAPI / langchain-core / httpx / sentence-transformers / Elasticsearch 8.17（knn + rank.rrf）

**Spec:** `docs/superpowers/specs/2026-09-04-hybrid-search-design.md`

## Global Constraints

- ES 8.17.10，`content_vector` 字段：`dense_vector`、dims=**1024**、`index=true`、`similarity: "l2_norm"`（BGE 官方推荐欧氏距离，不用 cosine）
- bge-large-zh-v1.5 **查询侧必须加指令前缀** `"为这个句子生成表示以用于检索相关文章:"`，**文档侧不加**——同步生成的文档向量必须无前缀
- 模型路径 `F:/models/bge-large-zh-v1.5`，设备 `cuda`（加载失败自动回退 `cpu`）
- **Java 端零改动**：只给 articles 索引新增 `content_vector` 字段，不动现有字段
- 降级哲学（规格 §6）：工具执行失败不抛异常，错误文本回模型兜底；混合检索任何环节失败 → 回退纯 BM25
- knn 的 `filter` 必须与 BM25 query 的 `bool.filter` 一致（`status: published`），保证两路候选同域
- 测试命令统一：`MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest <file> -v`（conda 环境 ai-agent，ES 请求一律 MockTransport 拦截，**不加载真实 BGE 模型**）
- commit 信息用中文 + `feat:`/`refactor:`/`test:` 前缀，不加 AI 尾注（用户偏好）

---

### Task 1: 新增混合检索配置项

**Files:**
- Modify: `app/config.py`（在 `search_page_size` 附近加 3 项）
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.embedding_model_path: str = "F:/models/bge-large-zh-v1.5"`、`Settings.embedding_device: str = "cuda"`、`Settings.hybrid_search: bool = True`——后续所有任务读取这三个字段

- [ ] **Step 1: 写失败测试**

在 `tests/test_config.py` 追加（先读该文件，确认现有 `Settings(_env_file=None)` 的写法后按同样模式写）：

```python
def test_hybrid_search_defaults():
    s = Settings(_env_file=None)
    assert s.embedding_model_path == "F:/models/bge-large-zh-v1.5"
    assert s.embedding_device == "cuda"
    assert s.hybrid_search is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_config.py::test_hybrid_search_defaults -v`
Expected: FAIL（`AttributeError`，字段不存在）

- [ ] **Step 3: 实现**

在 `app/config.py` 的 `search_page_size` 之后加：

```python
    embedding_model_path: str = "F:/models/bge-large-zh-v1.5"
    embedding_device: str = "cuda"  # 加载失败自动回退 cpu
    hybrid_search: bool = True  # 混合检索（BM25+向量+RRF）总开关
```

- [ ] **Step 4: 跑测试确认通过**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: 新增混合检索配置项（模型路径/设备/总开关）"
```

---

### Task 2: BGE 向量化模块 app/embedding.py

**Files:**
- Create: `app/embedding.py`
- Test: `tests/test_embedding.py`

**Interfaces:**
- Consumes: `Settings.embedding_model_path`、`Settings.embedding_device`（Task 1）
- Produces:
  - `get_embedder() -> SentenceTransformer`——懒加载 + 模块级缓存；指定设备加载失败回退 `cpu`
  - `embed_query(text: str) -> list[float]`——输入加查询指令前缀后编码
  - `embed_documents(texts: list[str]) -> list[list[float]]`——不加前缀
  - 模块级 `_embedder` 缓存变量（测试重置用）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_embedding.py`：

```python
# -*- coding: utf-8 -*-
"""BGE 向量化单测：指令前缀区分、懒加载、设备回退（全程 mock 模型，不加载真实 BGE）。"""
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
    assert emb._embedder.calls[0][0] == "为这个句子生成表示以用于检索相关文章:微服务"
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_embedding.py -v`
Expected: FAIL（`ModuleNotFoundError: app.embedding`）

- [ ] **Step 3: 实现**

创建 `app/embedding.py`：

```python
# -*- coding: utf-8 -*-
"""BGE 文本向量化：懒加载 + query/passage 指令前缀区分（规格 2026-09-04 §4.1）。"""
import logging

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

# bge-large-zh-v1.5 官方查询指令：查询侧必须加，文档侧不加
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章:"

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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_embedding.py -v`
Expected: PASS（5 个）

- [ ] **Step 5: 提交**

```bash
git add app/embedding.py tests/test_embedding.py
git commit -m "feat: BGE 向量化模块（懒加载/指令前缀/设备回退）"
```

---

### Task 3: 向量同步模块 app/vector_sync.py

**Files:**
- Create: `app/vector_sync.py`
- Test: `tests/test_vector_sync.py`

**Interfaces:**
- Consumes: `Settings.es_*`、`Settings.embedding_*`、`Settings.hybrid_search`（Task 1）；`embed_documents`（Task 2，可注入）；`build_es_client`（来自 `app.agent.tools._client`）
- Produces:
  - `async ensure_mapping(settings, client=None) -> None`——PUT mapping 补 `content_vector` 字段（幂等）
  - `async sync_vectors(settings, client=None, embed_documents=None) -> dict`——增量：只给缺向量的文档生成并写回，返回 `{"scanned","missing","updated","failed"}`
  - `async rebuild_vectors(settings, client=None, embed_documents=None) -> dict`——`_update_by_query` 清空所有 `content_vector` 后全量 `sync_vectors`
  - 三者 `client` 均可为 `None`（内部用 `build_es_client(settings)` 自建）；`embed_documents` 为 `None` 时导入 `app.embedding.embed_documents`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_vector_sync.py`：

```python
# -*- coding: utf-8 -*-
"""向量同步单测：mapping 幂等、增量只补缺失、rebuild 先清空、单条失败继续（MockTransport 全程拦截）。"""
import json

import httpx
import pytest

from app.config import Settings
from app.vector_sync import ensure_mapping, rebuild_vectors, sync_vectors


def make_settings() -> Settings:
    return Settings(
        es_url="http://es.test:9200",
        es_username="elastic",
        es_password="pw123",
        hybrid_search=True,
        _env_file=None,
    )


def fake_embed_documents(texts):
    return [[float(len(t)), 0.0] for t in texts]


def test_ensure_mapping_puts_mapping():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"acknowledged": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    asyncio_run(ensure_mapping(make_settings(), client))
    assert captured["method"] == "PUT"
    assert captured["url"] == "http://es.test/articles/_mapping"
    props = captured["body"]["properties"]["content_vector"]
    assert props["type"] == "dense_vector"
    assert props["dims"] == 1024
    assert props["similarity"] == "l2_norm"


def test_sync_vectors_only_updates_missing():
    calls = []

    def make_handler():
        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.method, str(request.url), json.loads(request.content)))
            if request.method == "PUT":
                return httpx.Response(200, json={"acknowledged": True})
            if str(request.url).endswith("/_search"):
                return httpx.Response(200, json={"hits": {"hits": [
                    {"_source": {"id": "a1", "content": "SSE协议", "content_vector": [1.0]}},
                    {"_source": {"id": "a2", "content": "响应式编程"}},
                ]}})
            if "/_update/a2" in str(request.url):
                return httpx.Response(200, json={"result": "updated"})
            raise AssertionError(f"unexpected: {request.method} {request.url}")
        return handler

    client = httpx.AsyncClient(transport=httpx.MockTransport(make_handler()), base_url="http://es.test")
    result = asyncio_run(sync_vectors(make_settings(), client, embed_documents=fake_embed_documents))
    assert result == {"scanned": 2, "missing": 1, "updated": 1, "failed": 0}
    # 只有 a2（缺向量）被写回，且没发生 a1 的更新请求
    update_calls = [c for c in calls if c[0] == "POST" and "/_update/" in c[1]]
    assert len(update_calls) == 1
    assert update_calls[0][1] == "http://es.test/articles/_update/a2"
    # 文档侧向量不带查询指令前缀（fake 按原文长度编码，直接断言 content 原样传入）
    assert fake_embed_documents(["响应式编程"]) == [[5.0, 0.0]]


def test_sync_vectors_continues_on_single_failure():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            return httpx.Response(200, json={"acknowledged": True})
        if str(request.url).endswith("/_search"):
            return httpx.Response(200, json={"hits": {"hits": [
                {"_source": {"id": "a1", "content": "AAA"}},
                {"_source": {"id": "a2", "content": "BBBB"}},
            ]}})
        if "/_update/a1" in str(request.url):
            raise httpx.ConnectError("es down", request=request)
        return httpx.Response(200, json={"result": "updated"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    result = asyncio_run(sync_vectors(make_settings(), client, embed_documents=fake_embed_documents))
    assert result["updated"] == 1
    assert result["failed"] == 1


def test_rebuild_clears_then_syncs():
    methods = []

    async def handler(request: httpx.Request) -> httpx.Response:
        methods.append((request.method, str(request.url), json.loads(request.content)))
        if request.method == "PUT":
            return httpx.Response(200, json={"acknowledged": True})
        if str(request.url).endswith("/_search"):
            return httpx.Response(200, json={"hits": {"hits": [
                {"_source": {"id": "a1", "content": "AAA"}},
            ]}})
        if str(request.url).endswith("/_update_by_query"):
            return httpx.Response(200, json={"updated": 12})
        if "/_update/a1" in str(request.url):
            return httpx.Response(200, json={"result": "updated"})
        raise AssertionError(f"unexpected: {request.method} {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    result = asyncio_run(rebuild_vectors(make_settings(), client, embed_documents=fake_embed_documents))
    assert result["updated"] == 1
    ubq = [m for m in methods if m[1].endswith("/_update_by_query")]
    assert len(ubq) == 1
    assert ubq[0][2]["script"]["source"] == "ctx._source.remove('content_vector')"


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_vector_sync.py -v`
Expected: FAIL（`ModuleNotFoundError: app.vector_sync`）

- [ ] **Step 3: 实现**

创建 `app/vector_sync.py`：

```python
# -*- coding: utf-8 -*-
"""向量同步：确保 mapping、增量补向量、全量重建（规格 2026-09-04 §4.2）。"""
import logging

import httpx

from app.agent.tools._client import build_es_client
from app.config import Settings

logger = logging.getLogger(__name__)

_MAPPING_BODY = {
    "properties": {
        "content_vector": {
            "type": "dense_vector",
            "dims": 1024,
            "index": True,
            "similarity": "l2_norm",
        }
    }
}


def _resolve_client(settings: Settings, client: httpx.AsyncClient | None) -> httpx.AsyncClient:
    return client or build_es_client(settings)


def _resolve_embed_documents(embed_documents):
    if embed_documents is None:
        from app.embedding import embed_documents as _impl
        return _impl
    return embed_documents


def _auth(settings: Settings):
    return (settings.es_username, settings.es_password) if settings.es_username else None


async def ensure_mapping(settings: Settings, client: httpx.AsyncClient | None = None) -> None:
    """补 content_vector 字段（幂等：已存在则 ES 返回 200）。"""
    client = _resolve_client(settings, client)
    resp = await client.put("/articles/_mapping", json=_MAPPING_BODY, auth=_auth(settings))
    resp.raise_for_status()


async def sync_vectors(
    settings: Settings,
    client: httpx.AsyncClient | None = None,
    embed_documents=None,
) -> dict:
    """增量同步：只给缺 content_vector 的文档生成向量并写回。"""
    client = _resolve_client(settings, client)
    embed_documents = _resolve_embed_documents(embed_documents)
    await ensure_mapping(settings, client)
    resp = await client.post(
        "/articles/_search",
        json={"query": {"match_all": {}}, "size": 10000, "_source": ["id", "content"]},
        auth=_auth(settings),
    )
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    missing = [h["_source"] for h in hits if not h.get("_source", {}).get("content_vector")]
    vectors = embed_documents([doc.get("content") or "" for doc in missing]) if missing else []
    updated = failed = 0
    for doc, vec in zip(missing, vectors):
        try:
            await client.post(
                f"/articles/_update/{doc['id']}",
                json={"doc": {"content_vector": vec}},
                auth=_auth(settings),
            )
            updated += 1
        except httpx.HTTPError as e:
            logger.warning("写回向量失败 id=%s: %s", doc.get("id"), e)
            failed += 1
    logger.info("向量同步：扫描 %d，缺失 %d，成功 %d，失败 %d", len(hits), len(missing), updated, failed)
    return {"scanned": len(hits), "missing": len(missing), "updated": updated, "failed": failed}


async def rebuild_vectors(
    settings: Settings,
    client: httpx.AsyncClient | None = None,
    embed_documents=None,
) -> dict:
    """全量重建：清空所有 content_vector 后重新同步。"""
    client = _resolve_client(settings, client)
    await ensure_mapping(settings, client)
    await client.post(
        "/articles/_update_by_query",
        json={"script": {"source": "ctx._source.remove('content_vector')"}},
        auth=_auth(settings),
    )
    return await sync_vectors(settings, client=client, embed_documents=embed_documents)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_vector_sync.py -v`
Expected: PASS（4 个）

- [ ] **Step 5: 提交**

```bash
git add app/vector_sync.py tests/test_vector_sync.py
git commit -m "feat: 文章向量同步模块（mapping/增量/全量重建）"
```

---

### Task 4: search_articles 混合检索改造

**Files:**
- Modify: `app/agent/tools/search_articles.py`（整体改写）
- Modify: `app/agent/tools/__init__.py`（`build_tools` 透传 `embed_query`）
- Test: `tests/test_tools.py`（适配现有 6 个 + 新增混合检索用例）

**Interfaces:**
- Consumes: `Settings.hybrid_search`、`Settings.search_page_size`（Task 1）；`embed_query`（Task 2，可注入）；`client`（现有注入模式）
- Produces:
  - `build_search_articles(settings, client, embed_query=None) -> BaseTool`——`embed_query` 为 `None` 时导入 `app.embedding.embed_query`
  - `build_tools(settings, client=None, embed_query=None) -> list`——透传给 `build_search_articles`（graph.py 的 `build_tools(settings)` 调用不受影响）
  - 返回工具名与顺序不变：`[search_articles, get_article_content, list_articles]`

- [ ] **Step 1: 改写 search_articles 并适配测试（先写测试再改实现，测试先行）**

在 `tests/test_tools.py` 顶部加 fake（复用现有 `make_settings`）：

```python
def fake_embed_query(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]
```

改写现有 `test_search_articles_query_shape_and_auth`，把 `build_tools(make_settings(), client=client)` 改为 `build_tools(make_settings(), client=client, embed_query=fake_embed_query)`，并在原断言后追加混合 body 断言：

```python
    # 混合检索：knn + rank.rrf，knn 过滤与 query 过滤同域
    assert body["knn"]["field"] == "content_vector"
    assert body["knn"]["query_vector"] == [0.1, 0.2, 0.3]
    assert body["knn"]["filter"] == {"term": {"status": "published"}}
    assert body["knn"]["k"] == 5
    assert body["rank"] == {"rrf": {}}
```

其余现有测试 `test_get_article_content_truncates`、`test_get_article_content_not_found`、`test_list_articles_shape` 不改（不涉及 search）；`test_tool_error_returns_text_not_raise` 和 `test_no_auth_when_username_empty` 的 `build_tools` 调用同样加 `embed_query=fake_embed_query`（避免真实加载 BGE）。

新增 3 个用例：

```python
@pytest.mark.asyncio
async def test_hybrid_disabled_uses_bm25_only():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"hits": {"hits": []}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    settings = make_settings()
    settings.hybrid_search = False
    tools = build_tools(settings, client=client, embed_query=fake_embed_query)
    await tools[0].ainvoke({"keyword": "微服务"})
    assert "knn" not in captured["body"]
    assert "rank" not in captured["body"]


@pytest.mark.asyncio
async def test_hybrid_falls_back_to_bm25_when_embed_fails():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"hits": {"hits": []}})

    def broken_embed(text):
        raise RuntimeError("model load failed")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client, embed_query=broken_embed)
    result = await tools[0].ainvoke({"keyword": "微服务"})
    assert "knn" not in captured["body"]
    assert json.loads(result) == []


@pytest.mark.asyncio
async def test_hybrid_400_falls_back_to_bm25():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "knn" in body:
            return httpx.Response(400, json={"error": {"type": "search_phase_execution_exception"}})
        return httpx.Response(200, json={"hits": {"hits": [SEARCH_HIT]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client, embed_query=fake_embed_query)
    result = await tools[0].ainvoke({"keyword": "微服务"})
    assert len(calls) == 2
    assert "knn" in calls[0] and "knn" not in calls[1]
    assert "SSE协议" in result
```

- [ ] **Step 2: 跑测试确认失败**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_tools.py -v`
Expected: 新用例 FAIL（body 无 `knn`/`rank`；`build_tools` 尚未接受 `embed_query` 参数）

- [ ] **Step 3: 实现**

改写 `app/agent/tools/search_articles.py`：

```python
# -*- coding: utf-8 -*-
"""工具：按关键词搜索已发布文章（search_articles），BM25+向量混合检索（规格 2026-09-04 §5）。"""
import json
import logging

import httpx
from langchain_core.tools import BaseTool, tool

from app.config import Settings

from ._client import _clean_highlight

logger = logging.getLogger(__name__)


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


def _hybrid_body(keyword: str, query_vector: list[float], page_size: int) -> dict:
    body = _bm25_body(keyword, page_size)
    return {
        "knn": {
            "field": "content_vector",
            "query_vector": query_vector,
            "k": page_size,
            "filter": {"term": {"status": "published"}},
        },
        "query": body["query"],
        "rank": {"rrf": {}},
        "highlight": body["highlight"],
        "size": body["size"],
        "_source": body["_source"],
    }


def build_search_articles(
    settings: Settings,
    client: httpx.AsyncClient,
    embed_query=None,
) -> BaseTool:
    # 工具请求始终按 settings 的凭据认证 ES：外部传入的 client（如测试的 MockTransport）可能不含 auth
    auth = (settings.es_username, settings.es_password) if settings.es_username else None
    if embed_query is None:
        from app.embedding import embed_query as _default_embed_query
        embed_query = _default_embed_query

    @tool
    async def search_articles(keyword: str) -> str:
        """按关键词搜索博客已发布文章（BM25+向量混合检索）。

        Args:
            keyword: 搜索关键词，如“微服务”“SSE”
        Returns:
            JSON 列表：每项含 id、title、tags、createdAt 与命中内容片段（highlight）
        """
        query_vector = None
        if settings.hybrid_search:
            try:
                query_vector = embed_query(keyword)
            except Exception as e:
                logger.warning("embedding 失败，回退纯 BM25：%s", e)
        body = _hybrid_body(keyword, query_vector, settings.search_page_size) if query_vector is not None else _bm25_body(keyword, settings.search_page_size)
        try:
            resp = await client.post("/articles/_search", json=body, auth=auth)
            if resp.status_code == 400 and query_vector is not None:
                # content_vector 字段缺失等 knn 错误 → 回退纯 BM25 重试
                logger.warning("混合检索返回 400，回退纯 BM25")
                resp = await client.post("/articles/_search", json=_bm25_body(keyword, settings.search_page_size), auth=auth)
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
```

改 `app/agent/tools/__init__.py` 的 `build_tools` 签名与列表构造：

```python
def build_tools(settings: Settings, client: httpx.AsyncClient | None = None, embed_query=None) -> list:
    client = client or build_es_client(settings)
    return [
        build_search_articles(settings, client, embed_query=embed_query),
        build_get_article_content(settings, client),
        build_list_articles(settings, client),
    ]
```

（`__init__.py` 顶部保留原导入不变，`build_search_articles` 已存在。）

- [ ] **Step 4: 跑测试确认通过**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_tools.py -v`
Expected: PASS（9 个：原 6 个 + 新增 3 个）

- [ ] **Step 5: 全量回归**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest -q`
Expected: 67 通过 + 1 预存失败（`test_config.py::test_defaults`，与本改动无关）

- [ ] **Step 6: 提交**

```bash
git add app/agent/tools/search_articles.py app/agent/tools/__init__.py tests/test_tools.py
git commit -m "feat: search_articles 混合检索（knn+BM25+RRF）与降级回退"
```

---

### Task 5: 启动同步 lifespan + 手动重建脚本

**Files:**
- Modify: `app/main.py`（新增 lifespan，启动时增量同步向量）
- Create: `scripts/rebuild_vectors.py`
- Test: `tests/test_lifespan.py`（新建）

**Interfaces:**
- Consumes: `sync_vectors`（Task 3，`app.main` 内 import 后由测试 monkeypatch）、`Settings.hybrid_search`
- Produces: `app.main` 的 `lifespan` 上下文管理器（FastAPI `lifespan=` 参数）；`scripts/rebuild_vectors.py` 独立入口

- [ ] **Step 1: 写失败测试**

创建 `tests/test_lifespan.py`：

```python
# -*- coding: utf-8 -*-
"""lifespan 测试：启动触发向量同步且异常不阻断服务（sync_vectors 全程 mock，不加载真实 BGE）。"""
from fastapi.testclient import TestClient


def test_lifespan_syncs_vectors_on_startup(monkeypatch):
    called = {}

    async def fake_sync(settings):
        called["yes"] = True
        return {"scanned": 0, "missing": 0, "updated": 0, "failed": 0}

    monkeypatch.setattr("app.main.sync_vectors", fake_sync)
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/chat/stop", json={"conversationId": "x"})
        assert resp.status_code == 200
    assert called.get("yes") is True


def test_lifespan_tolerates_sync_failure(monkeypatch):
    async def bad_sync(settings):
        raise RuntimeError("ES down")

    monkeypatch.setattr("app.main.sync_vectors", bad_sync)
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/chat/stop", json={"conversationId": "x"})
        assert resp.status_code == 200  # 同步失败不阻断启动
```

- [ ] **Step 2: 跑测试确认失败**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_lifespan.py -v`
Expected: FAIL（`TypeError: FastAPI() got an unexpected keyword argument 'lifespan'` 或 `AttributeError: module 'app.main' has no attribute 'sync_vectors'`）

- [ ] **Step 3: 实现**

`app/main.py` 修改（顶部 import 区加两行，`app = FastAPI(...)` 处改）：

```python
from contextlib import asynccontextmanager

from app.vector_sync import sync_vectors
```

在 `app = FastAPI(title="blog-agent")` 之前加：

```python
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时增量同步文章向量；失败仅告警，不阻断服务（降级为纯 BM25）。"""
    if settings.hybrid_search:
        try:
            result = await sync_vectors(settings)
            logger.info("启动向量同步完成：%s", result)
        except Exception:
            logger.warning("启动向量同步失败，检索将回退纯 BM25", exc_info=True)
    yield
```

改为：

```python
app = FastAPI(title="blog-agent", lifespan=lifespan)
```

创建 `scripts/rebuild_vectors.py`：

```python
# -*- coding: utf-8 -*-
"""手动全量重建文章向量：清空 content_vector 后重新生成写回。

用法:
  /d/tool1/anancoda/envs/ai-agent/python.exe scripts/rebuild_vectors.py
"""
import asyncio
import logging

from app.config import get_settings
from app.vector_sync import rebuild_vectors

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


async def main() -> None:
    result = await rebuild_vectors(get_settings())
    print("重建完成:", result)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_lifespan.py -v`
Expected: PASS（2 个）

- [ ] **Step 5: 全量回归**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest -q`
Expected: 全部通过（含 Task 4 的 67+1，加上 lifespan 2 个与 embedding 4 个、vector_sync 4 个、config 新增 1 个——总量按实际跑数核对，唯一允许失败仍是 `test_config.py::test_defaults` 预存项）

- [ ] **Step 6: 提交**

```bash
git add app/main.py scripts/rebuild_vectors.py tests/test_lifespan.py
git commit -m "feat: 启动向量同步（lifespan）+ 手动全量重建脚本"
```

---

## 手工验证（可选，实现完跑一遍真 ES）

1. 跑重建脚本：`/d/tool1/anancoda/envs/ai-agent/python.exe scripts/rebuild_vectors.py`（对着 `100.110.148.14:9200`，首次会 PUT mapping 补字段）
2. 验证向量落库：查询 `http://100.110.148.14:9200/articles/_search` 确认 `content_vector` 有值、长度 1024
3. 启动服务 `uvicorn app.main:app`，问 "博客里有没有关于 SSE 的文章" 确认混合检索命中；把 `HYBRID_SEARCH=false` 重启再验证回退路径
