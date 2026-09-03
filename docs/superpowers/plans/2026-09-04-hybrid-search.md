# search_articles 混合检索（BM25+向量+RRF）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `search_articles` 升级为 BM25 关键词 + BGE 向量双路综合检索（Python 侧 RRF 融合），文章按"段落优先 + token 兜底 + 重叠"切 chunk 存独立 `article_chunks` 索引。

**Architecture:** BGE（本地 GPU）生成 1024 维向量；文章按段落优先规则切 chunk，逐 chunk 嵌入写入独立 `article_chunks` 索引（chunk 即文档，冗余文章元数据）；`search_articles` 两路独立请求——BM25 查 articles、knn 查 article_chunks，chunk 按 article_id 归并回文章级，Python 侧 RRF 融合取 top。启动时全量重建 chunk 索引，`scripts/rebuild_vectors.py` 手动重建。任何环节失败降级为仅 BM25 路，不崩服务。

**Tech Stack:** Python 3.12 / FastAPI / langchain-core / httpx / sentence-transformers / Elasticsearch 8.17（knn + dense_vector）

**Spec:** `docs/superpowers/specs/2026-09-04-hybrid-search-design.md`（v2）

## Global Constraints

- BGE：`F:/models/bge-large-zh-v1.5`（1024 维，上下文 512 token），设备 `cuda`（失败回退 `cpu`）；**查询侧加指令前缀** `"为这个句子生成表示以用于检索相关文章:"`，**文档侧不加**
- `article_chunks` 索引：文档 id=`{article_id}-{chunk_index}`；字段 `article_id(keyword)/chunk_index(integer)/content(text)/title(text)/tags(keyword)/createdAt(date)/content_vector(dense_vector, dims=1024, index=true, similarity=l2_norm)`
- 切块：段落优先（代码块独立段、按空行分段）+ 段累积到 `max_tokens` + 超长段 token 窗口重叠切（`overlap_tokens`）；段落级 chunk 间不重叠
- RRF：`score = Σ 1/(60 + rank + 1)`，knn 路 chunk 按 article_id 归并取**最小 rank**
- **Java 端零改动**：articles 索引不动，chunk 索引由 Python 侧建
- 降级哲学（规格 §6）：任何一路失败 → 仅另一路结果，服务不崩；工具执行失败不抛异常，错误文本回模型兜底
- knn 的 `filter` 与 BM25 路 `bool.filter` 一致（`status: published`）
- 测试命令统一：`MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest <file> -v`（conda 环境 ai-agent，ES 请求一律 MockTransport 拦截，**不加载真实 BGE 模型**——embed/count_tokens 全部注入 fake）
- commit 信息用中文 + `feat:`/`refactor:`/`test:` 前缀，不加 AI 尾注（用户偏好）

---

### Task 1: 新增混合检索与切块配置项

**Files:**
- Modify: `app/config.py`（在 `search_page_size` 附近加 5 项）
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.embedding_model_path: str = "F:/models/bge-large-zh-v1.5"`、`Settings.embedding_device: str = "cuda"`、`Settings.hybrid_search: bool = True`、`Settings.chunk_max_tokens: int = 256`、`Settings.chunk_overlap_tokens: int = 32`——后续所有任务读取

- [ ] **Step 1: 写失败测试**

在 `tests/test_config.py` 追加（先读该文件，确认现有 `Settings(_env_file=None)` 的写法后按同样模式写）：

```python
def test_hybrid_search_defaults():
    s = Settings(_env_file=None)
    assert s.embedding_model_path == "F:/models/bge-large-zh-v1.5"
    assert s.embedding_device == "cuda"
    assert s.hybrid_search is True
    assert s.chunk_max_tokens == 256
    assert s.chunk_overlap_tokens == 32
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
    chunk_max_tokens: int = 256  # 切块目标 token 数
    chunk_overlap_tokens: int = 32  # 超长段窗口重叠 token 数
```

- [ ] **Step 4: 跑测试确认通过**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: 新增混合检索与切块配置项"
```

---

### Task 2: BGE 向量化模块 app/embedding.py

**Files:**
- Create: `app/embedding.py`
- Test: `tests/test_embedding.py`

**Interfaces:**
- Consumes: `Settings.embedding_model_path`、`Settings.embedding_device`（Task 1）
- Produces:
  - `get_embedder() -> SentenceTransformer`——懒加载 + 模块级 `_embedder` 缓存；指定设备加载失败回退 `cpu`；实例含 `.tokenizer`（vector_sync 计 token 用）
  - `embed_query(text: str) -> list[float]`——输入加查询指令前缀后编码
  - `embed_documents(texts: list[str]) -> list[list[float]]`——不加前缀；空文本兜底为单空格

- [ ] **Step 1: 写失败测试**

创建 `tests/test_embedding.py`：

```python
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

### Task 3: 切块纯函数 app/chunking.py

**Files:**
- Create: `app/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: 无（纯函数，`count_tokens` 由调用方注入）
- Produces:
  - `split_content(content: str, count_tokens: Callable[[str], int], max_tokens: int = 256, overlap_tokens: int = 32) -> list[str]`——段落优先 + token 兜底 + 重叠；空输入返回 `[]`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_chunking.py`：

```python
# -*- coding: utf-8 -*-
"""切块单测：段落合并、代码块独立、超长段窗口重叠、边界防御（count_tokens 用 len 注入）。"""
from app.chunking import split_content


def test_empty_content():
    assert split_content("", count_tokens=len) == []
    assert split_content("   \n  ", count_tokens=len) == []


def test_short_paragraphs_merge_into_one_chunk():
    text = "第一段。\n\n第二段。\n\n第三段。"
    chunks = split_content(text, count_tokens=len, max_tokens=100)
    assert len(chunks) == 1
    assert "第一段" in chunks[0] and "第三段" in chunks[0]


def test_paragraph_exceeding_max_starts_new_chunk():
    text = "A" * 60 + "\n\n" + "B" * 60
    chunks = split_content(text, count_tokens=len, max_tokens=50)
    assert len(chunks) == 2
    assert chunks[0] == "A" * 60  # 单段超限不切（段落完整），仅分段


def count_lines(text: str) -> int:
    """测试用 token 计数：按行数（每行 1 token）。"""
    return text.count("\n") + 1


def test_code_block_is_not_split():
    text = "介绍段落。\n```python\ncode_a = 1\ncode_b = 2\n```\n结尾段落。"
    chunks = split_content(text, count_tokens=len, max_tokens=50)
    assert len(chunks) == 1  # 短内容合并为一个 chunk
    assert "```python\ncode_a = 1\ncode_b = 2\n```" in chunks[0]  # 代码块完整未被切开


def test_long_block_windows_overlap():
    # 每行 1 token（按行计数），max_tokens=10，overlap=3 → 窗口 10 行、重叠 3 token
    text = "\n".join(f"L{i}" for i in range(30))
    chunks = split_content(text, count_tokens=count_lines, max_tokens=10, overlap_tokens=3)
    assert len(chunks) >= 4
    assert chunks[0].splitlines()[0] == "L0"
    assert chunks[1].splitlines()[0] == "L7"  # 10 - 3 = 7，下一窗口重叠 3 token
    assert chunks[0].splitlines()[-1] == "L9"


def test_long_block_zero_overlap_no_infinite_loop():
    text = "\n".join(f"L{i}" for i in range(30))
    chunks = split_content(text, count_tokens=count_lines, max_tokens=10, overlap_tokens=0)
    assert len(chunks) >= 3  # 至少前进一行，不陷入死循环
```

- [ ] **Step 2: 跑测试确认失败**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_chunking.py -v`
Expected: FAIL（`ModuleNotFoundError: app.chunking`）

- [ ] **Step 3: 实现**

创建 `app/chunking.py`：

```python
# -*- coding: utf-8 -*-
"""文章切块：段落优先 + token 兜底 + 重叠（规格 2026-09-04 §4.2）。纯函数，不依赖模型。"""
from collections.abc import Callable

_FENCE = "```"


def split_content(
    content: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 256,
    overlap_tokens: int = 32,
) -> list[str]:
    """段落优先切块：代码块独立段、段落合并累积、超长段 token 窗口重叠切。"""
    if not content or not content.strip():
        return []
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for block in _split_blocks(content):
        block_tokens = count_tokens(block)
        if buf and buf_tokens + block_tokens > max_tokens:
            chunks.append("\n\n".join(buf))  # 段落间保留空行
            buf, buf_tokens = [], 0
        if block_tokens > max_tokens:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, buf_tokens = [], 0
            chunks.extend(_split_long_block(block, count_tokens, max_tokens, overlap_tokens))
        else:
            buf.append(block)
            buf_tokens += block_tokens
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _split_blocks(content: str) -> list[str]:
    """Markdown 分段：代码块整体独立段，其余按空行分隔。"""
    blocks: list[str] = []
    buf: list[str] = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(_FENCE):
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            # 收集整个代码块（含首尾围栏）为一个 block
            code_lines = [lines[i]]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(_FENCE):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):  # 闭合围栏
                code_lines.append(lines[i])
                i += 1
            blocks.append("\n".join(code_lines))
        elif stripped:
            buf.append(lines[i])
            i += 1
        else:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            i += 1
    if buf:
        blocks.append("\n".join(buf))
    return blocks


def _split_long_block(
    block: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """超长段按 token 窗口切，窗口间重叠 overlap_tokens（按行前缀和定位重叠点）。"""
    lines = block.splitlines() or [""]
    line_tokens = [max(count_tokens(line), 1) for line in lines]  # 每行至少 1 token，防空行 0
    prefix = [0]
    for t in line_tokens:
        prefix.append(prefix[-1] + t)
    chunks: list[str] = []
    start = 0
    n = len(lines)
    while start < n:
        end = start
        while end < n and prefix[end + 1] - prefix[start] < max_tokens:
            end += 1
        end = max(end, start + 1)  # 至少一行，防死循环；单行超限则该行自成一 chunk
        chunks.append("\n".join(lines[start:end]))
        if end >= n:
            break
        target = prefix[end] - overlap_tokens  # 下一窗口起点：与当前窗口尾部重叠 overlap_tokens
        start = end
        while start > 0 and prefix[start] > target:
            start -= 1
        if start >= end:  # 防御（overlap=0 等）：至少前进一行
            start = end - 1
    return chunks
```

- [ ] **Step 4: 跑测试确认通过**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_chunking.py -v`
Expected: PASS（6 个）

- [ ] **Step 5: 提交**

```bash
git add app/chunking.py tests/test_chunking.py
git commit -m "feat: 文章切块模块（段落优先+token 兜底+重叠）"
```

---

### Task 4: 向量同步模块 app/vector_sync.py

**Files:**
- Create: `app/vector_sync.py`
- Test: `tests/test_vector_sync.py`

**Interfaces:**
- Consumes: `Settings.es_*`、`Settings.chunk_*`、`Settings.hybrid_search`（Task 1）；`embed_documents`（Task 2，可注入）；`split_content`（Task 3）
- Produces:
  - `async ensure_index(settings, client=None) -> None`——PUT 建 `article_chunks` 索引（幂等：400 + `resource_already_exists_exception` 视为成功）
  - `async sync_vectors(settings, client=None, embed_documents=None, count_tokens=None) -> dict`——**全量重建**：删旧索引（404 忽略）→ 建新 → 读 articles → 切块 → 嵌入 → 逐 chunk `PUT /article_chunks/_doc/{article_id}-{i}`。返回 `{"articles","chunks","updated","failed"}`
  - `async rebuild_vectors(settings, client=None, embed_documents=None, count_tokens=None) -> dict`——等价全量重建，供手动脚本
  - `embed_documents`/`count_tokens` 为 `None` 时默认取 `app.embedding`（`count_tokens` 默认用 `get_embedder().tokenizer` 计数）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_vector_sync.py`：

```python
# -*- coding: utf-8 -*-
"""向量同步单测：索引幂等、全量重建、chunk 写字段、单条失败继续（MockTransport 全程拦截）。"""
import json

import httpx
import pytest

from app.config import Settings
from app.vector_sync import ensure_index, rebuild_vectors, sync_vectors


def make_settings() -> Settings:
    return Settings(
        es_url="http://es.test:9200",
        es_username="elastic",
        es_password="pw123",
        hybrid_search=True,
        chunk_max_tokens=50,
        chunk_overlap_tokens=10,
        _env_file=None,
    )


def fake_embed_documents(texts):
    return [[float(len(t)), 0.0] for t in texts]


def run(coro):
    import asyncio
    return asyncio.run(coro)


def test_ensure_index_power_idempotent():
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.method == "PUT":
            return httpx.Response(400, json={}, text='{"error":{"type":"resource_already_exists_exception"}}')
        raise AssertionError(f"unexpected: {request.method} {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    run(ensure_index(make_settings(), client))  # 不抛异常
    assert calls == ["http://es.test/article_chunks"]


def test_sync_vectors_rebuilds_chunk_index():
    calls = []

    def make_handler():
        async def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            method = request.method
            body = json.loads(request.content) if request.content else None
            calls.append((method, url, body))
            if method == "DELETE" and url.endswith("/article_chunks"):
                return httpx.Response(200, json={"acknowledged": True})
            if method == "PUT" and url.endswith("/article_chunks"):
                return httpx.Response(200, json={"acknowledged": True})
            if url.endswith("/articles/_search"):
                return httpx.Response(200, json={"hits": {"hits": [
                    {"_source": {"id": "a1", "title": "SSE协议", "tags": [], "createdAt": "2026-08-15",
                                 "content": "第一段。\n\n第二段。"}},
                ]}})
            if method == "PUT" and "/article_chunks/_doc/" in url:
                return httpx.Response(200, json={"result": "created"})
            raise AssertionError(f"unexpected: {method} {url}")
        return handler

    client = httpx.AsyncClient(transport=httpx.MockTransport(make_handler()), base_url="http://es.test")
    result = run(sync_vectors(make_settings(), client, embed_documents=fake_embed_documents, count_tokens=len))
    assert result == {"articles": 1, "chunks": 1, "updated": 1, "failed": 0}
    # 删除旧索引 → 重建 → 读文章 → 写 chunk（4 个请求，顺序断言）
    assert calls[0][0] == "DELETE" and calls[0][1].endswith("/article_chunks")
    assert calls[1][0] == "PUT" and calls[1][1].endswith("/article_chunks")
    assert calls[2][1].endswith("/articles/_search")
    doc_call = calls[3]
    assert doc_call[1] == "http://es.test/article_chunks/_doc/a1-0"
    doc = doc_call[2]
    assert doc["article_id"] == "a1" and doc["chunk_index"] == 0
    assert doc["title"] == "SSE协议" and doc["content"] == "第一段。\n\n第二段。"
    assert doc["content_vector"] == [10.0, 0.0]  # fake: len("第一段。\n\n第二段。")=10


def test_sync_vectors_continues_on_single_failure():
    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "DELETE" or (request.method == "PUT" and url.endswith("/article_chunks")):
            return httpx.Response(200, json={"acknowledged": True})
        if url.endswith("/articles/_search"):
            return httpx.Response(200, json={"hits": {"hits": [
                {"_source": {"id": "a1", "title": "T", "tags": [], "createdAt": None, "content": "AAA"}},
                {"_source": {"id": "a2", "title": "U", "tags": [], "createdAt": None, "content": "BBBB"}},
            ]}})
        if "/article_chunks/_doc/" in url and url.endswith("/a1-0"):
            raise httpx.ConnectError("es down", request=request)
        return httpx.Response(200, json={"result": "created"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    result = run(sync_vectors(make_settings(), client, embed_documents=fake_embed_documents, count_tokens=len))
    assert result["updated"] == 1
    assert result["failed"] == 1


def test_rebuild_vectors_equals_sync():
    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "DELETE" or (request.method == "PUT" and url.endswith("/article_chunks")):
            return httpx.Response(200, json={"acknowledged": True})
        if url.endswith("/articles/_search"):
            return httpx.Response(200, json={"hits": {"hits": []}})
        raise AssertionError(f"unexpected: {request.method} {url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    result = run(rebuild_vectors(make_settings(), client, embed_documents=fake_embed_documents, count_tokens=len))
    assert result == {"articles": 0, "chunks": 0, "updated": 0, "failed": 0}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_vector_sync.py -v`
Expected: FAIL（`ModuleNotFoundError: app.vector_sync`）

- [ ] **Step 3: 实现**

创建 `app/vector_sync.py`：

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_vector_sync.py -v`
Expected: PASS（4 个）

- [ ] **Step 5: 提交**

```bash
git add app/vector_sync.py tests/test_vector_sync.py
git commit -m "feat: article_chunks 索引全量重建同步模块"
```

---

### Task 5: search_articles 两路混合检索 + Python RRF

**Files:**
- Modify: `app/agent/tools/search_articles.py`（整体改写）
- Modify: `app/agent/tools/__init__.py`（`build_tools` 透传 `embed_query`）
- Test: `tests/test_tools.py`（适配现有 6 个 + 新增混合检索用例）

**Interfaces:**
- Consumes: `Settings.hybrid_search`、`Settings.search_page_size`（Task 1）；`embed_query`（Task 2，可注入）；`client`（现有注入模式）
- Produces:
  - `build_search_articles(settings, client, embed_query=None) -> BaseTool`——`embed_query` 为 `None` 时导入 `app.embedding.embed_query`
  - `_fuse(bm25_hits: list[dict], knn_hits: list[dict], page_size: int) -> list[dict]`——两路文章级 RRF 融合（模块级函数，直接单测）
  - `build_tools(settings, client=None, embed_query=None) -> list`——透传（graph.py 的 `build_tools(settings)` 不受影响）；返回工具顺序不变

- [ ] **Step 1: 先写 _fuse 纯函数测试（测试先行）**

在 `tests/test_tools.py` 顶部加 import 与两个纯函数用例：

```python
from app.agent.tools.search_articles import _fuse
```

```python
def test_fuse_two_route_rrf_ranking():
    bm25_hits = [
        {"_source": {"id": "A", "title": "A文", "tags": [], "createdAt": "t"},
         "highlight": {"content": ["<em>微</em>服务命中"]}},
        {"_source": {"id": "B", "title": "B文", "tags": [], "createdAt": "t"}},
    ]
    knn_hits = [
        {"_source": {"article_id": "B", "title": "B文", "tags": [], "createdAt": "t", "content": "B 的 chunk"}},
        {"_source": {"article_id": "C", "title": "C文", "tags": [], "createdAt": "t", "content": "C 的 chunk"}},
    ]
    items = _fuse(bm25_hits, knn_hits, page_size=5)
    assert [i["id"] for i in items] == ["B", "A", "C"]  # B: 1/62+1/61 最高；A: 1/61；C: 1/61
    assert items[0]["snippet"] == "B 的 chunk"[:150]  # knn-only 命中用 chunk 内容截取


def test_fuse_uses_highlight_snippet_and_dedup_chunks():
    bm25_hits = [
        {"_source": {"id": "A", "title": "A文", "tags": [], "createdAt": "t"},
         "highlight": {"content": ["<em>SSE</em>协议"]}},
    ]
    knn_hits = [
        {"_source": {"article_id": "A", "title": "A文", "tags": [], "createdAt": "t", "content": "c1"}},
        {"_source": {"article_id": "A", "title": "A文", "tags": [], "createdAt": "t", "content": "c2"}},
    ]
    items = _fuse(bm25_hits, knn_hits, page_size=5)
    assert len(items) == 1  # 同文章多 chunk 去重
    assert items[0]["snippet"] == "SSE协议"  # BM25 路 highlight 优先，去 em 标签
```

- [ ] **Step 2: 跑 _fuse 测试确认失败**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_tools.py::test_fuse_two_route_rrf_ranking tests/test_tools.py::test_fuse_uses_highlight_snippet_and_dedup_chunks -v`
Expected: FAIL（`ImportError: cannot import name '_fuse'`）

- [ ] **Step 3: 改写 search_articles 实现**

整体改写 `app/agent/tools/search_articles.py`：

```python
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
    if embed_query is None:
        from app.embedding import embed_query as _default_embed_query
        embed_query = _default_embed_query

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
            try:
                query_vector = embed_query(keyword)
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

- [ ] **Step 4: 适配与新增 search_articles 集成测试**

先读 `tests/test_tools.py` 现有内容。在顶部加：

```python
def fake_embed_query(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]
```

改造现有 `test_search_articles_query_shape_and_auth`：handler 改为分发两个请求并记录全部；`build_tools` 调用加 `embed_query=fake_embed_query`：

```python
@pytest.mark.asyncio
async def test_search_articles_query_shape_and_auth():
    """断言两路请求形状（BM25 body 原样、knn body 过滤一致）与 BasicAuth 头。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        captured.setdefault("urls", []).append(url)
        captured.setdefault("auths", []).append(request.headers.get("Authorization", ""))
        captured.setdefault("bodies", []).append(json.loads(request.content))
        if "/article_chunks/" in url:
            return httpx.Response(200, json={"hits": {"hits": []}})
        return httpx.Response(200, json={"hits": {"hits": [SEARCH_HIT]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client, embed_query=fake_embed_query)
    result = await tools[0].ainvoke({"keyword": "微服务"})

    assert captured["urls"] == [
        "http://es.test/articles/_search",
        "http://es.test/article_chunks/_search",
    ]
    assert all(a.startswith("Basic ") for a in captured["auths"])
    bm25, knn = captured["bodies"]
    assert bm25["query"]["bool"]["filter"] == {"term": {"status": "published"}}
    assert bm25["query"]["bool"]["must"]["multi_match"]["fields"] == ["title^3", "content", "summary"]
    assert bm25["size"] == 5
    assert "title" in bm25["highlight"]["fields"]
    assert knn["knn"]["field"] == "content_vector"
    assert knn["knn"]["query_vector"] == [0.1, 0.2, 0.3]
    assert knn["knn"]["filter"] == {"term": {"status": "published"}}  # 与 BM25 路过滤同域
    assert knn["knn"]["k"] == 25  # search_page_size * 5
    # 高亮片段去标签、结果可解析、含标题
    assert "<em>" not in result
    assert "SSE协议" in result
    assert json.loads(result)[0]["title"] == "SSE协议"
```

其余现有测试（`test_get_article_content_truncates`、`test_get_article_content_not_found`、`test_list_articles_shape`）不改；`test_tool_error_returns_text_not_raise` 与 `test_no_auth_when_username_empty` 的 `build_tools` 调用加 `embed_query=fake_embed_query`（handler 需对 `/article_chunks/` 请求返回空 hits，避免 404 影响断言）。

`test_no_auth_when_username_empty` 的 handler 改为：

```python
    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] += request.headers.get("Authorization", "")
        if "/article_chunks/" in str(request.url):
            return httpx.Response(200, json={"hits": {"hits": []}})
        return httpx.Response(200, json={"hits": {"hits": []}})
```

（`captured["auth"]` 初始化为 `""`，两路都断言后仍为空字符串。）

新增 3 个降级用例：

```python
@pytest.mark.asyncio
async def test_hybrid_disabled_only_bm25_request():
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"hits": {"hits": []}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    settings = make_settings()
    settings.hybrid_search = False
    tools = build_tools(settings, client=client, embed_query=fake_embed_query)
    await tools[0].ainvoke({"keyword": "微服务"})
    assert urls == ["http://es.test/articles/_search"]  # 只有 BM25 一路


@pytest.mark.asyncio
async def test_hybrid_falls_back_when_embed_fails():
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"hits": {"hits": []}})

    def broken_embed(text):
        raise RuntimeError("model load failed")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client, embed_query=broken_embed)
    result = await tools[0].ainvoke({"keyword": "微服务"})
    assert urls == ["http://es.test/articles/_search"]  # embed 失败 → 仅 BM25 路
    assert json.loads(result) == []


@pytest.mark.asyncio
async def test_knn_400_falls_back_to_bm25_only():
    urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        urls.append(url)
        if "/article_chunks/" in url:
            return httpx.Response(400, json={"error": {"type": "search_phase_execution_exception"}})
        return httpx.Response(200, json={"hits": {"hits": [SEARCH_HIT]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client, embed_query=fake_embed_query)
    result = await tools[0].ainvoke({"keyword": "微服务"})
    assert len(urls) == 2  # 发了两路，knn 400 被吞
    assert json.loads(result)[0]["title"] == "SSE协议"  # BM25 路结果照常返回
```

- [ ] **Step 5: 跑测试确认通过**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest tests/test_tools.py -v`
Expected: PASS（原 6 个适配 + `_fuse` 2 个 + 降级 3 个 = 11 个）

- [ ] **Step 6: 全量回归**

Run: `MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest -q`
Expected: 全部通过；唯一允许失败为 `test_config.py::test_defaults`（预存，与本改动无关）

- [ ] **Step 7: 提交**

```bash
git add app/agent/tools/search_articles.py app/agent/tools/__init__.py tests/test_tools.py
git commit -m "feat: search_articles 两路混合检索（BM25+knn+Python RRF）与降级"
```

---

### Task 6: 启动全量重建 lifespan + 手动重建脚本

**Files:**
- Modify: `app/main.py`（新增 lifespan，启动时全量重建 chunk 索引）
- Create: `scripts/rebuild_vectors.py`
- Test: `tests/test_lifespan.py`（新建）

**Interfaces:**
- Consumes: `sync_vectors`（Task 4，`app.main` 内 import 后由测试 monkeypatch）、`Settings.hybrid_search`
- Produces: `app.main` 的 `lifespan` 上下文管理器（FastAPI `lifespan=` 参数）；`scripts/rebuild_vectors.py` 独立入口

- [ ] **Step 1: 写失败测试**

创建 `tests/test_lifespan.py`：

```python
# -*- coding: utf-8 -*-
"""lifespan 测试：启动触发向量全量重建且异常不阻断服务（sync_vectors 全程 mock，不加载真实 BGE）。"""
from fastapi.testclient import TestClient


def test_lifespan_syncs_vectors_on_startup(monkeypatch):
    called = {}

    async def fake_sync(settings):
        called["yes"] = True
        return {"articles": 0, "chunks": 0, "updated": 0, "failed": 0}

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
Expected: FAIL（`TypeError: FastAPI() got an unexpected keyword argument 'lifespan'`）

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
    """启动时全量重建 chunk 索引；失败仅告警，不阻断服务（检索降级为纯 BM25 路）。"""
    if settings.hybrid_search:
        try:
            result = await sync_vectors(settings)
            logger.info("启动向量重建完成：%s", result)
        except Exception:
            logger.warning("启动向量重建失败，检索将仅走 BM25 路", exc_info=True)
    yield
```

改为：

```python
app = FastAPI(title="blog-agent", lifespan=lifespan)
```

创建 `scripts/rebuild_vectors.py`：

```python
# -*- coding: utf-8 -*-
"""手动全量重建文章向量：重建 article_chunks 索引（删旧建新 + 切块 + 嵌入写回）。

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
Expected: 全部通过（唯一允许失败仍是 `test_config.py::test_defaults` 预存项）

- [ ] **Step 6: 提交**

```bash
git add app/main.py scripts/rebuild_vectors.py tests/test_lifespan.py
git commit -m "feat: 启动向量全量重建（lifespan）+ 手动重建脚本"
```

---

## 手工验证（可选，实现完跑一遍真 ES）

1. 跑重建脚本：`/d/tool1/anancoda/envs/ai-agent/python.exe scripts/rebuild_vectors.py`（对着 `100.110.148.14:9200`，会删旧建新 `article_chunks` 索引）
2. 验证向量落库：查询 `http://100.110.148.14:9200/article_chunks/_search?size=3` 确认 `content_vector` 有值、长度 1024、`article_id`/`chunk_index` 字段正确
3. 启动服务 `uvicorn app.main:app`，问 "博客里有没有关于 SSE 的文章" 确认混合检索命中；把 `HYBRID_SEARCH=false` 重启再验证回退路径
