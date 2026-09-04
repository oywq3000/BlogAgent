# search_articles 混合检索（BM25 + 向量 + RRF）设计文档

- 日期：2026-09-04
- 状态：已评审（用户逐项确认）
- 修订：v2（2026-09-04）——从"articles 同索引整篇向量"升级为**独立 `article_chunks` 索引 + 段落优先切块**（用户决策：长文语义不丢失、检索粒度到段落）
- 前置：`docs/superpowers/specs/2026-08-23-blog-agent-design.md`（本设计为其检索增强升级）

## 1. 目标与背景

把 `search_articles` 从纯关键词检索升级为**真正的向量 RAG**：BM25 关键词与 BGE 向量双路综合检索（hybrid search），结果在 Python 侧按 RRF 融合。

- **动机**：架构完备性——对齐行业标准 RAG 形态（向量检索 + 混合检索 + 切块），非召回痛点驱动
- **切块决策**：文章按"段落优先 + token 兜底 + 重叠"切 chunk，每 chunk 一个向量，存独立 `article_chunks` 索引。长文（超 bge 512 token 上限）后半段不再丢失，检索粒度细化到段落
- **范围**：只改 `search_articles` 一个工具 + 新增向量/切块基础设施；`get_article_content`、`list_articles`、LangGraph 图、Java 侧零改动

### 已验证的硬事实

| 事实 | 结果 |
|---|---|
| BGE 模型 | 本地 `F:/models/bge-large-zh-v1.5`（bge-large-zh-v1.5，**1024 维**，上下文 512 token） |
| GPU | ai-agent conda 环境 CUDA 可用（GTX 1650） |
| ES 版本 | 8.17.10（`100.110.148.14:9200`，BasicAuth）——原生支持 `knn` + `dense_vector` |
| 依赖 | 环境已装 sentence-transformers 5.7.0 / langchain-huggingface 1.2.2 |
| 量级 | articles 索引约 12 篇，切块后约几百 chunk，全量重建秒级 |

## 2. 总体架构与数据流

```
【启动时 / scripts/rebuild_vectors.py】（全量重建，量小可接受）
articles 全量读取
  → 每篇按段落优先切 chunk（token 兜底 + 重叠）
  → BGE（本地 GPU）逐 chunk 生成向量
  → 写入 article_chunks 索引（chunk 即文档，冗余 title/tags/createdAt）

【查询时】
search_articles(keyword)
  → embed_query(keyword) 生成查询向量（bge 指令前缀）
  → 请求1：BM25 查 articles 索引（原逻辑，含 highlight）
  → 请求2：knn 查 article_chunks 索引（top K 个 chunk）
  → chunk 按 article_id 归并回文章级
  → Python 侧 RRF 融合两路文章 id，取 top search_page_size
  → 组装 item（snippet：BM25 路用 highlight，knn-only 用 chunk 内容截取）
```

模块划分：

| 模块 | 职责 | 依赖 |
|---|---|---|
| `app/embedding.py`（新） | BGE 懒加载 + embed_documents / embed_query | sentence-transformers, config |
| `app/chunking.py`（新） | 段落优先 + token 兜底 + 重叠的切块纯函数 | 无（count_tokens 注入） |
| `app/vector_sync.py`（新） | ensure_index() + sync_vectors()（全量重建 chunk 索引） | embedding, chunking, httpx, config |
| `scripts/rebuild_vectors.py`（新） | 手动全量重建入口 | vector_sync |
| `app/agent/tools/search_articles.py`（改） | 两路检索 + Python RRF 融合 + 降级 | embedding, config |
| `app/main.py`（改） | lifespan 启动时全量重建 | vector_sync |
| `app/config.py`（改） | 新增配置项 | 无 |

## 3. ES 索引变更

### 3.1 新增 `article_chunks` 索引（本设计核心）

```json
PUT /article_chunks
{
  "mappings": {
    "properties": {
      "article_id": { "type": "keyword" },
      "chunk_index": { "type": "integer" },
      "status": { "type": "keyword" },
      "content": { "type": "text" },
      "title": { "type": "text" },
      "tags": { "type": "keyword" },
      "createdAt": { "type": "date" },
      "content_vector": {
        "type": "dense_vector",
        "dims": 1024,
        "index": true,
        "similarity": "l2_norm"
      }
    }
  }
}
```

- **文档 id**：`{article_id}-{chunk_index}`（如 `2088-0`）
- **chunk 即文档**：每个 chunk 一条，冗余 `title/tags/createdAt/status`（从文章复制）——knn-only 命中的文章无需回查 articles 即可组装完整 item，且 knn 的 `status: published` filter 有字段可匹配（**必须存在**，否则未映射字段的 term 查询静默返回空，向量路恒 0 命中）
- **同步只建已发布文章**：读 articles 时 query 用 `term: {status: published}`，草稿不建 chunk
- `l2_norm`：BGE 官方推荐欧氏距离，不用 cosine
- **articles 索引零改动**（Java 侧零改动），新索引由 Python 侧建（PUT 幂等：已存在则跳过）
- **全量重建策略**：chunk 是文章派生物，无法简单"缺失补齐"（文章更新后旧 chunk 需作废）——量小（12 篇/几百 chunk）直接删索引重建，启动即自愈

## 4. 新组件设计

### 4.1 `app/embedding.py`

```python
def get_embedder() -> SentenceTransformer   # 懒加载 + 缓存；device 按 EMBEDDING_DEVICE（cuda 失败自动回退 cpu）
def embed_query(text: str) -> list[float]    # 带 bge query 指令前缀
def embed_documents(texts: list[str]) -> list[list[float]]   # 不带前缀；空文本兜底为单空格
```

- **懒加载**：首次调用才加载模型，避免拖慢服务启动
- **bge 指令前缀**（关键）：查询侧加 `"为这个句子生成表示以用于检索相关文章："`，文档（passage）侧不加
- 超长文本由 encode 按模型 max_seq_length（512 token）截断；chunk 尺寸远小于该上限，正常不触发

### 4.2 `app/chunking.py`（纯函数，不依赖模型）

```python
def split_content(content: str, count_tokens: Callable[[str], int],
                  max_tokens: int = 256, overlap_tokens: int = 32) -> list[str]
```

**切块规则（段落优先 + token 兜底 + 重叠）**：

1. **Markdown 分段**：```` ``` ```` 围栏代码块视为独立段；其余按空行分隔为段落
2. **段累积**：相邻短段落合并累积，达到 `max_tokens` 即成一个 chunk——**段落级 chunk 之间不重叠**（段落是完整语义单元，边界完整，不切坏语义）
3. **token 兜底**：单个段超过 `max_tokens`（如超长无空行文字）→ 该段按 token 窗口切分，窗口 `max_tokens`、步长 `max_tokens - overlap_tokens`——**窗口间重叠 `overlap_tokens`**，避免边界上下文丢失
4. 输出 chunk 文本列表（每项非空）

### 4.3 `app/vector_sync.py`

- `ensure_index(client, settings)`：PUT 建 `article_chunks` 索引（幂等：`resource_already_exists_exception` 视为成功）
- `sync_vectors(settings, client=None, embed_documents=None, count_tokens=None) -> dict`：**全量重建**——删索引重建 → 分页读 articles（`_source: [id, title, tags, createdAt, content]`）→ 每篇 `split_content` → `embed_documents` → 逐 chunk `PUT /article_chunks/_doc/{article_id}-{i}`。返回 `{"articles", "chunks", "updated", "failed"}`
- `rebuild_vectors(...) -> dict`：等价全量重建，供手动脚本调用（与 sync 同实现）
- `embed_documents` / `count_tokens` 均可注入（`None` 时默认：`app.embedding.embed_documents`、`get_embedder().tokenizer` 计数）——测试注入 fake，不加载真实模型
- 单条写入失败：记日志继续，不中断

## 5. search_articles 改造（两路检索 + Python RRF）

```
请求1（BM25 路，查 articles，原 body 不变）：
  { "query": { bool: { must: { multi_match: {...} }, filter: {term: {status: published}} } },
    "highlight": {...}, "size": page_size, "_source": [id,title,tags,createdAt] }

请求2（向量路，查 article_chunks，仅 hybrid 开启且 embed_query 成功时）：
  { "knn": { "field": "content_vector", "query_vector": <vec>,
             "k": <knn_k=search_page_size*5>, "filter": {"term": {"status": "published"}} },
    "_source": ["article_id", "content", "title", "tags", "createdAt", "chunk_index"] }
```

**Python 侧融合（RRF）**：

1. BM25 路命中按序记 rank 0..n-1；向量路 chunk 命中**按 article_id 归并**，每篇文章取**首次出现的最小 rank**，两路各对文章累加 `score = Σ 1/(60 + rank + 1)`
2. 按 score 降序取 `search_page_size` 篇
3. 组装 item：`id/title/tags/createdAt` 两路都有（chunk 索引冗余了元数据）；`snippet` 优先用 BM25 路 highlight，**knn-only 命中**的文章用其首个 chunk 的 `content` 前 150 字截取
4. 输出 JSON 数组结构不变：`[{id, title, tags, createdAt, snippet?}]`

**为什么不走 ES 原生 `rank.rrf`**：RRF 要求 knn 与 query 在同一 `_search` 请求内、同一索引上；本设计两路分属不同索引（articles / article_chunks），无法单请求融合，故 Python 侧实现 RRF（公式相同，效果等价）。

**可注入性**：`build_search_articles(settings, client, embed_query=None)`，`embed_query` 为 `None` 时导入 `app.embedding.embed_query`；`build_tools` 透传。

## 6. 错误处理与降级链（延续规格 §6 哲学）

| 故障 | 行为 |
|---|---|
| embedding 模型加载失败 / GPU 不可用 | 日志警告；跳过请求2，仅 BM25 路 |
| `embed_query` 抛异常 | 同上（回退单路） |
| 请求2（knn）返回 400（索引未建/字段缺失） | 日志警告；忽略向量路，仅用 BM25 路结果 |
| 启动全量重建失败（ES 不可达） | 日志警告；**不阻断服务启动**；检索退化为纯 BM25 |
| 请求1（BM25）失败 | 现有错误文本回模型兜底（不变） |

- 两路独立失败互不影响：任何一路失败，另一路照常返回，服务不崩
- knn 请求的 `filter` 与 BM25 路 `bool.filter` 一致（`status: published`），两路候选同域

## 7. 配置项（app/config.py 新增）

| 项 | 默认 | 说明 |
|---|---|---|
| `EMBEDDING_MODEL_PATH` | `F:/models/bge-large-zh-v1.5` | BGE 模型目录 |
| `EMBEDDING_DEVICE` | `cuda` | 加载失败自动 fallback cpu |
| `HYBRID_SEARCH` | `true` | 混合检索总开关（false 时完全回退现状：仅 BM25 一路，不建/不查 chunk 索引） |
| `CHUNK_MAX_TOKENS` | `256` | chunk 目标 token 数 |
| `CHUNK_OVERLAP_TOKENS` | `32` | 超长段窗口重叠 token 数 |

## 8. 测试策略

- **chunking.py**：段落合并、代码块独立、超长段窗口重叠（含 overlap=0 边界）、空文本、count_tokens 注入（用 `len` 或 fake）
- **vector_sync.py**：MockTransport 断言索引幂等建、删旧建新、每篇切块后逐 chunk 写入（文档 id / 冗余字段 / 向量）、单条失败继续
- **search_articles**：两路请求形状（BM25 body 原样 / knn body 含 filter）；归并 + RRF 排序正确性；snippet 两种来源；knn-only 命中组装完整 item；降级路径（embed 失败、knn 400、`HYBRID_SEARCH=false`）各自只走 BM25 路；`build_tools` 传 `embed_query` fake 不加载真实模型
- **embedding.py**：指令前缀、懒加载单例、设备回退、空文本兜底（mock SentenceTransformer）
- **lifespan**：启动触发 sync_vectors、异常不阻断（monkeypatch）
- 全量回归：现有测试适配 + 新增全通过；唯一允许失败仍为 `test_config.py::test_defaults`（预存）

## 9. 决策记录

1. **Python 侧 RRF 而非 ES 原生 `rank.rrf`**：两路分属不同索引无法单请求融合；Python 实现同一公式（`1/(k+rank)`，k=60），效果等价且可单测
2. **独立 `article_chunks` 索引而非 articles 加字段**：knn 要求"一文档一向量"，一篇多 chunk 无法塞进单文档；独立索引保持 Java 零改动，chunk 冗余元数据避免回查
3. **段落优先 + token 兜底 + 重叠**：固定 token 窗口会切在语义中间（代码块/句子被切开）；段落是博客天然语义单元，段落级不重叠保语义完整，仅超长段用重叠窗口防边界丢失
4. **全量重建而非增量**：chunk 是文章派生物，文章更新后旧 chunk 必须作废，增量无法判定；量小（12 篇）删索引重建秒级完成
5. **l2_norm 而非 cosine**：BGE 官方推荐欧氏距离
6. **query 指令前缀仅查询侧**：bge-large-zh 官方要求查询加指令、文档不加
7. **不做 rerank**（YAGNI）：文章量级小，RRF 已足够；rerank 需额外模型与管线
