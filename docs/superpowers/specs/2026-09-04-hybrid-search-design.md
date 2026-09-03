# search_articles 混合检索（BM25 + 向量 + RRF）设计文档

- 日期：2026-09-04
- 状态：已评审（用户逐项确认）
- 前置：`docs/superpowers/specs/2026-08-23-blog-agent-design.md`（本设计为其检索增强升级）

## 1. 目标与背景

把 `search_articles` 从纯关键词检索升级为**真正的向量 RAG**：与现有索引检索（IK 分词 BM25）做两路综合检索（hybrid search），ES 原生 RRF 融合。

- **动机**：架构完备性——对齐行业标准的 RAG 形态（向量检索 + 混合检索），非召回痛点驱动
- **范围**：只改 `search_articles` 一个工具 + 新增向量基础设施；`get_article_content`、`list_articles`、LangGraph 图、Java 侧零改动

### 已验证的硬事实

| 事实 | 结果 |
|---|---|
| BGE 模型 | 本地 `F:/models/bge-large-zh-v1.5`（bge-large-zh-v1.5，**1024 维**） |
| GPU | ai-agent conda 环境 CUDA 可用（GTX 1650） |
| ES 版本 | 8.17.10（`100.110.148.14:9200`，BasicAuth）——原生支持 `knn` + `rank: rrf` |
| 依赖 | 环境已装 sentence-transformers 5.7.0 / langchain-huggingface 1.2.2 |
| 量级 | articles 索引约 12 篇，向量同步成本可忽略 |

## 2. 总体架构与数据流

```
【启动时 / scripts/rebuild_vectors.py】
ES articles 全量扫描
  → 缺 content_vector 的文章 → BGE（本地 GPU）生成向量
  → ensure_mapping() 补字段（首次）+ update 写回

【查询时】
search_articles(keyword)
  → embed_query(keyword) 生成查询向量（bge 指令前缀）
  → 一个 _search 请求：knn(content_vector) + 原 BM25 query + rank: {rrf: {}}
  → 融合后 hits 结构与现状完全兼容，解析逻辑不变
```

模块划分：

| 模块 | 职责 | 依赖 |
|---|---|---|
| `app/embedding.py`（新） | BGE 懒加载 + embed_documents / embed_query | sentence-transformers, config |
| `app/vector_sync.py`（新） | ensure_mapping() + sync_vectors()（增量）+ rebuild_vectors()（全量） | embedding, httpx, config |
| `scripts/rebuild_vectors.py`（新） | 手动全量重建入口 | vector_sync |
| `app/agent/tools/search_articles.py`（改） | 混合检索 body + 降级回退 | embedding, config |
| `app/main.py`（改） | lifespan 启动时触发增量同步 | vector_sync |
| `app/config.py`（改） | 新增配置项 | 无 |

## 3. ES 索引变更（一次性迁移）

```json
PUT /articles/_mapping
{
  "properties": {
    "content_vector": {
      "type": "dense_vector",
      "dims": 1024,
      "index": true,
      "similarity": "l2_norm"
    }
  }
}
```

- 只**新增**字段，不动现有字段，Java 侧（oy-blog）**零改动**
- `similarity: l2_norm`：BGE 官方明确推荐欧氏距离，不推荐 cosine
- 迁移由 `vector_sync.ensure_mapping()` 首次运行自动执行（幂等）

## 4. 新组件设计

### 4.1 `app/embedding.py`

```python
def get_embedder() -> SentenceTransformer   # 懒加载 + 缓存；device 按 EMBEDDING_DEVICE（cuda 失败自动 fallback cpu）
def embed_query(text: str) -> list[float]    # 带 bge query 指令前缀
def embed_documents(texts: list[str]) -> list[list[float]]   # 不带前缀
```

- **懒加载**：首次调用才加载模型，避免拖慢服务启动
- **bge 指令前缀**（关键）：bge-large-zh-v1.5 查询需加 `"为这个句子生成表示以用于检索相关文章:"`，文档（passage）不加；同步生成文档向量时不含前缀，查询向量含前缀
- 文本归一化：embed_documents 前对空/超长文本做截断保护（模型 max_seq_length 512，按 tokenizer 截断）

### 4.2 `app/vector_sync.py`

- `ensure_mapping(client)`：PUT mapping 补 `content_vector` 字段（幂等，已存在则跳过）
- `sync_vectors()`：分页读取 articles（from/size，`_source: [id, content]`；当前量级一次取全即可，不引入 scroll 复杂度），筛出缺 `content_vector` 的文档 → `embed_documents` → 逐条 `POST /articles/_update/{id}` 写回（doc 部分更新，不动其他字段）。**只补缺失（增量自愈）**
- `rebuild_vectors()`：先 `POST /articles/_update_by_query` 清空 `content_vector`（script），再全量 `sync_vectors()`
- 失败处理：单条失败记录日志继续；ES 不可达 → 抛异常由调用方决定（启动侧降级，见 §6）

### 4.3 `scripts/rebuild_vectors.py`

独立可运行脚本：加载 settings + client → `rebuild_vectors()` → 打印重建统计（总数/成功/失败）。

## 5. search_articles 改造

原 BM25 body 升级为混合检索（`HYBRID_SEARCH=true` 时）：

```json
{
  "knn": {
    "field": "content_vector",
    "query_vector": "<embed_query(keyword)>",
    "k": <search_page_size>,
    "filter": { "term": { "status": "published" } }
  },
  "query": { "bool": { "must": { "multi_match": {...原样 } }, "filter": {"term": {"status": "published"}} } },
  "rank": { "rrf": {} },
  "highlight": { ...原样 },
  "size": <search_page_size>,
  "_source": [...原样]
}
```

- **knn 必须与 query 同过滤条件**：knn 的 `filter` 与 query 的 bool.filter 一致（`status: published`），保证两路候选同域
- `rank.rrf`：ES 原生倒数排名融合，k=60 默认值
- highlight 保留：RRF 融合结果中 BM25 路命中仍有 highlight，向量路无（snippet 逻辑 `hl.get("content") or hl.get("title")` 天然兼容）
- `HYBRID_SEARCH=false` 时 body 完全回退现状（纯 BM25），代码路径保留
- **可注入性**：`build_search_articles` 增加可选参数 `embed_query: Callable[[str], list[float]] | None = None`（默认取 `app.embedding.embed_query`），测试注入 fake，避免加载真实模型；`build_tools` 透传该参数

## 6. 错误处理与降级链（延续规格 §6 哲学）

| 故障 | 行为 |
|---|---|
| embedding 模型加载失败 / GPU 不可用 | 日志警告；`search_articles` 回退纯 BM25 body |
| `content_vector` 字段不存在（mapping 未建成） | knn 查询会返回 400（`search_phase_execution_exception`）；捕获后回退纯 BM25 body，服务照常 |
| 启动同步失败（ES 写不可达） | 日志警告；**不阻断服务启动**；检索回退纯 BM25 |
| ES 查询失败 | 现有错误文本回模型兜底（不变） |

- 混合检索对 ES 是**无破坏性**的：字段已存在但部分文档无向量时，knn 只对已有向量的文档检索，RRF 融合仍返回 BM25 路结果。因此即使同步从未成功，服务也可用
- 降级判定放在请求层：构造混合 body 前先尝试 `embed_query`，失败即回退纯 BM25；若 ES 返回 400（字段不存在）也回退纯 BM25，错误文本模式与现有 `httpx.HTTPError` 分支一致

## 7. 配置项（app/config.py 新增）

| 项 | 默认 | 说明 |
|---|---|---|
| `EMBEDDING_MODEL_PATH` | `F:/models/bge-large-zh-v1.5` | BGE 模型目录 |
| `EMBEDDING_DEVICE` | `cuda` | 加载失败自动 fallback cpu |
| `HYBRID_SEARCH` | `true` | 混合检索总开关 |

## 8. 测试策略

- **search_articles**（test_tools.py 适配）：MockTransport 断言 body 含 `knn.field` / `rank.rrf` / 双路 `status: published` 过滤一致；embedding 通过 `embed_query` 注入 mock（不加载真实模型）；`HYBRID_SEARCH=false` 断言回退纯 BM25 body；embedding 抛异常 → 断言回退 body
- **embedding.py**：`embed_query` 带指令前缀、`embed_documents` 不带；长文本截断；懒加载只实例化一次（mock SentenceTransformer）
- **vector_sync.py**：MockTransport 断言 mapping 请求幂等、增量只更新缺失项、rebuild 先清空后全量；单条失败继续
- 全量回归：67 现有测试 + 新增测试全通过（1 个预存失败 `test_config.py::test_defaults` 与本改动无关）

## 9. 决策记录

1. **RRF 而非线性加权**：ES 8.17 原生 `rank.rrf`，一个请求完成融合，无分数归一化调参负担；线性加权需手调权重且跨检索器分数不可比
2. **同索引加字段而非独立向量索引**：一个请求天然融合、数据一致性由文档生命周期保证；只加字段不动现有字段，Java 零改动
3. **l2_norm 而非 cosine**：BGE 官方推荐欧氏距离（bge 的训练与评估基于 L2）
4. **不做 chunking / rerank**（YAGNI）：全文仅 12 篇且 content 直接入库，向量对整个 content 生成即可；rerank 需要额外模型与管线，收益低于成本
5. **启动增量同步 + 手动重建脚本**：12 篇秒级，无定时组件负担；增量只补缺失避免重复写
6. **query 指令前缀仅查询侧**：bge-large-zh 是对称/非对称混合模型，官方要求查询侧加指令、文档侧不加——这是检索效果的关键细节，同步时生成的文档向量必须无前缀
