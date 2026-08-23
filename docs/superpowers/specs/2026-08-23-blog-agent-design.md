# 博客 AI Agent（Python 服务）设计文档

- 日期：2026-08-23
- 状态：已评审（用户逐节确认）
- 关联仓库：`G:\JavaWorkSpace\oy-blog`（Java 侧协议已存在，本设计**不改动 Java 代码**）

## 1. 目标与背景

用 LangChain 1.0+ / LangGraph 开发一个真实的博客 AI Agent 服务，替换当前协议桩
`oy-blog/scripts/agent_stub.py`，实现：

1. **纯对话助手**：多轮上下文、DeepSeek 深度思考模式
2. **博客内容问答（RAG）**：检索博客文章回答问题
3. **工具调用**：搜索文章、读文章全文、浏览文章列表
4. **写作辅助**：收集博客素材，产出大纲/草稿

### 关键约束

- **Java 端零改动**：`agent-service`（端口 8095）已实现会话管理、消息落库、SSE 转发，
  只依赖 Python 端实现两个协议端点（见 §4）
- **无状态**：会话历史由 Java 维护并随请求传入，Python 不落库、不用 LangGraph checkpointer
  （决策记录见 §12）
- **工具只读**：不自动发布文章（需要登录态，超出 v1 范围）
- **LLM：DeepSeek**（OpenAI 兼容接口），API key 通过环境变量提供
- **博客数据直连 ES**：`articles` 索引含全文 `content` 字段（IK 分词），搜索与读全文
  均直连 Elasticsearch，不经过 gateway / search-service / article-service
- **部署无关**：纯环境变量配置，裸跑或容器化均可（生产部署方式暂未定）
- 开发环境：conda 环境 `ai-agent`（Python 3.12.7，已装 fastapi/uvicorn/pydantic/httpx，
  需新增 langchain/langgraph/langchain-deepseek/pydantic-settings）

## 2. 总体架构

```
前端 ⇄ gateway(8080) ⇄ agent-service(Java:8095) ──SSE──> Python Agent (0.0.0.0:8001)
                                        │
                                        └──> MySQL(会话/消息)
Python Agent ──httpx+BasicAuth──> ES (articles 索引，含全文)
```

一个 FastAPI 进程，作为 stub 的**无缝替换**。模块划分：

| 模块 | 职责 | 依赖 |
|---|---|---|
| `app/main.py` | FastAPI 入口：`/chat/stream`、`/chat/stop` 两个端点 | sse_protocol, stream_registry, config |
| `app/sse_protocol.py` | SSE 帧序列化（与 Java StreamParser 逐字节对齐） | 无 |
| `app/stream_registry.py` | conversationId → asyncio.Task 登记表；stop=取消任务；同会话重复流=取消旧的再开新的 | 无 |
| `app/llm.py` | ChatDeepSeek 工厂 + deepThinking/model 映射 | config |
| `app/agent/graph.py` | LangGraph StateGraph：agent 节点 + tools 节点 + tools_condition 边 | llm, tools |
| `app/agent/prompts.py` | 系统提示词（人设/工具使用引导/写作辅助） | 无 |
| `app/agent/tools.py` | 3 个只读工具，httpx 直连 ES | config |
| `app/config.py` | pydantic-settings 读环境变量 | 无 |

## 3. LangGraph Agent 图设计

### State 与图结构

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # 系统提示 + 历史 + 用户消息
```

```
START → agent 节点（ChatDeepSeek + bind_tools(博客工具)）
              │  tools_condition
              ├── 有 tool_calls → tools 节点（ToolNode 执行）→ 回 agent 节点
              └── 无 → END
```

- `recursion_limit` ≈ 10，防工具循环死循环
- 不用 checkpointer（无状态）
- 流式：`graph.astream(input, stream_mode="messages")` 产出 `(chunk, metadata)` 元组

### DeepSeek thinking 处理

| deepThinking | 策略 | 说明 |
|---|---|---|
| false | `deepseek-chat` 普通模式 | 支持工具调用 |
| true（策略 A，默认） | `deepseek-chat` + `thinking` 参数开启（V3.1+ 混合思考） | **保留工具调用**，流式时 `reasoning_content` 独立成流 |
| true（策略 B，可配置） | `deepseek-reasoner` | 纯深度推理，**不支持工具调用**（思考模式下工具降级） |

配置项 `THINKING_MODE=hybrid|reasoner` 切换策略。

> ⚠️ 风险标注：`thinking` 参数的确切行为与 `reasoning_content` 的流式形态
> **必须用探测脚本对真实 DeepSeek API 验证**（实施计划的第一步任务）。
> 策略 A 验证失败则回退策略 B。langchain-deepseek 的 `ChatDeepSeek` 将
> `reasoning_content` 暴露在 `additional_kwargs["reasoning_content"]`。

### 流式事件映射

| 流内容 | 协议事件 |
|---|---|
| `AIMessageChunk.additional_kwargs["reasoning_content"]` | `thinking {content}` |
| `AIMessageChunk.content` | `token {content}` |
| `tool_call_chunks` / ToolMessage（工具执行过程） | 静默（协议只有 4 种事件） |
| 流正常结束 | `done {messageId: py-<uuid>}` |

### 取消机制

- `/chat/stream` 内 `asyncio.create_task(graph.astream(...))`，登记进 registry
- `/chat/stop` → `task.cancel()`；客户端断开（GeneratorExit）同样取消并清理
- 要求：取消后模型确实不再产生输出

## 4. Java ↔ Python 协议（不可变，来自 agent_stub.py 契约）

```
POST /chat/stream  {conversationId, userId, message, history:[{role,content}], deepThinking, model}
    -> text/event-stream，事件 token{content} / thinking{content} / done{messageId} / error{code,message}
POST /chat/stop    {conversationId} -> {"ok": true}
```

- 无鉴权，仅内网直连
- 帧格式：`event: <名>\ndata: <JSON>\n\n`（帧以 `\n\n` 分隔；行以 `event:`/`data:` 前缀识别，与 Java StreamParser 解析规则逐字节对齐）
- 手动 yield 帧（与 stub 相同），不用 sse-starlette 默认格式
- message 空 → `error {400}`；同会话已有活动流 → 取消旧的再开新的（Java 侧 409 是主守卫，Python 侧兜底）
- Java 忽略 Python 生成的 messageId，用自己生成的 id 落库

## 5. 工具设计（直连 ES）

三个只读工具，httpx.AsyncClient + BasicAuth（凭据留空则不启用认证），目标索引 `articles`：

1. `search_articles(keyword)` → ES `_search`：title/content/summary 多字段匹配 +
   `status=published` 过滤 + highlight 片段（去除 `<em>` 标签），返回 id/标题/命中片段/标签
2. `get_article_content(article_id)` → ES `_doc/{id}`：标题 + 全文（截断到
   `ARTICLE_CONTENT_MAX_CHARS`，默认 4000 字控上下文）
3. `list_articles(page, page_size)` → ES `_search` 按 createdAt 降序：id/标题/摘要/标签/日期

- 查询逻辑为 search-service 的 Python 侧最小重实现（status 过滤 + highlight），
  author/时间范围等 Agent 用不上的能力不迁移
- ✅ 已实测验证（2026-08-23）：`100.110.148.14:9200` 凭据连通，`articles` 索引 green、
  12 篇文章；`multi_match(title^3/content/summary) + term(status=published) + highlight`
  查询返回真实命中与 `<em>` 高亮片段。以实际 mapping 为准（注意：真实索引中 `slug` 为
  text+keyword 子字段，另有 `category` 字段，与 Java 实体略有差异）
- 已知现象：IK 对短关键词（如"微服务"）会拆词导致宽泛命中，按相关度排序即可，不影响 v1
- 工具执行失败 → ToolNode 把错误文本回给模型，模型兜底回答；**ES 挂掉不影响纯对话**

### 系统提示词要点

- 人设：oy 的个人博客 AI 助手；中文、友好、简洁
- 用户问博客内容时优先用工具检索，回答引用文章标题（链接格式留待后续配置）
- 写作辅助：引导先收集素材（搜索/读全文）再产出结构化大纲/草稿
- 思考模式的提示词注意事项按 DeepSeek 官方建议、以验证结果为准

## 6. 错误处理矩阵

| 场景 | 行为 |
|---|---|
| 参数不完整（message 空） | `error {400}` 后关闭流 |
| DeepSeek 限流/超时/服务错误 | `error {原始码, 友好文案}`，如 429 → "请求过于频繁，请稍后再试" |
| API key 缺失 | 启动时 fail-fast，日志明确提示 |
| ES 工具失败（挂了/超时） | 不中断对话：错误文本回给模型兜底；纯聊天不受影响 |
| 客户端断开 / stop | 取消任务，静默收尾（不发 done，Java 已处理收尾逻辑） |
| 生成中未知异常 | `error {500}` + 完整堆栈日志 |

## 7. 配置

pydantic-settings + `.env`（`.env.example` 提供模板）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | （必填） | 启动 fail-fast 校验 |
| `DEEPSEEK_BASE_URL` | 官方地址 | 可覆盖 |
| `AGENT_HOST` / `AGENT_PORT` | 0.0.0.0 / 8001 | 与 Java `agent.python.base-url` 默认值一致 |
| `ES_URL` | http://192.168.200.130:9200 | 可覆盖为 Tailscale IP（100.110.148.14:9200） |
| `ES_USERNAME` / `ES_PASSWORD` | 空 | 空 = 不启用认证。真实凭据（elastic）由用户提供，**只写进 `.env`（git 忽略），绝不进版本库**；`.env.example` 只放占位符 |
| `MODEL_DEFAULT` | deepseek-chat | `model` 字段映射基准 |
| `THINKING_MODE` | hybrid | hybrid=策略A / reasoner=策略B |
| `MOCK_LLM` | 0 | 1=联调假模式（假回复+假思考流，不花 API 额度） |
| `ARTICLE_CONTENT_MAX_CHARS` | 4000 | 全文截断长度 |
| `SEARCH_PAGE_SIZE` | 5 | 搜索默认返回条数 |

## 8. 项目结构

```
g:\agentWorkplace\BlogAgent\
├── app/
│   ├── main.py            # FastAPI 路由
│   ├── config.py          # pydantic-settings
│   ├── sse_protocol.py    # SSE 帧序列化
│   ├── stream_registry.py # 活动流登记/取消
│   ├── llm.py             # ChatDeepSeek 工厂
│   └── agent/
│       ├── graph.py       # StateGraph 构建
│       ├── prompts.py     # 系统提示词
│       └── tools.py       # ES 三工具
├── tests/
│   ├── test_protocol.py   # 帧格式逐字节断言
│   ├── test_tools.py      # httpx MockTransport 模拟 ES
│   └── test_stream.py     # Mock LLM 走完整流
├── .env.example          # 占位符模板
├── .gitignore            # 忽略 .env、.pytest_cache 等
├── README.md
└── requirements.txt
```

依赖：fastapi、uvicorn、langchain、langgraph、langchain-deepseek、httpx、
pydantic-settings（版本以 conda `ai-agent` 环境实测为准）。

## 9. 数据流时序

```
前端 → gateway(8080) → Java agent-service(8095)
  1. 校验参数 / 同会话 409 / 会话 upsert / 落库用户消息 / 取最近20条历史
  2. POST http://<agent-python>:8001/chat/stream
Python Agent
  3. 校验 → 取消同会话旧任务 → 组装 messages → create_task(graph.astream) 登记 registry
  4. StreamingResponse 逐帧 yield：
     agent节点流式 chunk → thinking / token 事件
     tool_call → ToolNode 直连 ES 执行 → 结果回填 → 继续生成（可多轮循环）
  5. 结束 → done{messageId}
Java 收 done → 落库 assistant 消息（含 thinking 内容与时长统计）→ 转发前端 → 流关闭

中断路径：前端停止 → Java dispose + POST /chat/stop → Python task.cancel() → 静默收尾
```

## 10. 测试策略（TDD）

1. **单测**
   - `test_protocol.py`：帧序列化逐字节断言（对照 Java StreamParser 解析规则）
   - `test_tools.py`：httpx MockTransport 模拟 ES 响应，断言查询体（status 过滤、
     highlight 配置）与 BasicAuth 头
   - `test_stream.py`：Mock LLM 跑完整流，断言事件序列、done/error 收尾、取消语义
2. **协议集成**：`MOCK_LLM=1` 启动服务，curl 验证事件序列与 stub 行为等价
   （正常流 / error 分支 / 中断 / stop 取消）
3. **真实联调**：真实 key + Java agent-service 全链路跑通一次（含前端 SSE 展示），
   验证清单写入 README

## 11. 验收标准

1. **协议兼容**：替换 stub 后 Java 端零改动、全链路可用
2. **四能力可演示**：
   - 纯对话：多轮上下文正确（Java 传历史）
   - RAG：问"博客里有没有关于微服务的文章"能命中真实文章并正确引用；
     问具体文章内容能给出准确内容
   - 工具调用：多轮工具调用正确，工具失败时模型兜底
   - 写作辅助：产出引用博客素材的结构化大纲/草稿
3. **deepThinking**：思考过程以 thinking 事件流式呈现，Java 端 thinkingTime 统计正确
4. **stop 生效**：中途停止后模型不再产生输出
5. **降级**：ES 挂掉时纯对话仍可用；MOCK_LLM=1 无 key 联调全通
6. **测试全绿** + 部署无关（纯 env 配置，裸跑/容器均可）

## 12. 决策记录：对话历史由 Java 管理 vs LangGraph checkpointer

**结论：v1 由 Java 管理历史（随请求传入），Python 无状态、不用 checkpointer。**

理由：
1. Checkpointer 无法消除 Java 的 MySQL 存储（前端会话/消息 UI 依赖它），引入只会产生
   第二份对话状态与同步问题
2. Checkpointer 的价值（中断恢复、人工审批、崩溃续跑、免重传历史）对单轮流式问答均不成立
3. 无状态带来的收益（随意重启、水平扩展、无持久卷依赖、并发/stop 语义简单）在部署方式
   未定时尤为重要
4. 将来若需要，LangGraph checkpointer 是 `compile(checkpointer=...)` 一个参数
   （thread_id=conversationId），增量引入成本低，无需重构图

重新考虑的触发条件：
- Agent 进化为长生命周期工作流（跨请求的后台任务、多轮人工审批流程）
- 需要 Java 不知情的派生记忆（对话摘要、用户偏好画像）—— 此时更可能是为 Python 增加
  按 conversationId 索引的记忆存储，而非迁移对话历史本身
