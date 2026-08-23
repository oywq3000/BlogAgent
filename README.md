# BlogAgent — 博客 AI Agent 服务（Python）

## 1. 项目简介

博客 AI Agent 服务，基于 LangChain 1.0+ / LangGraph + FastAPI 实现，用于替换
oy-blog Java 侧原来的协议桩 `oy-blog/scripts/agent_stub.py`。

- **协议不变**：`/chat/stream`（SSE）+ `/chat/stop` 两个端点与 stub 完全一致，Java 端**零改动**。
- **核心能力**：纯对话（多轮历史）、博客内容问答（RAG）、工具调用（搜索/读全文/浏览列表）、写作辅助、DeepSeek 深度思考模式（thinking 流式呈现）。
- **降级友好**：Elasticsearch 挂掉不影响纯对话；`MOCK_LLM=1` 无 API key 即可全链路联调。

## 2. 环境准备

本机 conda 环境 `ai-agent` 的 Python 解释器路径（本机 `conda activate` 不可用，直接使用绝对路径）：

```bash
/d/tool1/anancoda/envs/ai-agent/python.exe
```

安装依赖：

```bash
cd /g/agentWorkplace/BlogAgent
/d/tool1/anancoda/envs/ai-agent/python.exe -m pip install -r requirements.txt
```

配置环境变量（模板 → 实际配置）：

```bash
cp .env.example .env
```

> `.env` 已被 git 忽略，只放本机配置（含真实凭据），**绝不提交**；`.env.example` 仅含占位符。

## 3. 环境变量表

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key（必填，`MOCK_LLM=1` 时可留空） |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址，默认 `https://api.deepseek.com` |
| `AGENT_HOST` | 服务监听地址，默认 `0.0.0.0` |
| `AGENT_PORT` | 服务监听端口，默认 `8001`（与 Java `agent.python.base-url` 默认值一致） |
| `ES_URL` | Elasticsearch 地址（博客文章索引 `articles`），默认 `http://192.168.200.130:9200` |
| `ES_USERNAME` | ES 用户名（空 = 不启用认证） |
| `ES_PASSWORD` | ES 密码（空 = 不启用认证） |
| `MODEL_DEFAULT` | 默认模型，默认 `deepseek-chat` |
| `THINKING_MODE` | 思考模式：`hybrid`（deepseek-chat + thinking 参数）/ `reasoner`（deepseek-reasoner） |
| `MODEL_ALLOWLIST` | 允许使用的模型列表：`deepseek-chat,deepseek-reasoner` |
| `MOCK_LLM` | `1` = 联调假模式（固定假回复 + 模拟思考流，不花 API 额度），默认 `0` |
| `ARTICLE_CONTENT_MAX_CHARS` | 文章全文截断长度（控制上下文），默认 `4000` |
| `SEARCH_PAGE_SIZE` | 搜索默认返回条数，默认 `5` |

## 4. 本地运行

```bash
cd /g/agentWorkplace/BlogAgent && MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

- 首次启动冷导入 langchain 较慢（本机实测约 20 秒），日志出现 `Uvicorn running on http://0.0.0.0:8001` 即就绪。
- **`MOCK_LLM=1` 联调模式**：不调用 DeepSeek、不花 API 额度，返回固定回复（内容为 `【MOCK】收到：<用户消息>`）与模拟思考流；`DEEPSEEK_API_KEY` 可留空。适合 Java 联调与协议验证。
- 非 mock 模式启动时若缺少 `DEEPSEEK_API_KEY`，服务启动即 fail-fast 报错。
- 无鉴权，仅内网直连。

快速自检（正常流）：

```bash
curl -s -N -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"conversationId":"itest-1","userId":"u1","message":"hello","history":[],"deepThinking":false,"model":"default"}'
```

## 5. 与 Java agent-service 联调

Java agent-service 通过配置 `agent.python.base-url`（默认即 `http://localhost:8001`，即本服务地址）指向本服务，协议与 `agent_stub.py` 完全一致，**Java 端无需任何改动**。

### 端点

| 端点 | 请求体 | 响应 |
|---|---|---|
| `POST /chat/stream` | `{conversationId, userId, message, history:[{role,content}], deepThinking, model}` | `text/event-stream`，事件 `token` / `thinking` / `done` / `error` |
| `POST /chat/stop` | `{conversationId}` | `{"ok": true, "conversationId": "..."}` |

### 帧格式

帧格式：`event: <名>\ndata: <JSON>\n\n`（帧以 `\n\n` 分隔；行以 `event:` / `data:` 前缀识别，与 Java `PythonSseClient.StreamParser` 解析规则逐字节对齐，中文原样输出）。

| 事件 | data | 说明 |
|---|---|---|
| `token` | `{"content": "..."}` | 输出内容片段，可多帧，按顺序拼接即全文 |
| `thinking` | `{"content": "..."}` | 思考过程（`deepThinking=true` 时在首个 `token` 之前发出） |
| `done` | `{"messageId": "py-..."}` | 流正常结束的终帧（Java 忽略此 messageId，用自己生成的 id 落库） |
| `error` | `{"code": 400, "message": "参数不完整"}` | 出错，流随后关闭 |

行为约定：

- `message` 为空 → 单个 `error {400, 参数不完整}` 后关闭流。
- 模型侧错误（限流/超时等）→ `error {原始码或 5xx, 友好文案}`（如 429 → "请求过于频繁，请稍后再试"）。
- 同一会话已有活动流时再开新流 → 取消旧流（Java 侧 409 为主守卫，Python 侧兜底）。
- `POST /chat/stop` 中途停止 → 当前流立即静默结束，**不发 `done`**。
- 客户端断开连接 → 服务端取消任务并清理，静默收尾。

## 6. 测试

```bash
cd /g/agentWorkplace/BlogAgent && MOCK_LLM=1 /d/tool1/anancoda/envs/ai-agent/python.exe -m pytest -v
```

当前全量 47 个测试通过（config / graph / llm / protocol / registry / stream / tools）。

## 7. 验收清单

- [ ] 1. **协议兼容**：替换 stub 后 Java 端零改动、全链路可用
- [ ] 2. **四能力可演示**：
  - [ ] 纯对话：多轮上下文正确（Java 传历史）
  - [ ] RAG：问"博客里有没有关于微服务的文章"能命中真实文章并正确引用；问具体文章内容能给出准确内容
  - [ ] 工具调用：多轮工具调用正确，工具失败时模型兜底
  - [ ] 写作辅助：产出引用博客素材的结构化大纲/草稿
- [ ] 3. **deepThinking**：思考过程以 thinking 事件流式呈现，Java 端 thinkingTime 统计正确
- [ ] 4. **stop 生效**：中途停止后模型不再产生输出
- [ ] 5. **降级**：ES 挂掉时纯对话仍可用；MOCK_LLM=1 无 key 联调全通
- [ ] 6. **测试全绿** + 部署无关（纯 env 配置，裸跑/容器均可）

> 验收清单由 Task 10（真实联调）逐条勾选并记录结果。
