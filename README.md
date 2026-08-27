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

| 变量                          | 说明                                                                                     |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| `DEEPSEEK_API_KEY`          | DeepSeek API Key（必填，`MOCK_LLM=1` 时可留空）                                        |
| `DEEPSEEK_BASE_URL`         | DeepSeek API 地址，默认`https://api.deepseek.com`                                      |
| `AGENT_HOST`                | 服务监听地址，默认`0.0.0.0`                                                            |
| `AGENT_PORT`                | 服务监听端口，默认`8001`（与 Java `agent.python.base-url` 默认值一致）               |
| `ES_URL`                    | Elasticsearch 地址（博客文章索引`articles`），默认 `http://192.168.200.130:9200`     |
| `ES_USERNAME`               | ES 用户名（空 = 不启用认证）                                                             |
| `ES_PASSWORD`               | ES 密码（空 = 不启用认证）                                                               |
| `MODEL_DEFAULT`             | 默认模型，默认`deepseek-chat`                                                          |
| `THINKING_MODE`             | 思考模式：`hybrid`（deepseek-chat + thinking 参数）/ `reasoner`（deepseek-reasoner） |
| `MODEL_ALLOWLIST`           | 允许使用的模型列表：`deepseek-chat,deepseek-reasoner`                                  |
| `MOCK_LLM`                  | `1` = 联调假模式（固定假回复 + 模拟思考流，不花 API 额度），默认 `0`                 |
| `ARTICLE_CONTENT_MAX_CHARS` | 文章全文截断长度（控制上下文），默认`4000`                                             |
| `SEARCH_PAGE_SIZE`          | 搜索默认返回条数，默认`5`                                                              |

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
curl -s -N -X POST http://localhost:8001/chat/stream -H "Content-Type: application/json" -d "{\"conversationId\":\"itest-1\",\"userId\":\"u1\",\"message\":\"hello\",\"history\":[],\"deepThinking\":false,\"model\":\"default\"}"
```

## 5. 与 Java agent-service 联调

Java agent-service 通过配置 `agent.python.base-url`（默认即 `http://localhost:8001`，即本服务地址）指向本服务，协议与 `agent_stub.py` 完全一致，**Java 端无需任何改动**。

### 端点

| 端点                  | 请求体                                                                               | 响应                                                                        |
| --------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `POST /chat/stream` | `{conversationId, userId, message, history:[{role,content}], deepThinking, model}` | `text/event-stream`，事件 `token` / `thinking` / `done` / `error` |
| `POST /chat/stop`   | `{conversationId}`                                                                 | `{"ok": true, "conversationId": "..."}`                                   |

### 帧格式

帧格式：`event: <名>\ndata: <JSON>\n\n`（帧以 `\n\n` 分隔；行以 `event:` / `data:` 前缀识别，与 Java `PythonSseClient.StreamParser` 解析规则逐字节对齐，中文原样输出）。

| 事件         | data                                       | 说明                                                            |
| ------------ | ------------------------------------------ | --------------------------------------------------------------- |
| `token`    | `{"content": "..."}`                     | 输出内容片段，可多帧，按顺序拼接即全文                          |
| `thinking` | `{"content": "..."}`                     | 思考过程（`deepThinking=true` 时在首个 `token` 之前发出）   |
| `done`     | `{"messageId": "py-..."}`                | 流正常结束的终帧（Java 忽略此 messageId，用自己生成的 id 落库） |
| `error`    | `{"code": 400, "message": "参数不完整"}` | 出错，流随后关闭                                                |

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

当前全量 50 个测试通过（config / graph / llm / protocol / registry / stream / tools）。

## 7. 验收清单

- [ ] 1. **协议兼容**：替换 stub 后 Java 端零改动、全链路可用
  > 未验证：本机无 oy-blog Java 工程与前端环境，无法自动化联调。需用户手动启动
  > Java agent-service（8095）+ 前端，确认多轮历史、SSE 实时显示、thinking 折叠、
  > 会话/消息落库与 `/chat/stop` 按钮生效。
  >
- [X] 2. **四能力可演示**（2026-08-23 真实 DeepSeek + 直连 ES 验证）：
  - [X] 纯对话：单轮、多轮（`history` 传入）上下文均正确（追问"我叫什么名字"正确回答）
  - [X] RAG：问"博客里有没有关于微服务的文章"→ 模型调用 `search_articles` 命中真实文章并如实引用（索引 12 篇中无微服务专文，模型正确说明并列出真实标题）；问《SSE协议》→ 调用 `get_article_content` 后回答内容准确
  - [X] 工具调用：`search_articles` / `get_article_content` 真实调用（httpx 日志见 `articles/_search` 200）；工具失败时模型兜底（ES 不可达时告知检索暂不可用，不 500）
  - [X] 写作辅助：产出引用博客素材（《hashMap的底层原理》《倒排索引》）的结构化大纲
- [X] 3. **deepThinking**：思考过程以 thinking 事件流式呈现（171 帧真实推理内容，先于首个 token）
  > Java 端 thinkingTime 统计需 Java 环境确认（本机无 Java 工程，未验证）。
  >
- [X] 4. **stop 生效**：流进行中 `/chat/stop` → 生成立即静默结束、无 done 帧
- [X] 5. **降级**：ES 不可达时纯对话仍可用；问博客内容时模型兜底告知检索不可用（HTTP 200 不崩溃）；MOCK_LLM=1 无 key 联调全通
- [X] 6. **测试全绿**（全量 50 测试通过）+ 部署无关（纯 env 配置，本次以 `ES_URL` 环境变量覆盖直接生效）

> 验收记录（2026-08-23，真实 DeepSeek + 直连 ES 100.110.148.14:9200，全部通过）：
>
> - 逐 token 实证（规格 §3 核心）：`message="你好"` 产出 90+ 个连续小 token 帧（1-4 字/帧），非一次性整段
> - 工具调用实证：httpx 日志 `POST http://100.110.148.14:9200/articles/_search "HTTP/1.1 200 OK"`
> - stop 实证：流中途 `/chat/stop` → 立即静默收尾、无 done 帧
> - 降级实证：`ES_URL` 指向不可达地址重启后，纯对话正常、RAG 兜底回答（HTTP 200）
> - 注意：`.env` 的 `ES_URL` 若为本机不可达的旧默认值 `http://192.168.200.130:9200`，
>   需改为实际可直连地址（本机验证为 `http://100.110.148.14:9200`）RAG 才能命中文章。
>
> 回归验证记录（2026-08-27，完整回归：50 单测 + MOCK 协议联调 + 真实 DeepSeek/ES 全链路，全部通过）：
>
> - 单测：全量 **50 个通过**（config 6 / llm 9 / graph 5 / tools 6 / protocol 6 / registry 9 / stream 9），
>   含 registry.remove 身份检查修复（672d2aa）后的回归
> - MOCK 联调（8011 端口）：正常流 14 token→done、思考流 thinking 先发、空消息单 error 400、
>   多轮 history 正常处理、`/chat/stop` 静默取消无 done、同会话新流替换旧流（旧流无 done）
> - run.py 入口：以 `python run.py` 运行的实例正常服务（验证时 8001 即有该实例在跑）
> - 真实模式（ES_URL 覆盖为 `http://100.110.148.14:9200`）：
>   - 纯对话：37 token 帧正常收尾
>   - 多轮：history 传入后"我叫什么名字？"正确回答"小明"
>   - RAG：问"有没有关于微服务的文章"→ 多次 `search_articles`，如实报告无专文并引用真实标题
>     （《SSE协议》《FastAPI简介》《响应式编程》，已直连 ES 交叉验证均真实存在，索引 12 篇）
>   - 文章内容：问《SSE协议》→ `get_article_content` 后给出准确详细总结
>   - deepThinking：520 帧真实 thinking 全部先于 563 帧 token
>   - 写作辅助：产出引用博客素材的 Java 集合框架完整大纲
>   - 降级：`ES_URL` 指向不可达重启后纯对话正常、RAG 兜底如实告知检索不可用（HTTP 200 不崩溃）
> - 验收项 1（Java 端全链路）仍待用户在有 Java 工程的环境手动确认。
