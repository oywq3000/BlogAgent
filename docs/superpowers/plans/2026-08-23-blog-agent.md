# 博客 AI Agent（Python + LangGraph + DeepSeek）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangChain 1.0+ / LangGraph 实现博客 AI Agent 服务，无缝替换 `oy-blog/scripts/agent_stub.py`，提供纯对话、博客 RAG（直连 ES）、工具调用与写作辅助。

**Architecture:** FastAPI 单进程监听 8001，实现 Java 端已定义的 SSE 协议（`/chat/stream`、`/chat/stop`）。每请求构建 LangGraph 图（agent 节点 + ToolNode + tools_condition），`astream(stream_mode="messages")` 流式产出，`reasoning_content`→thinking 事件、`content`→token 事件。三个工具直连 ES `articles` 索引。无状态：历史由 Java 传入。

**Tech Stack:** Python 3.12（conda `ai-agent` 环境）、FastAPI、uvicorn、LangChain ≥1.0、LangGraph ≥1.0、langchain-deepseek、httpx、pydantic-settings、pytest + pytest-asyncio。

**Spec:** `docs/superpowers/specs/2026-08-23-blog-agent-design.md`（实施前必读，计划从规格推导）

## Global Constraints

- **Git 提交署名**：所有 commit 以 `oywq3000 <2603321762@qq.com>` 身份提交（本仓库已配置），**禁止**任何 AI 助手署名或 `Co-Authored-By` 尾注
- **Java 端零改动**：规格 §4 的 Java↔Python 协议不可变（事件名、字段、帧格式逐字节对齐）
- **密钥纪律**：真实密钥（DEEPSEEK_API_KEY、ES 密码）只进 `.env`（已 gitignore），绝不写入代码、测试、文档或提交信息
- **运行环境**：所有 python/pip/pytest/uvicorn 命令在 conda `ai-agent` 环境执行：`source /d/tool1/anancoda/etc/profile.d/conda.sh && conda activate ai-agent`
- **版本下限**：langchain ≥1.0、langgraph ≥1.0（安装后把实际版本锁定进 requirements.txt）
- **无状态**：不落库、不用 checkpointer（规格 §12 决策记录）
- **TDD**：每个任务的顺序是"先写失败测试 → 跑失败 → 最小实现 → 跑绿 → 提交"
- 代码注释与提交信息用中文（与 oy-blog 仓库风格一致）

---

### Task 1: 环境依赖 + DeepSeek thinking 探测（风险验证）

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.env`（用户填真实值，gitignore 已覆盖）
- Create: `scripts/probe_deepseek.py`
- Create: `scripts/probe_result.md`（探测结论记录）

**Interfaces:**
- Consumes: 无（首个任务）
- Produces: `requirements.txt`（后续任务安装依赖的基础）、`scripts/probe_result.md` 记录 ChatDeepSeek 的 thinking 参数用法结论，Task 6 的 `build_chat_model` 实现以它为准

**背景**：规格 §3 风险标注——`deepseek-chat` 的 `thinking` 参数（V3.1+ 混合思考）与 `reasoning_content` 流式形态必须先对真实 API 验证，验证失败则回退 `deepseek-reasoner` 策略。本任务把这条路走通并记录结论。

- [ ] **Step 1: 创建 requirements.txt**

```text
fastapi>=0.115
uvicorn>=0.30
httpx>=0.27
pydantic>=2.7
pydantic-settings>=2.3
langchain>=1.0
langgraph>=1.0
langchain-deepseek>=1.0
pytest>=8.0
pytest-asyncio>=0.23
```

- [ ] **Step 2: 安装依赖并锁定版本**

```bash
source /d/tool1/anancoda/etc/profile.d/conda.sh && conda activate ai-agent && cd /g/agentWorkplace/BlogAgent && pip install -r requirements.txt
```

安装成功后把实际版本锁定进 requirements.txt（替换 `>=` 为 `==`，仅 langchain / langgraph / langchain-deepseek / langchain-openai 相关行）：

```bash
pip freeze | grep -iE "^(langchain|langgraph|langchain-deepseek|langchain-openai|langchain-core)="
```

将输出版本逐行改写进 requirements.txt（如 `langchain==1.2.3`）。

- [ ] **Step 3: 创建 .env.example（占位符模板）**

```ini
# DeepSeek API（必填，MOCK_LLM=1 时可留空）
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 服务监听
AGENT_HOST=0.0.0.0
AGENT_PORT=8001

# Elasticsearch（博客文章索引 articles）
ES_URL=http://192.168.200.130:9200
ES_USERNAME=
ES_PASSWORD=

# 模型与思考模式
MODEL_DEFAULT=deepseek-chat
THINKING_MODE=hybrid
MODEL_ALLOWLIST=deepseek-chat,deepseek-reasoner

# 联调假模式（不花 API 额度）
MOCK_LLM=0

# 工具参数
ARTICLE_CONTENT_MAX_CHARS=4000
SEARCH_PAGE_SIZE=5
```

- [ ] **Step 4: 创建 .env 并填入真实值**

复制 .env.example 为 .env。`DEEPSEEK_API_KEY` 向用户索取；`ES_URL` 用户开发机可直连 `http://100.110.148.14:9200`（Tailscale），`ES_USERNAME=elastic`，`ES_PASSWORD` 向用户索取（用户已提供过，勿写入任何提交）。

- [ ] **Step 5: 编写探测脚本 scripts/probe_deepseek.py**

```python
# -*- coding: utf-8 -*-
"""DeepSeek thinking 参数与 reasoning_content 流式探测（一次性验证脚本）。

验证三件事（结论手工记录到 scripts/probe_result.md）：
  1. deepseek-chat + extra_body={"thinking": {"type": "enabled"}} 是否被接受
  2. 流式 chunk 的 additional_kwargs["reasoning_content"] 是否有思考内容
  3. 开启 thinking 后 tool_calls 是否仍可用
用法: conda activate ai-agent && python scripts/probe_deepseek.py
"""
import asyncio
import os

from dotenv import load_dotenv  # 若无则 pip install python-dotenv

from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

load_dotenv()


def dump(tag: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {detail}", flush=True)


async def probe_streaming() -> None:
    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        streaming=True,
        extra_body={"thinking": {"type": "enabled"}},
    )
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    async for chunk in model.astream([HumanMessage(content="1+1=? 只回答数字")]):
        reasoning_parts.append(chunk.additional_kwargs.get("reasoning_content") or "")
        content_parts.append(chunk.content if isinstance(chunk.content, str) else "")
    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)
    dump("thinking 参数被接受", True, "请求未报错")
    dump("reasoning_content 流式输出", bool(reasoning), f"思考内容 {len(reasoning)} 字")
    dump("content 正常输出", bool(content), f"回答: {content!r}")


async def probe_tool_calling() -> None:
    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        extra_body={"thinking": {"type": "enabled"}},
    )
    llm_with_tools = model.bind_tools([
        {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ])
    resp = await llm_with_tools.ainvoke([HumanMessage(content="北京天气怎么样？请调用工具查询")])
    tool_calls = resp.tool_calls or []
    dump("thinking 模式下工具调用", len(tool_calls) > 0, f"tool_calls 数量: {len(tool_calls)}")


async def main() -> None:
    print(f"DeepSeek base_url={os.environ.get('DEEPSEEK_BASE_URL', '默认')}", flush=True)
    await probe_streaming()
    await probe_tool_calling()


if __name__ == "__main__":
    asyncio.run(main())
```

注意：若 `from dotenv import load_dotenv` 导入失败，先 `pip install python-dotenv` 并加入 requirements.txt。

- [ ] **Step 6: 运行探测并记录结论**

```bash
source /d/tool1/anancoda/etc/profile.d/conda.sh && conda activate ai-agent && cd /g/agentWorkplace/BlogAgent && python scripts/probe_deepseek.py
```

把三项探测结论与关键参数形态写入 `scripts/probe_result.md`（格式：结论/证据/对 Task 6 的影响）。**判定标准**：三项全 PASS → 策略 A（hybrid）可行；"工具调用"FAIL → 规格 §3 策略 B（`THINKING_MODE=reasoner` 为默认回退）；"thinking 参数被接受"FAIL → 停止执行并向用户报告，重新讨论规格 §3。

- [ ] **Step 7: 提交**

```bash
cd /g/agentWorkplace/BlogAgent && git add requirements.txt .env.example scripts/ && git commit -m "chore: 依赖清单与 DeepSeek thinking 探测脚本"
```

（`.env` 已被 gitignore，`git add` 不会纳入。若担心误加，提交前 `git status` 确认无 .env。）

---

### Task 2: 项目骨架 + 配置模块（pydantic-settings）

**Files:**
- Create: `pytest.ini`
- Create: `app/__init__.py`（空文件）
- Create: `app/config.py`
- Test: `tests/__init__.py`（空文件）
- Test: `tests/test_config.py`
- Delete: `main.py`（仓库根目录的空文件，应用入口在 `app/main.py`，Task 8 创建）

**Interfaces:**
- Consumes: 无
- Produces: `Settings`（字段见下）、`get_settings() -> Settings`（lru_cache）、`Settings.require_api_key()`。Task 5/6/8 依赖这些名字。

- [ ] **Step 1: 写失败测试 tests/test_config.py**

```python
import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_defaults():
    s = Settings(_env_file=None)  # 不读 .env，测纯默认值
    assert s.agent_port == 8001
    assert s.model_default == "deepseek-chat"
    assert s.thinking_mode == "hybrid"
    assert s.article_content_max_chars == 4000
    assert s.search_page_size == 5
    assert s.mock_llm is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("ES_URL", "http://es.test:9200")
    monkeypatch.setenv("THINKING_MODE", "reasoner")
    monkeypatch.setenv("MOCK_LLM", "1")
    s = Settings(_env_file=None)
    assert s.es_url == "http://es.test:9200"
    assert s.thinking_mode == "reasoner"
    assert s.mock_llm is True


def test_invalid_thinking_mode_rejected():
    with pytest.raises(ValidationError):
        Settings(thinking_mode="magic", _env_file=None)


def test_require_api_key_mock_mode_ok():
    s = Settings(mock_llm=True, deepseek_api_key=None, _env_file=None)
    s.require_api_key()  # 不抛异常


def test_require_api_key_missing_raises():
    s = Settings(mock_llm=False, deepseek_api_key=None, _env_file=None)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        s.require_api_key()


def test_get_settings_cached(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("AGENT_PORT", "9999")
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    assert s1.agent_port == 9999
    get_settings.cache_clear()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
source /d/tool1/anancoda/etc/profile.d/conda.sh && conda activate ai-agent && cd /g/agentWorkplace/BlogAgent && python -m pytest tests/test_config.py -v
```

Expected: FAIL（ModuleNotFoundError: app.config）

- [ ] **Step 3: 创建 pytest.ini**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 4: 实现 app/config.py**

```python
# -*- coding: utf-8 -*-
"""应用配置：pydantic-settings 读环境变量 / .env。"""
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"

    agent_host: str = "0.0.0.0"
    agent_port: int = 8001

    es_url: str = "http://192.168.200.130:9200"
    es_username: str = ""
    es_password: str = ""

    model_default: str = "deepseek-chat"
    model_allowlist: list[str] = ["deepseek-chat", "deepseek-reasoner"]
    thinking_mode: str = "hybrid"  # hybrid=deepseek-chat+thinking 参数 / reasoner=deepseek-reasoner

    mock_llm: bool = False

    article_content_max_chars: int = 4000
    search_page_size: int = 5

    @field_validator("thinking_mode")
    @classmethod
    def _check_thinking_mode(cls, v: str) -> str:
        if v not in ("hybrid", "reasoner"):
            raise ValueError("THINKING_MODE 必须是 hybrid 或 reasoner")
        return v

    def require_api_key(self) -> None:
        """启动 fail-fast：非 mock 模式下缺 key 直接报错。"""
        if not self.mock_llm and not self.deepseek_api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY（.env），或设置 MOCK_LLM=1 使用联调假模式")


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: 删除根目录空文件 main.py**

```bash
cd /g/agentWorkplace/BlogAgent && git rm main.py
```

- [ ] **Step 6: 跑测试确认通过**

```bash
python -m pytest tests/test_config.py -v
```

Expected: 6 passed

- [ ] **Step 7: 提交**

```bash
git add pytest.ini app/ tests/ && git commit -m "feat: 项目骨架与配置模块（pydantic-settings）"
```

---

### Task 3: SSE 协议帧序列化

**Files:**
- Create: `app/sse_protocol.py`
- Test: `tests/test_protocol.py`

**Interfaces:**
- Consumes: 无
- Produces: `sse_event(name: str, data: dict) -> str`、`sse_error(code: int, message: str) -> str`。Task 8 用它生成所有协议帧。

- [ ] **Step 1: 写失败测试 tests/test_protocol.py**

```python
from app.sse_protocol import sse_error, sse_event


def test_token_frame_exact_bytes():
    # 与 Java StreamParser 解析规则逐字节对齐：event: 前缀、data: JSON、\n\n 分隔
    frame = sse_event("token", {"content": "你好"})
    assert frame == 'event: token\ndata: {"content": "你好"}\n\n'


def test_thinking_frame():
    frame = sse_event("thinking", {"content": "让我想想"})
    assert frame.startswith("event: thinking\n")
    assert '{"content": "让我想想"}' in frame
    assert frame.endswith("\n\n")


def test_done_frame():
    frame = sse_event("done", {"messageId": "py-abc123"})
    assert frame == 'event: done\ndata: {"messageId": "py-abc123"}\n\n'


def test_error_frame():
    frame = sse_error(429, "请求过于频繁，请稍后再试")
    assert frame == 'event: error\ndata: {"code": 429, "message": "请求过于频繁，请稍后再试"}\n\n'


def test_non_ascii_not_escaped():
    # 与 agent_stub.py 一致：ensure_ascii=False，中文原样输出
    assert "\\u" not in sse_event("token", {"content": "中文测试"})


def test_empty_content():
    assert sse_event("token", {"content": ""}) == 'event: token\ndata: {"content": ""}\n\n'
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python -m pytest tests/test_protocol.py -v
```

Expected: FAIL（ModuleNotFoundError: app.sse_protocol）

- [ ] **Step 3: 实现 app/sse_protocol.py**

```python
# -*- coding: utf-8 -*-
"""SSE 帧序列化：与 Java PythonSseClient.StreamParser 的解析规则逐字节对齐。

帧格式：event: <名>\ndata: <JSON>\n\n（帧以 \n\n 分隔；行以 event:/data: 前缀识别，
Java 侧 trim 后 startsWith 判断，JSON 用 Jackson 解析，中文可原样输出）。
"""
import json


def sse_event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_error(code: int, message: str) -> str:
    return sse_event("error", {"code": code, "message": message})
```

- [ ] **Step 4: 跑测试确认通过**

```bash
python -m pytest tests/test_protocol.py -v
```

Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add app/sse_protocol.py tests/test_protocol.py && git commit -m "feat: SSE 协议帧序列化"
```

---

### Task 4: 活动流注册表（stop 取消语义）

**Files:**
- Create: `app/stream_registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: 无
- Produces: `class StreamRegistry`，方法 `register(conversation_id: str, task: asyncio.Task) -> None`（同会话旧任务自动取消）、`get(conversation_id) -> asyncio.Task | None`、`cancel(conversation_id) -> bool`、`remove(conversation_id) -> None`；模块级单例 `registry`。Task 8 使用。

- [ ] **Step 1: 写失败测试 tests/test_registry.py**

```python
import asyncio

import pytest

from app.stream_registry import StreamRegistry


async def _sleeper():
    await asyncio.sleep(60)


@pytest.mark.asyncio
async def test_register_and_get():
    reg = StreamRegistry()
    task = asyncio.create_task(_sleeper())
    reg.register("c1", task)
    assert reg.get("c1") is task
    reg.remove("c1")
    assert reg.get("c1") is None


@pytest.mark.asyncio
async def test_register_duplicate_cancels_old():
    reg = StreamRegistry()
    old = asyncio.create_task(_sleeper())
    new = asyncio.create_task(_sleeper())
    reg.register("c1", old)
    reg.register("c1", new)
    await asyncio.sleep(0)  # 让取消生效
    assert old.cancelled()
    assert reg.get("c1") is new


@pytest.mark.asyncio
async def test_register_duplicate_ignores_finished_old():
    reg = StreamRegistry()
    old = asyncio.create_task(asyncio.sleep(0))
    await old
    new = asyncio.create_task(_sleeper())
    reg.register("c1", old)  # old 已完成，不应误取消
    reg.register("c1", new)
    assert not old.cancelled()
    assert reg.get("c1") is new


@pytest.mark.asyncio
async def test_cancel_active_returns_true():
    reg = StreamRegistry()
    task = asyncio.create_task(_sleeper())
    reg.register("c1", task)
    assert reg.cancel("c1") is True
    await asyncio.sleep(0)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_cancel_missing_returns_false():
    reg = StreamRegistry()
    assert reg.cancel("nobody") is False


@pytest.mark.asyncio
async def test_remove_idempotent():
    reg = StreamRegistry()
    reg.remove("nobody")  # 不抛异常
    task = asyncio.create_task(_sleeper())
    reg.register("c1", task)
    reg.remove("c1")
    reg.remove("c1")
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python -m pytest tests/test_registry.py -v
```

Expected: FAIL（ModuleNotFoundError: app.stream_registry）

- [ ] **Step 3: 实现 app/stream_registry.py**

```python
# -*- coding: utf-8 -*-
"""活动流登记表：conversationId -> 生成任务。

Java 侧已有同会话 409 守卫，这里是 Python 侧兜底：
同会话新流到来时取消旧任务（不再生成），stop 即取消。
"""
import asyncio


class StreamRegistry:

    def __init__(self) -> None:
        self._streams: dict[str, asyncio.Task] = {}

    def register(self, conversation_id: str, task: asyncio.Task) -> None:
        old = self._streams.get(conversation_id)
        if old is not None and not old.done():
            old.cancel()
        self._streams[conversation_id] = task

    def get(self, conversation_id: str) -> asyncio.Task | None:
        return self._streams.get(conversation_id)

    def cancel(self, conversation_id: str) -> bool:
        task = self._streams.get(conversation_id)
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    def remove(self, conversation_id: str) -> None:
        self._streams.pop(conversation_id, None)


registry = StreamRegistry()
```

- [ ] **Step 4: 跑测试确认通过**

```bash
python -m pytest tests/test_registry.py -v
```

Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add app/stream_registry.py tests/test_registry.py && git commit -m "feat: 活动流注册表（stop 取消语义）"
```

---

### Task 5: 博客检索三工具（直连 ES）

**Files:**
- Create: `app/agent/__init__.py`（空文件）
- Create: `app/agent/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `Settings`（Task 2）
- Produces: `build_es_client(settings: Settings) -> httpx.AsyncClient`；`build_tools(settings: Settings, client: httpx.AsyncClient | None = None) -> list`，返回 `[search_articles, get_article_content, list_articles]` 三个 `@tool`。Task 7 把工具列表交给 ToolNode。

**设计**（规格 §5）：三个工具均直连 ES `articles` 索引，只读；查询失败返回错误文本给模型兜底（不抛异常）；高亮片段去 `<em>` 标签；返回值是 JSON 字符串（`ensure_ascii=False`，LLM 友好）。

- [ ] **Step 1: 写失败测试 tests/test_tools.py**

```python
import json

import httpx
import pytest

from app.agent.tools import build_es_client, build_tools
from app.config import Settings


def make_settings() -> Settings:
    return Settings(
        es_url="http://es.test:9200",
        es_username="elastic",
        es_password="pw123",
        search_page_size=5,
        article_content_max_chars=100,
        _env_file=None,
    )


SEARCH_HIT = {
    "_source": {"id": "2088", "title": "SSE协议", "tags": [], "createdAt": "2026-08-15T21:52:52.000"},
    "highlight": {"content": ["<em>服务</em>端会一直主动推送消息"]},
}


@pytest.mark.asyncio
async def test_search_articles_query_shape_and_auth():
    """断言查询体（status 过滤、highlight、多字段匹配）与 BasicAuth 头。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization", "")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"hits": {"hits": [SEARCH_HIT]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client)
    result = await tools[0].ainvoke({"keyword": "微服务"})

    body = captured["body"]
    assert captured["url"] == "http://es.test/articles/_search"
    assert captured["auth"].startswith("Basic ")
    assert body["query"]["bool"]["filter"] == {"term": {"status": "published"}}
    assert body["query"]["bool"]["must"]["multi_match"]["query"] == "微服务"
    assert body["query"]["bool"]["must"]["multi_match"]["fields"] == ["title^3", "content", "summary"]
    assert body["size"] == 5
    assert "title" in body["highlight"]["fields"]
    # 高亮片段去标签、结果可解析、含标题
    assert "<em>" not in result
    assert "SSE协议" in result
    assert json.loads(result)[0]["title"] == "SSE协议"


@pytest.mark.asyncio
async def test_get_article_content_truncates():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://es.test/articles/_doc/a1"
        return httpx.Response(200, json={
            "_source": {"id": "a1", "title": "响应式编程", "content": "x" * 300}
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client)
    result = await tools[1].ainvoke({"article_id": "a1"})

    assert "响应式编程" in result
    assert len(json.loads(result)["content"]) == 100  # article_content_max_chars=100


@pytest.mark.asyncio
async def test_get_article_content_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"found": False})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client)
    result = await tools[1].ainvoke({"article_id": "ghost"})
    assert "未找到" in result


@pytest.mark.asyncio
async def test_list_articles_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["sort"] == [{"createdAt": "desc"}]
        assert body["from"] == 10  # page=2, size=10
        return httpx.Response(200, json={"hits": {"hits": [SEARCH_HIT]}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client)
    result = await tools[2].ainvoke({"page": 2, "page_size": 10})
    assert "SSE协议" in result


@pytest.mark.asyncio
async def test_tool_error_returns_text_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(make_settings(), client=client)
    result = await tools[0].ainvoke({"keyword": "任意"})
    assert "搜索失败" in result  # 错误文本回给模型兜底，不抛异常


@pytest.mark.asyncio
async def test_no_auth_when_username_empty():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json={"hits": {"hits": []}})

    settings = make_settings()
    settings.es_username = ""
    settings.es_password = ""
    client = build_es_client(settings)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://es.test")
    tools = build_tools(settings, client=client)
    await tools[0].ainvoke({"keyword": "x"})
    assert captured["auth"] == ""
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python -m pytest tests/test_tools.py -v
```

Expected: FAIL（ModuleNotFoundError: app.agent.tools）

- [ ] **Step 3: 实现 app/agent/tools.py**

```python
# -*- coding: utf-8 -*-
"""博客检索三工具：直连 ES articles 索引（只读）。

查询逻辑是 search-service 的 Python 侧最小重实现（status 过滤 + highlight）。
工具执行失败不抛异常：错误文本回给模型，由模型兜底回答（规格 §6）。
"""
import json
import re

import httpx
from langchain_core.tools import tool

from app.config import Settings

_EM_TAG = re.compile(r"</?em>")


def build_es_client(settings: Settings) -> httpx.AsyncClient:
    auth = (settings.es_username, settings.es_password) if settings.es_username else None
    return httpx.AsyncClient(base_url=settings.es_url, auth=auth, timeout=10.0)


def _clean_highlight(text: str) -> str:
    return _EM_TAG.sub("", text)


def build_tools(settings: Settings, client: httpx.AsyncClient | None = None) -> list:
    client = client or build_es_client(settings)

    @tool
    async def search_articles(keyword: str) -> str:
        """按关键词搜索博客已发布文章。

        Args:
            keyword: 搜索关键词，如“微服务”“SSE”
        Returns:
            JSON 列表：每项含 id、title、tags、createdAt 与命中内容片段（highlight）
        """
        body = {
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
            "size": settings.search_page_size,
            "_source": ["id", "title", "tags", "createdAt"],
        }
        try:
            resp = await client.post("/articles/_search", json=body)
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

    @tool
    async def get_article_content(article_id: str) -> str:
        """获取指定文章的完整内容。

        Args:
            article_id: 文章 id（先通过 search_articles 获得）
        Returns:
            JSON：title 与 content（超出上限截断，并注明已截断）
        """
        try:
            resp = await client.get(f"/articles/_doc/{article_id}")
        except httpx.HTTPError as e:
            return f"读取文章失败：{e.__class__.__name__}，请告知用户文章读取暂不可用"
        if resp.status_code == 404:
            return "未找到该文章，请确认文章 id 是否正确"
        resp.raise_for_status()
        src = resp.json().get("_source", {})
        content = src.get("content") or ""
        truncated = len(content) > settings.article_content_max_chars
        if truncated:
            content = content[: settings.article_content_max_chars]
        return json.dumps(
            {
                "title": src.get("title"),
                "content": content,
                "truncated": truncated,
            },
            ensure_ascii=False,
        )

    @tool
    async def list_articles(page: int = 1, page_size: int = 10) -> str:
        """按发布时间倒序浏览已发布文章列表。

        Args:
            page: 页码（从 1 开始）
            page_size: 每页条数（默认 10）
        Returns:
            JSON 列表：每项含 id、title、summary、tags、createdAt
        """
        body = {
            "query": {"term": {"status": "published"}},
            "sort": [{"createdAt": "desc"}],
            "from": max(page - 1, 0) * page_size,
            "size": page_size,
            "_source": ["id", "title", "summary", "tags", "createdAt"],
        }
        try:
            resp = await client.post("/articles/_search", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return f"浏览文章失败：{e.__class__.__name__}，请告知用户文章列表暂不可用"
        hits = resp.json().get("hits", {}).get("hits", [])
        items = [h.get("_source", {}) for h in hits]
        return json.dumps(items, ensure_ascii=False)

    return [search_articles, get_article_content, list_articles]
```

- [ ] **Step 4: 跑测试确认通过**

```bash
python -m pytest tests/test_tools.py -v
```

Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add app/agent/tools.py app/agent/__init__.py tests/test_tools.py && git commit -m "feat: 博客检索三工具（直连 ES）"
```

---

### Task 6: 模型工厂与系统提示词

**Files:**
- Create: `app/llm.py`
- Create: `app/agent/prompts.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Settings`（Task 2）、`scripts/probe_result.md` 的探测结论（Task 1，默认按策略 A：`extra_body={"thinking": {"type": "enabled"}}`）
- Produces: `select_model_name(model_field: str | None, settings: Settings) -> str`；`build_chat_model(deep_thinking: bool, model_field: str | None, settings: Settings) -> BaseChatModel`；`MockChatModel`（运行时联调假模型，`MOCK_LLM=1` 时使用）。Task 7/8 使用。

**背景**（规格 §3）：`deepThinking=false` → `deepseek-chat`；`true` + `THINKING_MODE=hybrid` → `deepseek-chat` + thinking 参数；`true` + `THINKING_MODE=reasoner` → `deepseek-reasoner`（不支持工具，模型自行降级）。`model` 字段经 allowlist 校验后透传。

- [ ] **Step 1: 写失败测试 tests/test_llm.py**

```python
import pytest
from langchain_deepseek import ChatDeepSeek

from app.config import Settings
from app.llm import MockChatModel, build_chat_model, select_model_name


def make_settings(**kw) -> Settings:
    base = dict(
        deepseek_api_key="sk-test",
        model_default="deepseek-chat",
        model_allowlist=["deepseek-chat", "deepseek-reasoner"],
        thinking_mode="hybrid",
        mock_llm=False,
    )
    base.update(kw)
    return Settings(**base, _env_file=None)


def test_select_default():
    assert select_model_name(None, make_settings()) == "deepseek-chat"
    assert select_model_name("default", make_settings()) == "deepseek-chat"
    assert select_model_name("", make_settings()) == "deepseek-chat"


def test_select_passthrough_allowed():
    assert select_model_name("deepseek-reasoner", make_settings()) == "deepseek-reasoner"


def test_select_rejects_unknown():
    with pytest.raises(ValueError, match="不允许"):
        select_model_name("gpt-4", make_settings())


def test_build_plain_chat():
    m = build_chat_model(False, "default", make_settings())
    assert isinstance(m, ChatDeepSeek)
    assert m.model_name == "deepseek-chat"


def test_build_hybrid_thinking_uses_extra_body():
    m = build_chat_model(True, "default", make_settings(thinking_mode="hybrid"))
    assert m.model_name == "deepseek-chat"
    assert m.extra_body == {"thinking": {"type": "enabled"}}


def test_build_reasoner_mode_switches_model():
    m = build_chat_model(True, "default", make_settings(thinking_mode="reasoner"))
    assert m.model_name == "deepseek-reasoner"


def test_build_mock_mode():
    m = build_chat_model(True, "default", make_settings(mock_llm=True))
    assert isinstance(m, MockChatModel)
    assert m.deep_thinking is True


@pytest.mark.asyncio
async def test_mock_model_stream_plain():
    m = MockChatModel(deep_thinking=False)
    chunks = [c async for c in m.astream([("human", "你好")])]
    text = "".join(c.content for c in chunks)
    assert text == "【MOCK】收到：你好"
    assert all(not c.additional_kwargs.get("reasoning_content") for c in chunks)


@pytest.mark.asyncio
async def test_mock_model_stream_thinking():
    m = MockChatModel(deep_thinking=True)
    chunks = [c async for c in m.astream([("human", "你好")])]
    reasoning = "".join(c.additional_kwargs.get("reasoning_content") or "" for c in chunks)
    assert "模拟思考" in reasoning
    text = "".join(c.content for c in chunks)
    assert text.startswith("【MOCK】")
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python -m pytest tests/test_llm.py -v
```

Expected: FAIL（ModuleNotFoundError: app.llm）

- [ ] **Step 3: 实现 app/agent/prompts.py**

```python
# -*- coding: utf-8 -*-
"""系统提示词。"""

SYSTEM_PROMPT = """你是 oy 的个人技术博客 AI 助手，服务于博客访客。
行为准则：
1. 用中文回答，友好、简洁、条理清晰；技术问题回答要准确，不确定时明说。
2. 当用户询问博客内容（有没有某类文章、某篇文章讲了什么、有哪些文章等）时，
   必须优先调用工具检索，再基于检索结果回答；引用文章时给出标题。
3. 回答"博客里有没有…"类问题时，先调用 search_articles 再下结论，不要凭空猜测。
4. 用户需要写作帮助（大纲、草稿）时，先用工具收集博客相关文章作为素材，
   再产出结构化内容，并注明素材出处。
5. 工具不可用时，如实告知检索暂不可用，并尽力用已有知识回答。
6. 不要编造博客中不存在的文章或链接。"""
```

- [ ] **Step 4: 实现 app/llm.py**

```python
# -*- coding: utf-8 -*-
"""模型工厂：DeepSeek 模型选择、thinking 策略、联调假模型。

探测结论（scripts/probe_result.md）默认验证通过策略 A：
deepseek-chat + extra_body={"thinking": {"type": "enabled"}}，
流式 chunk 的 additional_kwargs["reasoning_content"] 即思考内容。
"""
from typing import Any, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_deepseek import ChatDeepSeek

from app.config import Settings


def select_model_name(model_field: str | None, settings: Settings) -> str:
    name = settings.model_default if not model_field or model_field == "default" else model_field
    if name not in settings.model_allowlist:
        raise ValueError(f"模型 {name} 不在允许列表 {settings.model_allowlist} 中")
    return name


def build_chat_model(deep_thinking: bool, model_field: str | None, settings: Settings) -> BaseChatModel:
    if settings.mock_llm:
        return MockChatModel(deep_thinking=deep_thinking)
    name = select_model_name(model_field, settings)
    kwargs: dict[str, Any] = {
        "model": name,
        "api_key": settings.deepseek_api_key,
        "base_url": settings.deepseek_base_url,
        "streaming": True,
    }
    if deep_thinking:
        if settings.thinking_mode == "hybrid":
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:  # reasoner：纯推理模型，不支持工具调用（规格 §3 策略 B）
            kwargs["model"] = "deepseek-reasoner"
    return ChatDeepSeek(**kwargs)


class MockChatModel(BaseChatModel):
    """联调假模型（MOCK_LLM=1）：输出思考流 + 固定回复，不花 API 额度。"""

    deep_thinking: bool = False

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        last = str(messages[-1].content)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"【MOCK】收到：{last}"))])

    async def _astream(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> Iterator[AIMessageChunk]:
        import asyncio

        last = str(messages[-1].content)
        if self.deep_thinking:
            yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": "模拟思考过程……"})
            await asyncio.sleep(0.05)
        for ch in f"【MOCK】收到：{last}":
            yield AIMessageChunk(content=ch)
            await asyncio.sleep(0.05)
```

- [ ] **Step 5: 跑测试确认通过**

```bash
python -m pytest tests/test_llm.py -v
```

Expected: 9 passed

- [ ] **Step 6: 提交**

```bash
git add app/llm.py app/agent/prompts.py tests/test_llm.py && git commit -m "feat: DeepSeek 模型工厂与系统提示词"
```

---

### Task 7: LangGraph Agent 图

**Files:**
- Create: `app/agent/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `SYSTEM_PROMPT`（Task 6）、`build_tools`（Task 5）、`build_chat_model`/`get_settings`（Task 6/2）
- Produces: `class AgentState`（`messages: Annotated[list, add_messages]`）；`messages_from_request(history: list[dict], user_message: str) -> list[BaseMessage]`；`build_graph(model: BaseChatModel, tools: list) -> CompiledStateGraph`；`get_graph(deep_thinking: bool, model_field: str | None) -> CompiledStateGraph`（lru_cache 按参数缓存）。Task 8 使用。

**背景**（规格 §3）：`START → agent(bind_tools) → tools_condition → tools(ToolNode) ⇄ agent → END`，`recursion_limit=10`。

- [ ] **Step 1: 写失败测试 tests/test_graph.py**

```python
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.agent.graph import AgentState, build_graph, messages_from_request


def test_messages_from_request_builds_sequence():
    history = [
        {"role": "user", "content": "博客里有微服务文章吗"},
        {"role": "assistant", "content": "有的"},
    ]
    msgs = messages_from_request(history, "给我讲讲第一篇")
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage) and msgs[1].content == "博客里有微服务文章吗"
    assert isinstance(msgs[2], AIMessage) and msgs[2].content == "有的"
    assert isinstance(msgs[3], HumanMessage) and msgs[3].content == "给我讲讲第一篇"


def test_messages_from_request_skips_unknown_roles_and_none_history():
    history = [
        {"role": "system", "content": "不该出现"},
        {"role": "user", "content": None},
        {"role": "user", "content": "有效消息"},
    ]
    msgs = messages_from_request(history, "新问题")
    assert len(msgs) == 3  # SystemMessage + 有效历史 + 新问题
    assert messages_from_request(None, "hi")[1].content == "hi"


@tool
def dummy_tool(x: str) -> str:
    """测试工具。"""
    return f"tool said {x}"


@pytest.mark.asyncio
async def test_graph_tool_call_roundtrip():
    """agent 节点发起工具调用 -> ToolNode 执行 -> 结果回填 -> 最终回答。"""
    model = FakeListChatModel(responses=[
        AIMessage(
            content="",
            tool_calls=[{"name": "dummy_tool", "args": {"x": "1"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(content="答案是 42"),
    ])
    graph = build_graph(model, [dummy_tool])
    result = await graph.ainvoke({"messages": [HumanMessage(content="hi")]})
    msgs: list = result["messages"]
    assert msgs[-1].content == "答案是 42"
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "tool said 1"


@pytest.mark.asyncio
async def test_graph_no_tool_call_straight_answer():
    model = FakeListChatModel(responses=[AIMessage(content="直接回答")])
    graph = build_graph(model, [dummy_tool])
    result = await graph.ainvoke({"messages": [HumanMessage(content="hi")]})
    assert result["messages"][-1].content == "直接回答"


def test_agent_state_shape():
    # AgentState 必须含 messages 且 add_messages 合并语义
    state: AgentState = {"messages": []}
    assert "messages" in state
```

- [ ] **Step 2: 跑测试确认失败**

```bash
python -m pytest tests/test_graph.py -v
```

Expected: FAIL（ModuleNotFoundError: app.agent.graph）

- [ ] **Step 3: 实现 app/agent/graph.py**

```python
# -*- coding: utf-8 -*-
"""LangGraph Agent 图：agent 节点（模型+工具绑定）+ tools 节点 + 条件边。

规格 §3：START → agent → (tools_condition) → tools ⇄ agent → END。
无 checkpointer（无状态，规格 §12）；recursion_limit 防工具循环。
"""
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import build_tools
from app.config import get_settings
from app.llm import build_chat_model


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def messages_from_request(history: list[dict] | None, user_message: str) -> list[BaseMessage]:
    """Java 传来的 history（[{role, content}]）转 LangChain messages，末尾追加用户新消息。"""
    msgs: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in history or []:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
        # 其余 role 忽略（Java 只会传 user/assistant）
    msgs.append(HumanMessage(content=user_message))
    return msgs


def build_graph(model: BaseChatModel, tools: list) -> StateGraph:
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


@lru_cache(maxsize=8)
def get_graph(deep_thinking: bool, model_field: str | None) -> StateGraph:
    settings = get_settings()
    model = build_chat_model(deep_thinking, model_field, settings)
    tools = build_tools(settings)
    return build_graph(model, tools)
```

注意：`build_graph` 的返回注解写 `StateGraph`（编译结果），若 mypy/IDE 报类型不符，改为 `CompiledStateGraph`（`from langgraph.graph.state import CompiledStateGraph`）。

- [ ] **Step 4: 跑测试确认通过**

```bash
python -m pytest tests/test_graph.py -v
```

Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add app/agent/graph.py tests/test_graph.py && git commit -m "feat: LangGraph Agent 图（agent+tools 节点）"
```

---

### Task 8: FastAPI 端点与流式集成

**Files:**
- Create: `app/main.py`
- Create: `tests/conftest.py`
- Test: `tests/test_stream.py`

**Interfaces:**
- Consumes: `sse_event`/`sse_error`（Task 3）、`registry`（Task 4）、`get_graph`/`messages_from_request`（Task 7）、`get_settings`（Task 2）
- Produces: `app: FastAPI`（`POST /chat/stream`、`POST /chat/stop`）、`map_model_error(e: Exception) -> tuple[int, str]`。uvicorn 入口：`python -m uvicorn app.main:app`。

**流式架构**（规格 §9）：请求内创建 worker 协程跑 `graph.astream(stream_mode="messages")`，chunk 经 asyncio.Queue 转给响应生成器逐帧 yield；worker 被取消时（stop/客户端断开）往队列放 stop 哨兵，生成器静默收尾。映射规则：`reasoning_content`→thinking、`content`→token、其余静默、正常结束→done。

- [ ] **Step 1: 写失败测试 tests/conftest.py（先于实现创建，供测试环境注入 mock）**

```python
# -*- coding: utf-8 -*-
"""测试夹具：MOCK_LLM=1 环境 + httpx ASGITransport 客户端。"""
import os

# 必须在 import app.main 之前设置，使 get_settings() 读到 mock 模式
os.environ.setdefault("MOCK_LLM", "1")

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.agent.graph import get_graph  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_caches():
    get_settings.cache_clear()
    get_graph.cache_clear()
    yield
    get_settings.cache_clear()
    get_graph.cache_clear()


@pytest.fixture
async def client():
    from app import main

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        yield c
```

- [ ] **Step 2: 写失败测试 tests/test_stream.py**

```python
import asyncio
import json

import pytest

from app.main import map_model_error

PAYLOAD = {
    "conversationId": "conv_1",
    "userId": "u1",
    "message": "你好",
    "history": [{"role": "user", "content": "以前问过的问题"}],
    "deepThinking": False,
    "model": "default",
}


def parse_frames(text: str) -> list[tuple[str, dict]]:
    """按 Java StreamParser 规则解析 SSE 文本为 (event, data) 列表。"""
    frames = []
    for block in text.split("\n\n"):
        event, data = "", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if event:
            frames.append((event, json.loads(data)))
    return frames


async def read_all(client, payload: dict) -> list[tuple[str, dict]]:
    async with client.stream("POST", "/chat/stream", json=payload) as resp:
        text = await resp.aread()
    return parse_frames(text)


@pytest.mark.asyncio
async def test_normal_stream_token_then_done(client):
    frames = await read_all(client, PAYLOAD)
    events = [e for e, _ in frames]
    assert "thinking" not in events  # deepThinking=false 无思考
    assert events[0] == "token"
    assert events[-1] == "done"
    content = "".join(d["content"] for e, d in frames if e == "token")
    assert content == "【MOCK】收到：你好"
    done_id = frames[-1][1]["messageId"]
    assert done_id.startswith("py-")


@pytest.mark.asyncio
async def test_deep_thinking_emits_thinking_first(client):
    payload = {**PAYLOAD, "deepThinking": True}
    frames = await read_all(client, payload)
    events = [e for e, _ in frames]
    assert events[0] == "thinking"
    assert events[1] == "token"
    reasoning = "".join(d["content"] for e, d in frames if e == "thinking")
    assert "模拟思考" in reasoning
    assert events[-1] == "done"


@pytest.mark.asyncio
async def test_empty_message_returns_error_400(client):
    frames = await read_all(client, {**PAYLOAD, "message": "   "})
    assert frames == [("error", {"code": 400, "message": "参数不完整"})]


@pytest.mark.asyncio
async def test_missing_conversation_id_returns_error_400(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "conversationId"}
    frames = await read_all(client, payload)
    assert frames[0][0] == "error"
    assert frames[0][1]["code"] == 400


@pytest.mark.asyncio
async def test_stop_cancels_stream_without_done(client):
    frames: list[tuple[str, dict]] = []

    async def reader():
        async with client.stream("POST", "/chat/stream", json=PAYLOAD) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line[len("event:"):].strip()
                    frames.append((event, {}))

    task = asyncio.create_task(reader())
    # 等首帧到达后触发 stop（Mock 每 chunk 间隔 0.05s，流长 >0.5s）
    for _ in range(50):
        if frames:
            break
        await asyncio.sleep(0.05)
    assert frames, "首帧未到达，Mock 流未启动"

    resp = await client.post("/chat/stop", json={"conversationId": "conv_1"})
    assert resp.json() == {"ok": True, "conversationId": "conv_1"}

    await asyncio.wait_for(task, timeout=10)
    events = [e for e, _ in frames]
    assert "done" not in events  # stop 后静默收尾，无 done


@pytest.mark.asyncio
async def test_new_stream_replaces_old_same_conversation(client):
    frames1: list[str] = []

    async def reader1():
        async with client.stream("POST", "/chat/stream", json=PAYLOAD) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    frames1.append(line)

    task1 = asyncio.create_task(reader1())
    for _ in range(50):
        if frames1:
            break
        await asyncio.sleep(0.05)
    assert frames1

    # 同会话开新流：旧流被取消（静默结束），新流正常完成
    frames2 = await read_all(client, PAYLOAD)
    await asyncio.wait_for(task1, timeout=10)
    assert frames2[-1][0] == "done"


def test_map_model_error_429():
    code, msg = map_model_error(Exception("429 status code: Too Many Requests"))
    assert code == 429
    assert "频繁" in msg


def test_map_model_error_timeout():
    code, msg = map_model_error(Exception("connection timeout after 60s"))
    assert code == 504
    assert "超时" in msg


def test_map_model_error_unknown():
    code, msg = map_model_error(Exception("乱七八糟"))
    assert code == 500
```

- [ ] **Step 3: 跑测试确认失败**

```bash
python -m pytest tests/test_stream.py -v
```

Expected: FAIL（ModuleNotFoundError: app.main）

- [ ] **Step 4: 实现 app/main.py**

```python
# -*- coding: utf-8 -*-
"""FastAPI 入口：Java↔Python 协议的两个端点（规格 §4）。

POST /chat/stream -> SSE（token/thinking/done/error）
POST /chat/stop   -> {"ok": true}
无鉴权，仅内网直连。
"""
import asyncio
import logging
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk

from app.agent.graph import get_graph, messages_from_request
from app.config import get_settings
from app.sse_protocol import sse_error, sse_event
from app.stream_registry import registry

logger = logging.getLogger(__name__)

app = FastAPI(title="blog-agent")

settings = get_settings()
settings.require_api_key()  # 启动 fail-fast：非 mock 模式缺 key 直接报错


def map_model_error(e: Exception) -> tuple[int, str]:
    """模型/上游异常 -> 协议 (code, message)。"""
    text = str(e).lower()
    if "429" in text:
        return 429, "请求过于频繁，请稍后再试"
    if "401" in text or "invalid api key" in text:
        return 401, "API 密钥无效或已过期"
    if "402" in text or "insufficient" in text:
        return 402, "API 余额不足"
    if "timeout" in text:
        return 504, "AI 服务响应超时，请重试"
    return 500, "AI 服务暂不可用，请稍后重试"


def _stream_chunk_frames(chunk: AIMessageChunk) -> list[str]:
    """单 chunk -> 协议帧列表（规格 §3 映射表）。"""
    frames: list[str] = []
    reasoning = chunk.additional_kwargs.get("reasoning_content")
    if reasoning:
        frames.append(sse_event("thinking", {"content": reasoning}))
    if chunk.content:
        frames.append(sse_event("token", {"content": chunk.content}))
    return frames


@app.post("/chat/stream")
async def chat_stream(body: dict) -> StreamingResponse:
    conversation_id = str(body.get("conversationId") or "")
    message = str(body.get("message") or "").strip()
    deep_thinking = bool(body.get("deepThinking"))
    model = body.get("model")
    history = body.get("history") or []

    if not conversation_id or not message:
        async def error_once():
            yield sse_error(400, "参数不完整")
        return StreamingResponse(error_once(), media_type="text/event-stream")

    state = {"messages": messages_from_request(history, message)}

    async def gen():
        graph = get_graph(deep_thinking, model)
        queue: asyncio.Queue = asyncio.Queue()

        async def worker():
            try:
                async for chunk, _meta in graph.astream(state, stream_mode="messages"):
                    await queue.put(("chunk", chunk))
                await queue.put(("end", None))
            except asyncio.CancelledError:
                # stop / 客户端断开：放哨兵后结束（生成器静默收尾，不发 done）
                try:
                    await queue.put(("stop", None))
                finally:
                    raise
            except Exception as e:
                logger.exception("agent stream failed")
                await queue.put(("error", e))

        task = asyncio.create_task(worker())
        registry.register(conversation_id, task)
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "chunk":
                    if isinstance(payload, AIMessageChunk):
                        for frame in _stream_chunk_frames(payload):
                            yield frame
                elif kind == "end":
                    yield sse_event("done", {"messageId": f"py-{uuid.uuid4().hex}"})
                    return
                elif kind == "stop":
                    return  # 静默收尾（规格 §6）
                else:
                    code, msg = map_model_error(payload)
                    yield sse_error(code, msg)
                    return
        finally:
            registry.remove(conversation_id)
            if not task.done():
                task.cancel()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/chat/stop")
async def chat_stop(body: dict) -> dict:
    conversation_id = str(body.get("conversationId") or "")
    registry.cancel(conversation_id)
    return {"ok": True, "conversationId": conversation_id}
```

- [ ] **Step 5: 跑测试确认通过**

```bash
python -m pytest tests/test_stream.py -v
```

Expected: 10 passed（若 stop 用例偶发时序失败，重跑一次确认不是真 bug；连挂两次按 systematic-debugging 排查）

- [ ] **Step 6: 全量回归**

```bash
python -m pytest -v
```

Expected: 全部通过（48 个：test_config 6 + test_protocol 6 + test_registry 6 + test_tools 6 + test_llm 9 + test_graph 5 + test_stream 10）

- [ ] **Step 7: 提交**

```bash
git add app/main.py tests/conftest.py tests/test_stream.py && git commit -m "feat: /chat/stream 与 /chat/stop 端点"
```

---

### Task 9: MOCK 模式协议集成验证 + README

**Files:**
- Create: `README.md`
- Test: 无新增测试（本任务是运行验证 + 文档）

**Interfaces:**
- Consumes: 全部（Task 1-8）
- Produces: `README.md`（启动方法、env 表、验证清单）。真实联调（Task 10）以它为手册。

- [ ] **Step 1: 启动服务（MOCK 模式，后台运行）**

```bash
source /d/tool1/anancoda/etc/profile.d/conda.sh && conda activate ai-agent && cd /g/agentWorkplace/BlogAgent && MOCK_LLM=1 python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

后台运行（`run_in_background`）。等待 `Uvicorn running on http://0.0.0.0:8001` 日志。

- [ ] **Step 2: curl 验证正常流**

```bash
curl -s -N -X POST http://localhost:8001/chat/stream -H "Content-Type: application/json" -d '{"conversationId":"itest-1","userId":"u1","message":"hello","history":[],"deepThinking":false,"model":"default"}'
```

Expected（对照 stub 行为）：连续 `event: token` 帧（内容【MOCK】收到：hello），最后 `event: done` + `data: {"messageId": "py-..."}`，每帧以空行分隔。

- [ ] **Step 3: curl 验证深度思考流**

```bash
curl -s -N -X POST http://localhost:8001/chat/stream -H "Content-Type: application/json" -d '{"conversationId":"itest-2","userId":"u1","message":"思考一下","history":[],"deepThinking":true,"model":"default"}'
```

Expected：首个事件为 `thinking`（含"模拟思考"），随后 `token`，最后 `done`。

- [ ] **Step 4: curl 验证错误分支**

```bash
curl -s -N -X POST http://localhost:8001/chat/stream -H "Content-Type: application/json" -d '{"conversationId":"itest-3","userId":"u1","message":"","history":[],"deepThinking":false}'
```

Expected：仅一个 `event: error`，`data: {"code": 400, "message": "参数不完整"}`。

- [ ] **Step 5: curl 验证 stop**

开两个终端：终端 A 发正常流请求（`-N` 持续观察），终端 B 在流进行中执行：

```bash
curl -s -X POST http://localhost:8001/chat/stop -H "Content-Type: application/json" -d '{"conversationId":"itest-1"}'
```

Expected：B 返回 `{"ok":true,...}`；A 的流立即结束且**没有** done 帧。

- [ ] **Step 6: 停掉服务，编写 README.md**

README 内容（结构固定）：
1. 项目简介（博客 AI Agent，替换 agent_stub.py，Java 零改动）
2. 环境准备（conda ai-agent、`pip install -r requirements.txt`、复制 .env.example 为 .env）
3. 环境变量表（照抄 .env.example 注释）
4. 本地运行（uvicorn 命令；`MOCK_LLM=1` 联调模式说明）
5. 与 Java agent-service 联调（`AGENT_PYTHON_URL` 说明；协议帧格式表：token/thinking/done/error）
6. 测试（`python -m pytest -v`）
7. 验收清单（照抄规格 §11 的 6 条，每条留 `- [ ]` 勾选框，Task 10 逐条勾选并记录结果）

- [ ] **Step 7: 提交**

```bash
git add README.md && git commit -m "docs: README 与 MOCK 联调验证说明"
```

---

### Task 10: 真实 DeepSeek 全链路联调（验收）

**Files:**
- Modify: `README.md`（勾选验收清单并记录实测结果）
- Test: 无新增测试（手动验收，按规格 §11）

**Interfaces:**
- Consumes: 全部。需要：真实 DEEPSEEK_API_KEY（.env 已配）、ES 可直连（Tailscale IP + 凭据已配）、Java 微服务与前端可运行（用户环境）

- [ ] **Step 1: 启动真实模式服务**

```bash
source /d/tool1/anancoda/etc/profile.d/conda.sh && conda activate ai-agent && cd /g/agentWorkplace/BlogAgent && python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Expected：启动无报错（.env 的 key 生效，fail-fast 通过）。

- [ ] **Step 2: 直连验证四能力（curl）**

依次执行并在 README 记录输出摘要：
1. 纯对话：`message="你好，请介绍一下你自己"` → 有 token 流，结尾 done
2. 深度思考：`deepThinking=true` → 先 thinking 流（DeepSeek 真实推理）后 token
3. RAG：`message="博客里有没有关于微服务的文章？"` → 输出应引用真实文章标题（对照 ES 中 12 篇文章验证准确性）
4. 文章内容：追问 `"给我讲讲《SSE协议》这篇文章"` → 模型调用 `get_article_content` 后回答内容准确
5. 写作辅助：`"帮我写一篇 Java 集合框架相关的博客大纲，参考博客已有文章"` → 结构化大纲且引用博客素材
6. 工具调用过程可见：服务日志出现 `articles/_search` 请求（tools 被真实调用）

- [ ] **Step 3: stop 真实验证**

流进行中调 `/chat/stop`，观察：生成立即停止、Java 侧不落库半截消息（对照规格 §9 中断路径）。若发现取消后模型仍在输出（DeepSeek 连接未断），记录现象并在下一步排查。

- [ ] **Step 4: Java agent-service 全链路**

启动 Java agent-service（8095）与网关，确认 `agent.python.base-url` 指向本机 8001（默认即 `http://localhost:8001`），从博客前端发消息验证：多轮历史正确、前端 SSE 实时显示、thinking 折叠展示、会话列表/消息落库正常、`/chat/stop` 按钮生效。

- [ ] **Step 5: 降级验证**

临时把 .env 的 `ES_URL` 改成一个不可达地址并重启服务，验证：纯对话仍可用；问博客内容时模型兜底告知检索不可用（而非 500 崩溃）。验证后改回。

- [ ] **Step 6: 填写验收清单并提交**

按实测结果勾选 README 验收清单（规格 §11 六条），失败的条目如实记录。全部通过后：

```bash
git add README.md && git commit -m "docs: 真实联调验收记录"
```

失败条目若涉及代码缺陷：停止本任务，另开 systematic-debugging 处理；涉及规格变更：回到 brainstorming 更新规格后再动代码。
