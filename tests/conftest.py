# -*- coding: utf-8 -*-
"""测试夹具：MOCK_LLM=1 环境 + httpx ASGITransport 客户端。

说明（task-9 修复轮移除测试专用补丁后保留的两项，均不改业务代码）：
1. MOCK_LLM 生命周期：收集期（import app.main 的 fail-fast 检查）需要该环境变量，
   但保持全局设置会让 test_config.test_defaults（断言 mock_llm=False）失败。
   故会话开跑后移除，stream 测试由 client fixture 按测试注入。
2. 增量 ASGI transport：httpx 自带 ASGITransport 会缓冲整个响应体后才返回，
   测试端无法在流进行中看到首帧；本实现逐块转发 app 的 send()（asyncio.Queue
   桥接），行为与真实 uvicorn 一致，供 SSE 时序用例使用。
MockChatModel 的 bind_tools / 逐 token 回调已由 app/llm.py 原生实现（不再打补丁）。
"""
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import os

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.agent.graph import get_graph  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _pop_mock_llm_env_after_collection():
    """收集期（app.main 导入的 fail-fast 检查）需要 MOCK_LLM=1；会话开跑后移除。

    保持全局设置会让 test_config.test_defaults 失败——Settings(_env_file=None) 仍会读
    os.environ，断言 mock_llm is False。stream 测试所需的 mock 模式由 client fixture
    按测试用 monkeypatch.setenv 注入（测试结束后自动还原）。
    """
    os.environ.pop("MOCK_LLM", None)
    yield


@pytest.fixture(autouse=True)
def _clear_caches():
    get_settings.cache_clear()
    get_graph.cache_clear()
    yield
    get_settings.cache_clear()
    get_graph.cache_clear()


class _IncrementalStream(httpx.AsyncByteStream):
    """响应体流：逐块从队列读出，None 为结束哨兵。"""

    def __init__(self, queue: asyncio.Queue, app_task: asyncio.Task):
        self._queue = queue
        self._app_task = app_task

    async def __aiter__(self):
        ended_cleanly = False
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    ended_cleanly = True
                    return
                yield chunk
        finally:
            # 仅客户端提前断开（未读到结束哨兵）时终止仍在跑的 app 任务
            if not ended_cleanly and not self._app_task.done():
                self._app_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._app_task


class IncrementalASGITransport(httpx.AsyncBaseTransport):
    """真正增量传输的 ASGI transport（供 SSE 时序用例）。

    httpx 0.28.1 自带的 ASGITransport 会先 `await app(...)` 等整个响应跑完，
    把全部 body 缓冲到内存后才返回 Response——测试端永远无法在流进行中看到首帧
    （test_stop_cancels_stream_without_done 这类时序用例结构上必挂）。
    本实现把 app 的 send() 逐块转成 httpx 响应流（asyncio.Queue 桥接），
    行为与真实 uvicorn 一致。仅测试环境使用。

    两个 ASGI 语义要点（缺一不可，否则 StreamingResponse 挂起）：
    1. scope 必须带 asgi.spec_version>=2.4（如 "3.0"）。缺省时 Starlette 走 2.0
       老路径，会起 listen_for_disconnect 任务等 receive() 返回 http.disconnect。
    2. 请求体消费完后，receive() 必须先等响应结束再返回 http.disconnect
       （与 httpx 自带 ASGITransport 行为一致），供 disconnect 监听正确收尾。
    """

    def __init__(self, app):
        self._app = app

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(k.lower(), v) for (k, v) in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": ("127.0.0.1", 123),
            "root_path": "",
        }

        req_iter = request.stream.__aiter__()
        request_complete = False

        async def receive():
            nonlocal request_complete
            if request_complete:
                # 请求体已消费完：等响应结束，再按 ASGI 语义返回 disconnect
                await response_complete.wait()
                return {"type": "http.disconnect"}
            try:
                body = await req_iter.__anext__()
            except StopAsyncIteration:
                request_complete = True
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.request", "body": body, "more_body": True}

        started = asyncio.Event()
        response_complete = asyncio.Event()
        body_q: asyncio.Queue = asyncio.Queue()
        result = {"status": None, "headers": None, "exc": None}

        async def send(message):
            if message["type"] == "http.response.start":
                result["status"] = message["status"]
                result["headers"] = message.get("headers", [])
                started.set()
            elif message["type"] == "http.response.body":
                if message.get("body"):
                    await body_q.put(message["body"])
                if not message.get("more_body", False):
                    response_complete.set()
                    await body_q.put(None)

        async def run():
            try:
                await self._app(scope, receive, send)
            except Exception as e:  # noqa: BLE001
                result["exc"] = e
            finally:
                response_complete.set()
                await body_q.put(None)  # 兜底：异常/提前结束也终止响应流

        app_task = asyncio.create_task(run())
        await started.wait()
        if result["exc"] is not None and result["status"] is None:
            app_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await app_task
            raise result["exc"]
        assert result["status"] is not None, "ASGI app 未发送 http.response.start"

        return httpx.Response(
            status_code=result["status"],
            headers=result["headers"],
            stream=_IncrementalStream(body_q, app_task),
            request=request,
        )


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")  # 本测试期间的 get_settings() 读到 mock 模式
    from app import main

    transport = IncrementalASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as c:
        yield c
