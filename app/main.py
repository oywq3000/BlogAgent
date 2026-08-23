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
                async for chunk, _meta in graph.astream(
                    state, stream_mode="messages", config={"recursion_limit": 10}
                ):
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
            registry.remove(conversation_id, task)
            if not task.done():
                task.cancel()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/chat/stop")
async def chat_stop(body: dict) -> dict:
    conversation_id = str(body.get("conversationId") or "")
    registry.cancel(conversation_id)
    return {"ok": True, "conversationId": conversation_id}
