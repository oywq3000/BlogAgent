# -*- coding: utf-8 -*-
"""模型工厂：DeepSeek 模型选择、thinking 策略、联调假模型。

探测结论（scripts/probe_result.md）默认验证通过策略 A：
deepseek-chat + extra_body={"thinking": {"type": "enabled"}}，
流式 chunk 的 additional_kwargs["reasoning_content"] 即思考内容。
"""
import time
from typing import Any, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_deepseek import ChatDeepSeek

from app.config import Settings


def select_model_name(model_field: str | None, settings: Settings) -> str:
    name = settings.model_default if not model_field or model_field == "default" else model_field
    if name not in settings.model_allowlist:
        raise ValueError(f"模型 {name} 不允许使用（不在允许列表 {settings.model_allowlist} 中）")
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

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """no-op 覆写：Mock 输出完全由脚本决定，与工具绑定无关。

        langchain-core 1.5.5 的 BaseChatModel.bind_tools 直接抛 NotImplementedError，
        而 build_graph 无条件调用 model.bind_tools(tools)（生产 ChatDeepSeek 才有真实
        实现）。与 tests/test_graph.py 的 FakeModelWithTools 同一先例，仅对假模型语义无损。
        """
        return self

    def _iter_chunks(self, last: str) -> Iterator[ChatGenerationChunk]:
        """统一的 chunk 序列：deep_thinking 思考帧在前 + 逐字内容帧（各路径共用）。"""
        if self.deep_thinking:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="", additional_kwargs={"reasoning_content": "模拟思考过程……"}
                )
            )
        for ch in f"【MOCK】收到：{last}":
            yield ChatGenerationChunk(message=AIMessageChunk(content=ch))

    def _emit_token_callbacks(self, last: str, run_manager) -> None:
        """同步路径逐 token 触发 on_llm_new_token（0.05s/帧）。

        langgraph 的 stream_mode="messages" 靠 on_llm_new_token 捕获逐 token 帧；
        同步路径（invoke/_stream）不显式回调的话只产出全量消息，token/thinking 帧
        全部丢失。chunk 必须是 ChatGenerationChunk（langgraph v1 捕获器会忽略非
        ChatGenerationChunk 的回调）。
        """
        for chunk in self._iter_chunks(last):
            token = chunk.message.additional_kwargs.get("reasoning_content") or chunk.message.content
            run_manager.on_llm_new_token(token, chunk=chunk)
            time.sleep(0.05)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        last = str(messages[-1].content)
        if run_manager is not None:
            self._emit_token_callbacks(last, run_manager)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"【MOCK】收到：{last}"))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        last = str(messages[-1].content)
        for chunk in self._iter_chunks(last):
            if run_manager is not None:
                token = chunk.message.additional_kwargs.get("reasoning_content") or chunk.message.content
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
            time.sleep(0.05)

    async def _astream(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        import asyncio

        last = str(messages[-1].content)
        for chunk in self._iter_chunks(last):
            yield chunk
            await asyncio.sleep(0.05)
