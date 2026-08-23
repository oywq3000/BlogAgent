# -*- coding: utf-8 -*-
"""模型工厂：DeepSeek 模型选择、thinking 策略、联调假模型。

探测结论（scripts/probe_result.md）默认验证通过策略 A：
deepseek-chat + extra_body={"thinking": {"type": "enabled"}}，
流式 chunk 的 additional_kwargs["reasoning_content"] 即思考内容。
"""
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

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        last = str(messages[-1].content)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"【MOCK】收到：{last}"))])

    async def _astream(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        import asyncio

        last = str(messages[-1].content)
        if self.deep_thinking:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="", additional_kwargs={"reasoning_content": "模拟思考过程……"}
                )
            )
            await asyncio.sleep(0.05)
        for ch in f"【MOCK】收到：{last}":
            yield ChatGenerationChunk(message=AIMessageChunk(content=ch))
            await asyncio.sleep(0.05)
