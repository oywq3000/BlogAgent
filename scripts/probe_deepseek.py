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
