# -*- coding: utf-8 -*-
"""函数调用底层演示：请求怎么发、响应怎么解析、坏 JSON 怎么办。

用法：
    cd /g/agentWorkplace/BlogAgent
    /d/tool1/anancoda/envs/ai-agent/python.exe scripts/probe_tool_call_wire.py

纯本地演示，不碰真实 DeepSeek / ES。展示四件事：
  1. Python 函数 -> 请求里 tools 参数的线格式（JSON Schema）
  2. 发给模型的完整请求体长什么样
  3. 模型返回原文 -> LangChain 解析（含一条坏 JSON，看它去哪）
  4. 解析失败后 ToolNode / tools_condition 的真实行为
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool


@tool
def search_articles(keyword: str) -> str:
    """按关键词搜索博客已发布文章。

    Args:
        keyword: 搜索关键词，如“微服务”“SSE”
    Returns:
        JSON 列表：每项含 id、title、tags、createdAt 与命中内容片段
    """
    return '[{"id": "a1", "title": "《微服务入门》"}]'


TOOLS = [search_articles]
MSGS = [
    SystemMessage(content="你是博客助手，查博客内容必须用工具。"),
    HumanMessage(content="博客里有微服务文章吗？"),
]


def part(title: str) -> None:
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


# ---------- 1. 函数 -> 请求里的 tools 参数 ----------
part("① 函数被序列化成什么：请求里 tools 参数的线格式")
schema = convert_to_openai_tool(search_articles)
print(json.dumps(schema, ensure_ascii=False, indent=2))

# ---------- 2. 完整请求体 ----------
part("② ChatDeepSeek 真实发出的完整请求体（拦截 HTTP 层）")
import httpx
from langchain_deepseek import ChatDeepSeek

captured: dict = {}

def _fake_response(request: httpx.Request) -> httpx.Response:
    """捕获请求体，返回一个不含 tool_call 的假响应。"""
    captured["url"] = str(request.url)
    captured["body"] = json.loads(request.content.decode("utf-8"))
    return httpx.Response(200, json={
        "id": "chatcmpl-fake", "object": "chat.completion", "created": 0,
        "model": "deepseek-chat",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "【假响应】"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })

_model = ChatDeepSeek(
    model="deepseek-chat",
    api_key="sk-test",
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(transport=httpx.MockTransport(_fake_response)),
    extra_body={"thinking": {"type": "enabled"}},  # 与 app/llm.py 的 hybrid 策略一致
)
_model.bind_tools(TOOLS).invoke(MSGS)  # 触发一次真实调用链（请求被 MockTransport 拦截）
print("请求 URL:", captured["url"])
print()
print("请求体（这就是打到 DeepSeek 的完整 JSON）:")
print(json.dumps(captured["body"], ensure_ascii=False, indent=2)[:1600])

# ---------- 3. 模型返回原文 -> 解析 ----------
part("③ 模型返回原文 -> LangChain 解析（故意带一条坏 JSON）")
raw_message = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        # 正常的一条
        {"id": "call_1", "type": "function",
         "function": {"name": "search_articles", "arguments": '{"keyword": "微服务"}'}},
        # 坏 JSON：少一个右花括号
        {"id": "call_2", "type": "function",
         "function": {"name": "search_articles", "arguments": '{"keyword": "微服务"'}},
    ],
}
print("模型返回原文（简化）:")
print(json.dumps(raw_message, ensure_ascii=False, indent=2))

from langchain_openai.chat_models.base import _convert_dict_to_message

msg = _convert_dict_to_message(raw_message)
print("\nLangChain 解析后的 AIMessage:")
print("  tool_calls        :", msg.tool_calls)
print("  invalid_tool_calls:", msg.invalid_tool_calls)

# ---------- 4. 解析失败后 ToolNode / tools_condition 的行为 ----------
part("④ 坏 JSON 被丢弃后，图里会发生什么")
from typing import Annotated, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class S(TypedDict):
    messages: Annotated[list, add_messages]


def run_through_tools_node(msgs: list) -> list:
    """把消息喂进一个只有 tools 节点的小图，观察 ToolNode 的真实行为。"""
    node = ToolNode(TOOLS)
    g = StateGraph(S)
    g.add_node("tools", node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile().invoke({"messages": msgs})["messages"]


bad_only = AIMessage(
    content="",
    tool_calls=[],
    invalid_tool_calls=[{
        "name": "search_articles",
        "args": '{"keyword": "微服务"',
        "id": "call_2",
        "type": "invalid_tool_call",
        "error": "Function search_articles arguments ... are not valid JSON",
    }],
)
print("A) 只有坏 JSON 的 AIMessage -> ToolNode 执行后，消息列表：")
print("   ", [f"{m.__class__.__name__}: {str(m.content)[:40]!r}" for m in run_through_tools_node([bad_only])])
print("A) tools_condition 判定（坏调用不算数）:", tools_condition({"messages": [bad_only]}))

good_and_bad = AIMessage(
    content="",
    tool_calls=[{"name": "search_articles", "args": {"keyword": "微服务"}, "id": "call_1", "type": "tool_call"}],
    invalid_tool_calls=bad_only.invalid_tool_calls,
)
print("\nB) 好坏各一条 -> ToolNode 只执行好的：")
for m in run_through_tools_node([good_and_bad]):
    print(f"   -> {m.__class__.__name__}: {str(m.content)[:40]!r}")
