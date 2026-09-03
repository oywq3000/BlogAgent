# -*- coding: utf-8 -*-
"""逐节点调试脚本：观察 LangGraph 图每个节点的执行过程。

用法：
    cd /g/agentWorkplace/BlogAgent
    /d/tool1/anancoda/envs/ai-agent/python.exe scripts/probe_graph_nodes.py

用"剧本化假模型"控制模型行为：第一轮声明调用工具，第二轮直接回答，
从而完整重放 agent -> tools -> agent -> END 的循环（不碰真实 DeepSeek / ES）。

演示三种观察方式（由粗到细）：
  方式一  stream_mode="updates"   每个节点跑完后立刻产出一帧 (节点名, 该节点的状态更新)
  方式二  stream_mode="messages"  逐 token 产出，附元数据（哪个节点、第几步）
  方式三  astream_events          完整事件流（节点开始/结束、token、工具开始/结束）
"""
import asyncio
import sys
from pathlib import Path

# 允许从项目根目录直接运行 scripts/ 下的脚本（否则 import app 会失败）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.agent.graph import build_graph


class ScriptedModel(FakeMessagesListChatModel):
    """剧本化假模型：输出完全由 responses 决定；bind_tools 是 no-op（与测试同一先例）。"""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@tool
def search_articles(keyword: str) -> str:
    """按关键词搜索博客已发布文章。"""
    return f'[{{"id": "a1", "title": "《{keyword}入门》"}}]'


def build_demo_graph():
    """第一轮调工具，第二轮直接回答 -> 完整走一遍 agent->tools->agent->END。"""
    model = ScriptedModel(responses=[
        AIMessage(
            content="",
            tool_calls=[{
                "name": "search_articles",
                "args": {"keyword": "微服务"},
                "id": "call_1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="博客里有《微服务入门》这篇文章。"),
    ])
    return build_graph(model, [search_articles])


def initial_state() -> dict:
    return {
        "messages": [
            SystemMessage(content="你是博客助手，查博客内容必须用工具。"),
            HumanMessage(content="博客里有微服务文章吗？"),
        ],
    }


async def demo_updates(graph):
    """方式一：stream_mode='updates' —— 每个节点完成后产出 (节点名, 更新)。"""
    print("=" * 66)
    print("方式一：stream_mode='updates' —— 逐节点观察执行与产出")
    print("=" * 66)
    async for update in graph.astream(initial_state(), stream_mode="updates"):
        for node_name, node_update in update.items():
            print(f"\n>>> 节点 [{node_name}] 执行完毕，本轮对状态的更新：")
            for key, msgs in node_update.items():
                for m in msgs:
                    desc = f"{m.__class__.__name__}: {str(m.content)[:50]!r}"
                    if getattr(m, "tool_calls", None):
                        desc += f"  tool_calls={m.tool_calls}"
                    if isinstance(m, ToolMessage):
                        desc += f"  [工具返回值]"
                    print(f"    - {desc}")


async def demo_messages(graph):
    """方式二：stream_mode='messages' —— 逐 token 产出 + 元数据定位来源节点。"""
    print()
    print("=" * 66)
    print("方式二：stream_mode='messages' —— 每个 token + 它来自哪个节点/第几步")
    print("=" * 66)
    async for chunk, meta in graph.astream(initial_state(), stream_mode="messages"):
        node = meta.get("langgraph_node", "?")
        step = meta.get("langgraph_step", "?")
        text = str(chunk.content)
        reasoning = chunk.additional_kwargs.get("reasoning_content")
        if reasoning:
            print(f"  [step {step} | {node}] thinking: {reasoning[:40]!r}")
        if text:
            print(f"  [step {step} | {node}] token: {text!r}")


async def demo_events(graph):
    """方式三：astream_events(v2) —— 完整事件流：节点开始/结束、token、工具调用。"""
    print()
    print("=" * 66)
    print("方式三：astream_events —— 完整事件流（最细粒度）")
    print("=" * 66)
    async for ev in graph.astream_events(initial_state(), version="v2"):
        name = ev.get("name", "")
        event = ev.get("event")
        # 只打印关键事件，过滤掉内部碎事件
        if event in ("on_chain_start", "on_chain_end") and name in ("agent", "tools", "LangGraph"):
            print(f"  ● {event:15s} {name}")
        elif event == "on_llm_new_token":
            chunk = ev["data"]["chunk"]
            reasoning = chunk.additional_kwargs.get("reasoning_content")
            if reasoning:
                print(f"    · token(thinking): {reasoning[:30]!r}")
            elif chunk.content:
                print(f"    · token: {str(chunk.content)[:30]!r}")
        elif event in ("on_tool_start", "on_tool_end"):
            print(f"  ● {event:15s} {name}")


async def main():
    graph = build_demo_graph()
    print("图结构（规格 §3）：START -> agent -(要工具? )-> tools -> agent ... -> END")
    await demo_updates(graph)
    await demo_messages(graph)
    await demo_events(graph)


if __name__ == "__main__":
    asyncio.run(main())
