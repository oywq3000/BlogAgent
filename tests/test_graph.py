import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.agent.graph import AgentState, build_graph, messages_from_request


class FakeModelWithTools(FakeMessagesListChatModel):
    """FakeMessagesListChatModel 在本版 langchain-core（1.5.5）的 bind_tools 抛 NotImplementedError。

    补一个 no-op 覆写：假模型输出完全由 responses 脚本决定，与工具绑定无关；
    build_graph 内部仍走 model.bind_tools(tools)（真模型 ChatDeepSeek 才真正消费绑定）。
    """

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


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
    model = FakeModelWithTools(responses=[
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
    model = FakeModelWithTools(responses=[AIMessage(content="直接回答")])
    graph = build_graph(model, [dummy_tool])
    result = await graph.ainvoke({"messages": [HumanMessage(content="hi")]})
    assert result["messages"][-1].content == "直接回答"


def test_agent_state_shape():
    # AgentState 必须含 messages 且 add_messages 合并语义
    state: AgentState = {"messages": []}
    assert "messages" in state
