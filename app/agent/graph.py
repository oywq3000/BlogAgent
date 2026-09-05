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
from langgraph.graph.state import CompiledStateGraph
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
        if role == "user" and content:
            # content 为空（None/""）的历史消息跳过，避免空 HumanMessage 干扰模型
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
        # 其余 role 忽略（Java 只会传 user/assistant）
    msgs.append(HumanMessage(content=user_message))
    return msgs


def build_graph(model: BaseChatModel, tools: list) -> CompiledStateGraph:
    
    # 模型绑定工具
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition) # 条件边：根据模型输出判断是否调用工具
    graph.add_edge("tools", "agent")
    return graph.compile()


@lru_cache(maxsize=8)
def get_graph(deep_thinking: bool, model_field: str | None) -> CompiledStateGraph:
    settings = get_settings()
    model = build_chat_model(deep_thinking, model_field, settings)
    tools = build_tools(settings)
    return build_graph(model, tools)
