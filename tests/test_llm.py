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
