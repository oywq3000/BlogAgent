# -*- coding: utf-8 -*-
"""moderation 模块单测：结构化输出 + 解析容错 + 截断 + MOCK 触发词三态。"""
import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from app.config import Settings
from app.moderation import ModerationVerdict, _parse_verdict, _truncate, moderate_content


def _settings(mock: bool = True) -> Settings:
    return Settings(mock_llm=mock, moderation_content_max_chars=100)


class _FakeStructured:
    """伪结构化模型：with_structured_output 返回自身，invoke 返回预设输出。

    out 形如 with_structured_output(include_raw=True) 的返回：
    {"raw": BaseMessage, "parsed": ModerationVerdict|None, "parsing_error": 异常|None}
    """

    def __init__(self, out: dict):
        self._out = out

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        return self._out


def test_parse_normal():
    verdict, reason = _parse_verdict('{"verdict":"reject","reason":"含有政治敏感内容"}')
    assert verdict == "reject"
    assert reason == "含有政治敏感内容"


def test_parse_with_extra_text():
    # 模型有时会在 JSON 外多输出说明文字，容错提取第一个 {...}
    verdict, _ = _parse_verdict('判定结果：{"verdict":"approve","reason":"正常"}\n')
    assert verdict == "approve"


def test_parse_bad_json():
    verdict, reason = _parse_verdict("这不是JSON")
    assert verdict == "manual"
    assert "异常" in reason


def test_parse_empty():
    verdict, _ = _parse_verdict("")
    assert verdict == "manual"


def test_parse_unknown_verdict():
    verdict, _ = _parse_verdict('{"verdict":"maybe","reason":"x"}')
    assert verdict == "manual"


def test_truncate():
    assert _truncate("1234567890", 5) == "12345"
    assert _truncate("abc", 5) == "abc"


def test_mock_reject():
    verdict, reason = moderate_content("a1", "标题", "", "这是一篇违规文章", _settings())
    assert verdict == "reject"
    assert "违规" in reason


def test_mock_manual():
    verdict, _ = moderate_content("a1", "标题", "", "内容有些歧义", _settings())
    assert verdict == "manual"


def test_mock_approve():
    verdict, _ = moderate_content("a1", "标题", "", "正常的技术文章", _settings())
    assert verdict == "approve"


def test_truncation_applied_before_mock_judge():
    # 触发词在截断边界之外 → 不命中，防误判也验证截断确实生效
    long_text = "x" * 200 + "违规"
    verdict, _ = moderate_content("a1", "t", "", long_text, _settings())
    assert verdict == "approve"


def test_structured_success(monkeypatch):
    # 结构化输出直接命中 → 返回 pydantic 校验过的 (verdict, reason)
    out = {
        "raw": AIMessage(content='{"verdict":"reject","reason":"政治敏感"}'),
        "parsed": ModerationVerdict(verdict="reject", reason="政治敏感"),
        "parsing_error": None,
    }
    monkeypatch.setattr("app.moderation._build_model", lambda s: _FakeStructured(out))
    verdict, reason = moderate_content("a1", "标题", "", "正文", _settings(mock=False))
    assert verdict == "reject"
    assert reason == "政治敏感"


def test_structured_fallback_raw(monkeypatch):
    # 结构化解析失败（parsed=None）→ 用 raw 文本走 _parse_verdict 兜底解析
    out = {
        "raw": AIMessage(content='判定结果：{"verdict":"approve","reason":"正常"}\n'),
        "parsed": None,
        "parsing_error": Exception("parse fail"),
    }
    monkeypatch.setattr("app.moderation._build_model", lambda s: _FakeStructured(out))
    verdict, _ = moderate_content("a1", "标题", "", "正文", _settings(mock=False))
    assert verdict == "approve"


def test_structured_both_fail_manual(monkeypatch):
    # 结构化失败 + raw 也不可解析 → 回退 manual（绝不放行）
    out = {
        "raw": AIMessage(content="这不是JSON"),
        "parsed": None,
        "parsing_error": Exception("parse fail"),
    }
    monkeypatch.setattr("app.moderation._build_model", lambda s: _FakeStructured(out))
    verdict, reason = moderate_content("a1", "标题", "", "正文", _settings(mock=False))
    assert verdict == "manual"
    assert "异常" in reason


def test_verdict_schema_validation():
    # pydantic Literal 白名单：非法判定在解析层直接被拒
    with pytest.raises(ValidationError):
        ModerationVerdict(verdict="maybe")
        
        

