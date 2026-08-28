# -*- coding: utf-8 -*-
"""moderation 模块单测：解析容错 + 截断 + MOCK 触发词三态。"""
from app.config import Settings
from app.moderation import _parse_verdict, _truncate, moderate_content


def _settings(mock: bool = True) -> Settings:
    return Settings(mock_llm=mock, moderation_content_max_chars=100)


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
