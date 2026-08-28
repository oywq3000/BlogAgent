# -*- coding: utf-8 -*-
"""审核端点处理器单测（直接调函数，不起 HTTP 服务；依赖 MOCK_LLM=1 环境）。"""
from app.main import moderate_article


def test_missing_params_422():
    resp = moderate_article({"articleId": "a1", "title": "", "content": "正文"})
    assert resp.status_code == 422


def test_empty_article_id_accepted():
    resp = moderate_article({"articleId": "", "title": "标题", "content": "正常内容"})
    assert resp == {"verdict": "approve", "reason": "【MOCK】联调放行"}


def test_mock_approve():
    resp = moderate_article({"articleId": "a1", "title": "标题", "summary": "", "content": "正常内容"})
    assert resp == {"verdict": "approve", "reason": "【MOCK】联调放行"}


def test_mock_reject():
    resp = moderate_article({"articleId": "a1", "title": "标题", "summary": "", "content": "违规内容"})
    assert resp == {"verdict": "reject", "reason": "【MOCK】命中触发词：违规"}
