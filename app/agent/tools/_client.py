# -*- coding: utf-8 -*-
"""工具共享基础设施：ES 客户端构建与 highlight 清理。"""
import re

import httpx

from app.config import Settings

_EM_TAG = re.compile(r"</?em>")


def build_es_client(settings: Settings) -> httpx.AsyncClient:
    auth = (settings.es_username, settings.es_password) if settings.es_username else None
    return httpx.AsyncClient(base_url=settings.es_url, auth=auth, timeout=10.0)


def _clean_highlight(text: str) -> str:
    return _EM_TAG.sub("", text)
