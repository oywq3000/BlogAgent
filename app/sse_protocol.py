# -*- coding: utf-8 -*-
"""SSE 帧序列化：与 Java PythonSseClient.StreamParser 的解析规则逐字节对齐。

帧格式：event: <名>\ndata: <JSON>\n\n（帧以 \n\n 分隔；行以 event:/data: 前缀识别，
Java 侧 trim 后 startsWith 判断，JSON 用 Jackson 解析，中文可原样输出）。
"""
import json


def sse_event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_error(code: int, message: str) -> str:
    return sse_event("error", {"code": code, "message": message})
