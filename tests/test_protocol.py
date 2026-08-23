from app.sse_protocol import sse_error, sse_event


def test_token_frame_exact_bytes():
    # 与 Java StreamParser 解析规则逐字节对齐：event: 前缀、data: JSON、\n\n 分隔
    frame = sse_event("token", {"content": "你好"})
    assert frame == 'event: token\ndata: {"content": "你好"}\n\n'


def test_thinking_frame():
    frame = sse_event("thinking", {"content": "让我想想"})
    assert frame.startswith("event: thinking\n")
    assert '{"content": "让我想想"}' in frame
    assert frame.endswith("\n\n")


def test_done_frame():
    frame = sse_event("done", {"messageId": "py-abc123"})
    assert frame == 'event: done\ndata: {"messageId": "py-abc123"}\n\n'


def test_error_frame():
    frame = sse_error(429, "请求过于频繁，请稍后再试")
    assert frame == 'event: error\ndata: {"code": 429, "message": "请求过于频繁，请稍后再试"}\n\n'


def test_non_ascii_not_escaped():
    # 与 agent_stub.py 一致：ensure_ascii=False，中文原样输出
    assert "\\u" not in sse_event("token", {"content": "中文测试"})


def test_empty_content():
    assert sse_event("token", {"content": ""}) == 'event: token\ndata: {"content": ""}\n\n'
