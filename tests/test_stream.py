import asyncio
import json

import pytest

from app.main import map_model_error

PAYLOAD = {
    "conversationId": "conv_1",
    "userId": "u1",
    "message": "你好",
    "history": [{"role": "user", "content": "以前问过的问题"}],
    "deepThinking": False,
    "model": "default",
}


def parse_frames(text: str) -> list[tuple[str, dict]]:
    """按 Java StreamParser 规则解析 SSE 文本为 (event, data) 列表。"""
    frames = []
    for block in text.split("\n\n"):
        event, data = "", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if event:
            frames.append((event, json.loads(data)))
    return frames


async def read_all(client, payload: dict) -> list[tuple[str, dict]]:
    async with client.stream("POST", "/chat/stream", json=payload) as resp:
        text = (await resp.aread()).decode("utf-8")  # httpx aread() 返回 bytes
    return parse_frames(text)


@pytest.mark.asyncio
async def test_normal_stream_token_then_done(client):
    frames = await read_all(client, PAYLOAD)
    events = [e for e, _ in frames]
    assert "thinking" not in events  # deepThinking=false 无思考
    assert events[0] == "token"
    assert events[-1] == "done"
    content = "".join(d["content"] for e, d in frames if e == "token")
    assert content == "【MOCK】收到：你好"
    done_id = frames[-1][1]["messageId"]
    assert done_id.startswith("py-")


@pytest.mark.asyncio
async def test_deep_thinking_emits_thinking_first(client):
    payload = {**PAYLOAD, "deepThinking": True}
    frames = await read_all(client, payload)
    events = [e for e, _ in frames]
    assert events[0] == "thinking"
    assert events[1] == "token"
    reasoning = "".join(d["content"] for e, d in frames if e == "thinking")
    assert "模拟思考" in reasoning
    assert events[-1] == "done"


@pytest.mark.asyncio
async def test_empty_message_returns_error_400(client):
    frames = await read_all(client, {**PAYLOAD, "message": "   "})
    assert frames == [("error", {"code": 400, "message": "参数不完整"})]


@pytest.mark.asyncio
async def test_missing_conversation_id_returns_error_400(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "conversationId"}
    frames = await read_all(client, payload)
    assert frames[0][0] == "error"
    assert frames[0][1]["code"] == 400


@pytest.mark.asyncio
async def test_stop_cancels_stream_without_done(client):
    frames: list[tuple[str, dict]] = []

    async def reader():
        async with client.stream("POST", "/chat/stream", json=PAYLOAD) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line[len("event:"):].strip()
                    frames.append((event, {}))

    task = asyncio.create_task(reader())
    # 等首帧到达后触发 stop（Mock 每 chunk 间隔 0.05s，流长 >0.5s）
    for _ in range(50):
        if frames:
            break
        await asyncio.sleep(0.05)
    assert frames, "首帧未到达，Mock 流未启动"

    resp = await client.post("/chat/stop", json={"conversationId": "conv_1"})
    assert resp.json() == {"ok": True, "conversationId": "conv_1"}

    await asyncio.wait_for(task, timeout=10)
    events = [e for e, _ in frames]
    assert "done" not in events  # stop 后静默收尾，无 done


@pytest.mark.asyncio
async def test_new_stream_replaces_old_same_conversation(client):
    frames1: list[str] = []

    async def reader1():
        async with client.stream("POST", "/chat/stream", json=PAYLOAD) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    frames1.append(line)

    task1 = asyncio.create_task(reader1())
    for _ in range(50):
        if frames1:
            break
        await asyncio.sleep(0.05)
    assert frames1

    # 同会话开新流：旧流被取消（静默结束），新流正常完成
    frames2 = await read_all(client, PAYLOAD)
    await asyncio.wait_for(task1, timeout=10)
    assert frames2[-1][0] == "done"


def test_map_model_error_429():
    code, msg = map_model_error(Exception("429 status code: Too Many Requests"))
    assert code == 429
    assert "频繁" in msg


def test_map_model_error_timeout():
    code, msg = map_model_error(Exception("connection timeout after 60s"))
    assert code == 504
    assert "超时" in msg


def test_map_model_error_unknown():
    code, msg = map_model_error(Exception("乱七八糟"))
    assert code == 500
