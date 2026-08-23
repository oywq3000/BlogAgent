# -*- coding: utf-8 -*-
"""活动流登记表：conversationId -> 生成任务。

Java 侧已有同会话 409 守卫，这里是 Python 侧兜底：
同会话新流到来时取消旧任务（不再生成），stop 即取消。
"""
import asyncio


class StreamRegistry:

    def __init__(self) -> None:
        self._streams: dict[str, asyncio.Task] = {}

    def register(self, conversation_id: str, task: asyncio.Task) -> None:
        old = self._streams.get(conversation_id)
        if old is not None and not old.done():
            old.cancel()
        self._streams[conversation_id] = task

    def get(self, conversation_id: str) -> asyncio.Task | None:
        return self._streams.get(conversation_id)

    def cancel(self, conversation_id: str) -> bool:
        task = self._streams.get(conversation_id)
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    def remove(self, conversation_id: str) -> None:
        self._streams.pop(conversation_id, None)


registry = StreamRegistry()
