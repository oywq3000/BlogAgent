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

    def remove(self, conversation_id: str, task: asyncio.Task | None = None) -> None:
        """移除登记。

        带 task 参数时做身份检查：仅当登记里的任务正是该 task 才移除，
        防止同会话旧流的收尾误删新流的登记（替换时序竞态）；
        task 为 None 时保持旧行为（无条件 pop），向后兼容。
        """
        if task is not None and self._streams.get(conversation_id) is not task:
            return
        self._streams.pop(conversation_id, None)


registry = StreamRegistry()
