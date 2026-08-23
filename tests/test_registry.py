import asyncio

import pytest

from app.stream_registry import StreamRegistry


async def _sleeper():
    await asyncio.sleep(60)


@pytest.mark.asyncio
async def test_register_and_get():
    reg = StreamRegistry()
    task = asyncio.create_task(_sleeper())
    reg.register("c1", task)
    assert reg.get("c1") is task
    reg.remove("c1")
    assert reg.get("c1") is None


@pytest.mark.asyncio
async def test_register_duplicate_cancels_old():
    reg = StreamRegistry()
    old = asyncio.create_task(_sleeper())
    new = asyncio.create_task(_sleeper())
    reg.register("c1", old)
    reg.register("c1", new)
    await asyncio.sleep(0)  # 让取消生效
    assert old.cancelled()
    assert reg.get("c1") is new


@pytest.mark.asyncio
async def test_register_duplicate_ignores_finished_old():
    reg = StreamRegistry()
    old = asyncio.create_task(asyncio.sleep(0))
    await old
    new = asyncio.create_task(_sleeper())
    reg.register("c1", old)  # old 已完成，不应误取消
    reg.register("c1", new)
    assert not old.cancelled()
    assert reg.get("c1") is new


@pytest.mark.asyncio
async def test_cancel_active_returns_true():
    reg = StreamRegistry()
    task = asyncio.create_task(_sleeper())
    reg.register("c1", task)
    assert reg.cancel("c1") is True
    await asyncio.sleep(0)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_cancel_missing_returns_false():
    reg = StreamRegistry()
    assert reg.cancel("nobody") is False


@pytest.mark.asyncio
async def test_remove_idempotent():
    reg = StreamRegistry()
    reg.remove("nobody")  # 不抛异常
    task = asyncio.create_task(_sleeper())
    reg.register("c1", task)
    reg.remove("c1")
    reg.remove("c1")
