# -*- coding: utf-8 -*-
"""手动全量重建文章向量：重建 article_chunks 索引（删旧建新 + 切块 + 嵌入写回）。

用法:
  /d/tool1/anancoda/envs/ai-agent/python.exe scripts/rebuild_vectors.py
"""
import asyncio
import logging

from app.config import get_settings
from app.vector_sync import rebuild_vectors

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


async def main() -> None:
    result = await rebuild_vectors(get_settings())
    print("重建完成:", result)


if __name__ == "__main__":
    asyncio.run(main())
