# -*- coding: utf-8 -*-
"""文章切块：段落优先 + token 兜底 + 重叠（规格 2026-09-04 §4.2）。纯函数，不依赖模型。"""
from collections.abc import Callable

_FENCE = "```"


def split_content(
    content: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 256,
    overlap_tokens: int = 32,
) -> list[str]:
    """段落优先切块：代码块独立段、段落合并累积、超长段 token 窗口重叠切。"""
    if not content or not content.strip():
        return []
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for block in _split_blocks(content):
        block_tokens = count_tokens(block)
        if buf and buf_tokens + block_tokens > max_tokens:
            chunks.append("\n\n".join(buf))  # 段落间保留空行
            buf, buf_tokens = [], 0
        if block_tokens > max_tokens:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, buf_tokens = [], 0
            chunks.extend(_split_long_block(block, count_tokens, max_tokens, overlap_tokens))
        else:
            buf.append(block)
            buf_tokens += block_tokens
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _split_blocks(content: str) -> list[str]:
    """Markdown 分段：代码块整体独立段，其余按空行分隔。"""
    blocks: list[str] = []
    buf: list[str] = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(_FENCE):
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            # 收集整个代码块（含首尾围栏）为一个 block
            code_lines = [lines[i]]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(_FENCE):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):  # 闭合围栏
                code_lines.append(lines[i])
                i += 1
            blocks.append("\n".join(code_lines))
        elif stripped:
            buf.append(lines[i])
            i += 1
        else:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            i += 1
    if buf:
        blocks.append("\n".join(buf))
    return blocks


def _split_long_block(
    block: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """超长段按 token 窗口切，窗口间重叠 overlap_tokens（按行前缀和定位重叠点）。"""
    lines = block.splitlines() or [""]
    line_tokens = [max(count_tokens(line), 1) for line in lines]  # 每行至少 1 token，防空行 0
    prefix = [0]
    for t in line_tokens:
        prefix.append(prefix[-1] + t)
    chunks: list[str] = []
    start = 0
    n = len(lines)
    while start < n:
        if line_tokens[start] > max_tokens:
            # 行内兜底：超长单行按字符窗口切（中文 1 字≈1 token 近似），保证不触发模型 512 截断
            chunks.extend(_split_line_by_chars(lines[start], max_tokens, overlap_tokens))
            start += 1
            continue
        end = start
        while end < n and prefix[end + 1] - prefix[start] <= max_tokens:  # <= 让窗口填满到 max_tokens
            end += 1
        end = max(end, start + 1)  # 至少一行，防死循环
        chunks.append("\n".join(lines[start:end]))
        if end >= n:
            break
        target = prefix[end] - overlap_tokens  # 下一窗口起点：窗口 [start,end) 尾部绝对位置是 prefix[end]，重叠 overlap_tokens
        start = end
        while start > 0 and prefix[start] > target:
            start -= 1
        if start >= end:  # 防御（overlap=0 等）：至少前进一行
            start = end - 1
    return chunks


def _split_line_by_chars(line: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """行内字符窗口切：width=max_tokens 字符、step=max_tokens-overlap_tokens（至少 1）。"""
    width = max(max_tokens, 1)
    step = max(max_tokens - overlap_tokens, 1)
    return [line[i : i + width] for i in range(0, len(line), step)]
