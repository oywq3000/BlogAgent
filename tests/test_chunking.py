# -*- coding: utf-8 -*-
"""切块单测：段落合并、代码块独立、超长段窗口重叠、边界防御（count_tokens 用 len 注入）。"""
from app.chunking import split_content


def test_empty_content():
    assert split_content("", count_tokens=len) == []
    assert split_content("   \n  ", count_tokens=len) == []


def test_short_paragraphs_merge_into_one_chunk():
    text = "第一段。\n\n第二段。\n\n第三段。"
    chunks = split_content(text, count_tokens=len, max_tokens=100)
    assert len(chunks) == 1
    assert "第一段" in chunks[0] and "第三段" in chunks[0]


def test_paragraph_exceeding_max_starts_new_chunk():
    text = "A" * 60 + "\n\n" + "B" * 60
    chunks = split_content(text, count_tokens=len, max_tokens=50)
    assert len(chunks) == 2
    assert chunks[0] == "A" * 60  # 单段超限不切（段落完整），仅分段


def count_lines(text: str) -> int:
    """测试用 token 计数：按行数（每行 1 token）。"""
    return text.count("\n") + 1


def test_code_block_is_not_split():
    text = "介绍段落。\n```python\ncode_a = 1\ncode_b = 2\n```\n结尾段落。"
    chunks = split_content(text, count_tokens=len, max_tokens=50)
    assert len(chunks) == 1  # 短内容合并为一个 chunk
    assert "```python\ncode_a = 1\ncode_b = 2\n```" in chunks[0]  # 代码块完整未被切开


def test_long_block_windows_overlap():
    # 每行 1 token（按行计数），max_tokens=10，overlap=3 → 窗口 10 行、重叠 3 token
    text = "\n".join(f"L{i}" for i in range(30))
    chunks = split_content(text, count_tokens=count_lines, max_tokens=10, overlap_tokens=3)
    assert len(chunks) >= 4
    assert chunks[0].splitlines()[0] == "L0"
    assert chunks[1].splitlines()[0] == "L7"  # 10 - 3 = 7，下一窗口重叠 3 token
    assert chunks[0].splitlines()[-1] == "L9"


def test_long_block_zero_overlap_no_infinite_loop():
    text = "\n".join(f"L{i}" for i in range(30))
    chunks = split_content(text, count_tokens=count_lines, max_tokens=10, overlap_tokens=0)
    assert len(chunks) >= 3  # 至少前进一行，不陷入死循环
