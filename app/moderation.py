# -*- coding: utf-8 -*-
"""文章 AI 审核：三态判定（approve/reject/manual），供 POST /moderate/article 使用。

调用方：Java article-service 发布文章时同步调用，判定结果直接决定文章状态。
健壮性约定：模型输出解析失败一律回退 manual（转人工），绝不放行也绝不误杀。
"""
import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from app.config import Settings

MODERATION_PROMPT = """你是博客文章审核员。根据以下规则审核文章，判断是否违规。

违规规则（命中任意一条判 reject）：
1. 违法内容：宣扬违法犯罪、赌博、毒品、枪支等违法活动
2. 政治敏感：攻击国家制度、领导人，传播政治谣言，破坏社会稳定
3. 色情低俗：色情描写、性暗示、低俗挑逗内容
4. 广告引流：纯广告软文、推广联系方式、诱导付费或加群
5. 人身攻击：辱骂、威胁、诽谤、歧视他人
6. 垃圾内容：无意义灌水、乱码、明显测试文本

判定标准：
- 明确命中违规规则 → verdict=reject
- 完全正常、不涉及任何违规规则 → verdict=approve
- 无法确定是否违规（有歧义）→ verdict=manual

只输出一行 JSON，不要输出任何其他内容：
{"verdict":"approve|reject|manual","reason":"简短说明，不超过100字"}"""


def _build_model(settings: Settings) -> BaseChatModel:
    """审核专用模型：默认模型 + 非流式 + 60s 超时。

    不复用 llm.py 的 build_chat_model——那是聊天/思考流专用（streaming=True），
    审核只需一次完整 invoke。
    """
    return ChatDeepSeek(
        model=settings.model_default,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        streaming=False,
        timeout=60,
    )


def _parse_verdict(text: str) -> tuple[str, str]:
    """模型输出 -> (verdict, reason)。解析失败一律回退 manual。"""
    if not text:
        return "manual", "AI 未返回结果"
    match = re.search(r"\{[^{}]*\}", text, re.S)
    if not match:
        return "manual", "AI 返回格式异常"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "manual", "AI 返回格式异常"
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in ("approve", "reject", "manual"):
        return "manual", "AI 判定结果未知"
    reason = str(data.get("reason") or "").strip()[:500]
    return verdict, reason


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def moderate_content(
    article_id: str, title: str, summary: str, content: str, settings: Settings
) -> tuple[str, str]:
    """审核文章 -> (verdict, reason)。

    MOCK_LLM=1 联调模式：正文含"违规"→reject、含"歧义"→manual、否则 approve，
    无 API key 也能全链路验证三种结果。
    """
    full_text = _truncate(f"{title}\n{summary}\n{content}", settings.moderation_content_max_chars)
    if settings.mock_llm:
        if "违规" in full_text:
            return "reject", "【MOCK】命中触发词：违规"
        if "歧义" in full_text:
            return "manual", "【MOCK】命中触发词：歧义"
        return "approve", "【MOCK】联调放行"
    model = _build_model(settings)
    resp = model.invoke(
        [SystemMessage(content=MODERATION_PROMPT), HumanMessage(content=full_text)]
    )
    return _parse_verdict(str(resp.content))
