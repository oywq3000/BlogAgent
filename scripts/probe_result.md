# DeepSeek thinking 参数探测结果

- 日期: 2026-08-23
- 环境: conda ai-agent（Python 3.12），langchain-deepseek==1.1.0，langchain-core==1.5.5
- 探测脚本: `scripts/probe_deepseek.py`
- 判定: **三项全 PASS → 策略 A（hybrid）可行**

## 结论与证据

| # | 探测项 | 结果 | 证据 |
|---|--------|------|------|
| 1 | `deepseek-chat` + `extra_body={"thinking": {"type": "enabled"}}` 是否被接受 | PASS | 请求未报错；流式响应中返回了 `reasoning_content` 字段（思考内容约 100 字），证明参数真实生效而非被静默忽略 |
| 2 | 流式 chunk 的 `additional_kwargs["reasoning_content"]` 是否有思考内容 | PASS | 两次运行分别收集到 104 字 / 98 字思考内容；`content` 正常输出，回答 `'2'` |
| 3 | 开启 thinking 后 tool_calls 是否仍可用 | PASS | `bind_tools` 后调用返回 1 个 tool_call（`get_weather`），thinking 与工具调用可同时启用 |

运行输出摘要（第二次运行，UTF-8）：

```
DeepSeek base_url=默认
[PASS] thinking 参数被接受: 请求未报错
[PASS] reasoning_content 流式输出: 思考内容 98 字
[PASS] content 正常输出: 回答: '2'
[PASS] thinking 模式下工具调用: tool_calls 数量: 1
```

## 关键参数形态（Task 6 的 build_chat_model 以此为准）

- 启用思考：构造 `ChatDeepSeek` 时传 `extra_body={"thinking": {"type": "enabled"}}`（仅 `deepseek-chat` 需要；`deepseek-reasoner` 默认思考，不传此参数）。
- 读取思考内容：流式 chunk 的 `chunk.additional_kwargs["reasoning_content"]`，与 `content` 并存（分块返回，需自行拼接）。
- 非流式时思考内容位于 `response.additional_kwargs["reasoning_content"]`（本次未单独验证，按 DeepSeek 官方格式）。
- base_url：`.env` 未设 `DEEPSEEK_BASE_URL`，走 langchain-deepseek 默认值 `https://api.deepseek.com`，工作正常。

## 对 Task 6 的影响

- `THINKING_MODE=hybrid`（默认）可直接落地：`MODEL_DEFAULT=deepseek-chat` + thinking 参数启用，工具调用不受影响。
- 无需按规格 §3 回退到 `deepseek-reasoner` 策略 B；但 `MODEL_ALLOWLIST` 仍保留 `deepseek-reasoner` 作为备选。
- 流式输出层需透传 `reasoning_content`（如存入 `message.additional_kwargs`），供前端展示思考过程。
