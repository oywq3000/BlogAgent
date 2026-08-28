# -*- coding: utf-8 -*-
"""应用配置：pydantic-settings 读环境变量 / .env。"""
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"

    agent_host: str = "0.0.0.0"
    agent_port: int = 8001

    es_url: str = "http://192.168.200.130:9200"
    es_username: str = ""
    es_password: str = ""

    model_default: str = "deepseek-v4-flash"
    model_allowlist: list[str] = ["deepseek-v4-flash", "deepseek-v4-pro"]
    thinking_mode: str = "hybrid"  # hybrid=deepseek-chat+thinking 参数 / reasoner=deepseek-reasoner

    mock_llm: bool = False

    article_content_max_chars: int = 4000
    moderation_content_max_chars: int = 8000
    search_page_size: int = 5

    @field_validator("thinking_mode")
    @classmethod
    def _check_thinking_mode(cls, v: str) -> str:
        if v not in ("hybrid", "reasoner"):
            raise ValueError("THINKING_MODE 必须是 hybrid 或 reasoner")
        return v

    def require_api_key(self) -> None:
        """启动 fail-fast：非 mock 模式下缺 key 直接报错。"""
        if not self.mock_llm and not self.deepseek_api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY（.env），或设置 MOCK_LLM=1 使用联调假模式")


@lru_cache
def get_settings() -> Settings:
    return Settings()
