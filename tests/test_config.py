import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_defaults():
    s = Settings(_env_file=None)  # 不读 .env，测纯默认值
    assert s.agent_port == 8001
    assert s.model_default == "deepseek-chat"
    assert s.thinking_mode == "hybrid"
    assert s.article_content_max_chars == 4000
    assert s.search_page_size == 5
    assert s.mock_llm is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("ES_URL", "http://es.test:9200")
    monkeypatch.setenv("THINKING_MODE", "reasoner")
    monkeypatch.setenv("MOCK_LLM", "1")
    s = Settings(_env_file=None)
    assert s.es_url == "http://es.test:9200"
    assert s.thinking_mode == "reasoner"
    assert s.mock_llm is True


def test_invalid_thinking_mode_rejected():
    with pytest.raises(ValidationError):
        Settings(thinking_mode="magic", _env_file=None)


def test_require_api_key_mock_mode_ok():
    s = Settings(mock_llm=True, deepseek_api_key=None, _env_file=None)
    s.require_api_key()  # 不抛异常


def test_require_api_key_missing_raises():
    s = Settings(mock_llm=False, deepseek_api_key=None, _env_file=None)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        s.require_api_key()


def test_get_settings_cached(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("AGENT_PORT", "9999")
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    assert s1.agent_port == 9999
    get_settings.cache_clear()
