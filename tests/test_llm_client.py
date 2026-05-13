import unittest
import sys
import types
from types import SimpleNamespace


def _install_astrbot_stub():
    if "astrbot.api" in sys.modules:
        return
    astrbot_module = types.ModuleType("astrbot")
    api_module = types.ModuleType("astrbot.api")
    api_module.logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    sys.modules.setdefault("astrbot", astrbot_module)
    sys.modules["astrbot.api"] = api_module


_install_astrbot_stub()

from core.summary.llm_client import LLMClient


def _config(**overrides):
    values = {
        "llm_provider": "openai",
        "base_url": "",
        "api_key": "test-key",
        "model": "test-model",
        "api_version": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class LLMClientProviderCompatibilityTests(unittest.TestCase):
    def test_known_openai_compatible_provider_uses_max_tokens_on_first_request(self):
        client = LLMClient(_config(llm_provider="deepseek"))

        request = client.build_http_request(
            {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 321,
            }
        )

        self.assertNotIn("max_completion_tokens", request.json)
        self.assertEqual(request.json["max_tokens"], 321)

    def test_official_openai_provider_keeps_max_completion_tokens(self):
        client = LLMClient(_config(llm_provider="openai"))

        request = client.build_http_request(
            {
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 321,
            }
        )

        self.assertEqual(request.json["max_completion_tokens"], 321)
        self.assertNotIn("max_tokens", request.json)

    def test_missing_base_url_is_reported_once_for_ollama(self):
        client = LLMClient(
            _config(
                llm_provider="ollama",
                base_url="",
                api_key="",
            )
        )

        self.assertEqual(client.missing_fields().count("Base URL"), 1)


if __name__ == "__main__":
    unittest.main()
