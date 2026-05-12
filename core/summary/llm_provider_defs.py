"""Shared LLM provider defaults and labels."""
from __future__ import annotations


LLM_PROVIDER_OPTIONS = {
    "自定义 OpenAI 兼容": "openai_compatible",
    "OpenAI": "openai",
    "Azure OpenAI": "azure_openai",
    "Anthropic Claude": "anthropic",
    "Google Gemini": "gemini",
    "xAI Grok": "xai",
    "Ollama": "ollama",
    "DeepSeek": "deepseek",
    "Moonshot / Kimi": "moonshot",
    "阿里云百炼 / 通义千问": "qwen",
    "智谱 AI / GLM": "glm",
    "火山引擎方舟 / 豆包": "volcengine",
    "腾讯混元": "hunyuan",
    "百度千帆 / 文心": "qianfan",
    "Mistral AI": "mistral",
    "Groq": "groq",
    "OpenRouter": "openrouter",
    "SiliconFlow": "siliconflow",
    "Together AI": "together",
    "Fireworks AI": "fireworks",
    "DeepInfra": "deepinfra",
}

LLM_PROVIDER_DEFAULTS = {
    "openai_compatible": {
        "base_url": "",
        "api_version": "",
        "requires_api_key": True,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "azure_openai": {
        "base_url": "",
        "api_version": "2024-10-21",
        "requires_api_key": True,
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "api_version": "2023-06-01",
        "requires_api_key": True,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_version": "",
        "requires_api_key": True,
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "api_version": "",
        "requires_api_key": False,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_version": "",
        "requires_api_key": True,
    },
    "volcengine": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_version": "",
        "requires_api_key": True,
    },
    "hunyuan": {
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "qianfan": {
        "base_url": "https://qianfan.baidubce.com/v2",
        "api_version": "",
        "requires_api_key": True,
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_version": "",
        "requires_api_key": True,
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_version": "",
        "requires_api_key": True,
    },
}

LLM_PROVIDER_LABELS = {
    value: key for key, value in LLM_PROVIDER_OPTIONS.items()
}
