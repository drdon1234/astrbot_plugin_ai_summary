"""Provider-aware LLM adapter for AI summary requests."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import aiohttp


@dataclass(frozen=True)
class LLMProviderDefinition:
    key: str
    protocol: str
    default_base_url: str
    requires_api_key: bool
    default_api_version: str = ""


@dataclass
class ProviderHttpRequest:
    url: str
    headers: Dict[str, str]
    json: Dict[str, Any]


PROVIDER_DEFINITIONS: Dict[str, LLMProviderDefinition] = {
    "openai_compatible": LLMProviderDefinition(
        key="openai_compatible",
        protocol="openai",
        default_base_url="",
        requires_api_key=True,
    ),
    "openai": LLMProviderDefinition(
        key="openai",
        protocol="openai",
        default_base_url="https://api.openai.com/v1",
        requires_api_key=True,
    ),
    "azure_openai": LLMProviderDefinition(
        key="azure_openai",
        protocol="azure_openai",
        default_base_url="",
        requires_api_key=True,
        default_api_version="2024-10-21",
    ),
    "anthropic": LLMProviderDefinition(
        key="anthropic",
        protocol="anthropic",
        default_base_url="https://api.anthropic.com",
        requires_api_key=True,
        default_api_version="2023-06-01",
    ),
    "gemini": LLMProviderDefinition(
        key="gemini",
        protocol="gemini",
        default_base_url="https://generativelanguage.googleapis.com/v1beta",
        requires_api_key=True,
    ),
    "xai": LLMProviderDefinition(
        key="xai",
        protocol="openai",
        default_base_url="https://api.x.ai/v1",
        requires_api_key=True,
    ),
    "ollama": LLMProviderDefinition(
        key="ollama",
        protocol="ollama",
        default_base_url="http://localhost:11434",
        requires_api_key=False,
    ),
    "deepseek": LLMProviderDefinition(
        key="deepseek",
        protocol="openai",
        default_base_url="https://api.deepseek.com/v1",
        requires_api_key=True,
    ),
    "moonshot": LLMProviderDefinition(
        key="moonshot",
        protocol="openai",
        default_base_url="https://api.moonshot.cn/v1",
        requires_api_key=True,
    ),
    "qwen": LLMProviderDefinition(
        key="qwen",
        protocol="openai",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        requires_api_key=True,
    ),
    "glm": LLMProviderDefinition(
        key="glm",
        protocol="openai",
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        requires_api_key=True,
    ),
    "volcengine": LLMProviderDefinition(
        key="volcengine",
        protocol="openai",
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        requires_api_key=True,
    ),
    "hunyuan": LLMProviderDefinition(
        key="hunyuan",
        protocol="openai",
        default_base_url="https://api.hunyuan.cloud.tencent.com/v1",
        requires_api_key=True,
    ),
    "qianfan": LLMProviderDefinition(
        key="qianfan",
        protocol="openai",
        default_base_url="https://qianfan.baidubce.com/v2",
        requires_api_key=True,
    ),
    "mistral": LLMProviderDefinition(
        key="mistral",
        protocol="openai",
        default_base_url="https://api.mistral.ai/v1",
        requires_api_key=True,
    ),
    "groq": LLMProviderDefinition(
        key="groq",
        protocol="openai",
        default_base_url="https://api.groq.com/openai/v1",
        requires_api_key=True,
    ),
    "openrouter": LLMProviderDefinition(
        key="openrouter",
        protocol="openai",
        default_base_url="https://openrouter.ai/api/v1",
        requires_api_key=True,
    ),
    "siliconflow": LLMProviderDefinition(
        key="siliconflow",
        protocol="openai",
        default_base_url="https://api.siliconflow.cn/v1",
        requires_api_key=True,
    ),
    "together": LLMProviderDefinition(
        key="together",
        protocol="openai",
        default_base_url="https://api.together.xyz/v1",
        requires_api_key=True,
    ),
    "fireworks": LLMProviderDefinition(
        key="fireworks",
        protocol="openai",
        default_base_url="https://api.fireworks.ai/inference/v1",
        requires_api_key=True,
    ),
    "deepinfra": LLMProviderDefinition(
        key="deepinfra",
        protocol="openai",
        default_base_url="https://api.deepinfra.com/v1/openai",
        requires_api_key=True,
    ),
}


class LLMClient:
    """Build provider-specific requests and extract text responses."""

    def __init__(self, config: Any):
        self.config = config

    def is_configured(self) -> bool:
        """Return whether the selected provider has all required fields."""
        missing = self.missing_fields()
        return not missing

    def missing_fields(self) -> List[str]:
        """List user-facing configuration fields required by the active provider."""
        provider = self._provider_definition()
        missing: List[str] = []
        if not self._model():
            missing.append("模型")
        if provider.requires_api_key and not self._api_key():
            missing.append("API Key")
        if provider.protocol in {"openai", "azure_openai", "anthropic", "ollama"}:
            if provider.key != "ollama" and not self._base_url():
                missing.append("Base URL")
            if provider.key == "ollama" and not self._base_url():
                missing.append("Base URL")
        elif provider.protocol == "gemini" and not self._base_url():
            missing.append("Base URL")
        return missing

    def build_http_request(
        self,
        payload: Dict[str, Any],
        *,
        use_max_tokens: bool = False,
        drop_temperature: bool = False,
    ) -> ProviderHttpRequest:
        """Build a provider-specific HTTP request from a chat-style payload."""
        provider = self._provider_definition()
        model = self._model()
        if not model:
            raise RuntimeError("未配置 AI 总结模型")
        if provider.requires_api_key and not self._api_key():
            raise RuntimeError("未配置 AI 总结 API Key")

        builder = {
            "openai": self._build_openai_request,
            "azure_openai": self._build_azure_request,
            "anthropic": self._build_anthropic_request,
            "gemini": self._build_gemini_request,
            "ollama": self._build_ollama_request,
        }.get(provider.protocol)
        if not builder:
            raise RuntimeError(f"不支持的 LLM 协议: {provider.protocol}")
        return builder(
            payload,
            use_max_tokens=use_max_tokens,
            drop_temperature=drop_temperature,
        )

    async def complete(
        self,
        payload: Dict[str, Any],
        *,
        timeout_seconds: int,
    ) -> str:
        """Send a completion request and retry with provider compatibility tweaks."""
        timeout = aiohttp.ClientTimeout(total=max(10, int(timeout_seconds)))
        use_max_tokens = False
        drop_temperature = False
        working_payload = copy.deepcopy(payload)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(3):
                request = self.build_http_request(
                    working_payload,
                    use_max_tokens=use_max_tokens,
                    drop_temperature=drop_temperature,
                )
                try:
                    async with session.post(
                        request.url,
                        json=request.json,
                        headers=request.headers,
                    ) as response:
                        body = await response.text()
                        if response.status >= 400:
                            raise RuntimeError(
                                f"HTTP {response.status}: {body}"
                            )
                        return self.extract_content(json.loads(body))
                except RuntimeError as exc:
                    message = str(exc)
                    if attempt == 0 and self._should_retry_with_max_tokens(message):
                        use_max_tokens = True
                        continue
                    if attempt <= 1 and self._should_drop_temperature(message):
                        drop_temperature = True
                        continue
                    raise

        raise RuntimeError("LLM 请求失败")

    def extract_content(self, response: Dict[str, Any]) -> str:
        """Extract plain text from the active provider's response shape."""
        provider = self._provider_definition()
        parser = {
            "openai": self._extract_openai_content,
            "azure_openai": self._extract_openai_content,
            "anthropic": self._extract_anthropic_content,
            "gemini": self._extract_gemini_content,
            "ollama": self._extract_ollama_content,
        }.get(provider.protocol)
        if not parser:
            raise RuntimeError(f"不支持的 LLM 协议: {provider.protocol}")
        return parser(response)

    def _provider_definition(self) -> LLMProviderDefinition:
        provider_key = str(getattr(self.config, "llm_provider", "") or "").strip()
        return PROVIDER_DEFINITIONS.get(
            provider_key,
            PROVIDER_DEFINITIONS["openai_compatible"],
        )

    def _model(self) -> str:
        return str(getattr(self.config, "model", "") or "").strip()

    def _api_key(self) -> str:
        return str(getattr(self.config, "api_key", "") or "").strip()

    def _base_url(self) -> str:
        return str(getattr(self.config, "base_url", "") or "").strip().rstrip("/")

    def _api_version(self) -> str:
        value = str(getattr(self.config, "api_version", "") or "").strip()
        if value:
            return value
        provider = self._provider_definition()
        return provider.default_api_version

    def _build_openai_request(
        self,
        payload: Dict[str, Any],
        *,
        use_max_tokens: bool,
        drop_temperature: bool,
    ) -> ProviderHttpRequest:
        """Build an OpenAI-compatible chat completions request."""
        body = copy.deepcopy(payload)
        if use_max_tokens:
            max_tokens = body.pop(
                "max_completion_tokens",
                body.get("max_tokens", 0),
            )
            body["max_tokens"] = max_tokens
        if drop_temperature:
            body.pop("temperature", None)
        url = self._join_chat_completions_url(self._base_url() or self._provider_definition().default_base_url)
        headers = self._bearer_headers(self._api_key())
        return ProviderHttpRequest(url=url, headers=headers, json=body)

    def _build_azure_request(
        self,
        payload: Dict[str, Any],
        *,
        use_max_tokens: bool,
        drop_temperature: bool,
    ) -> ProviderHttpRequest:
        """Build an Azure OpenAI deployment chat completions request."""
        body = copy.deepcopy(payload)
        body.pop("model", None)
        if use_max_tokens:
            max_tokens = body.pop(
                "max_completion_tokens",
                body.get("max_tokens", 0),
            )
            body["max_tokens"] = max_tokens
        if drop_temperature:
            body.pop("temperature", None)
        base_url = self._base_url()
        if not base_url:
            raise RuntimeError("未配置 AI 总结 Base URL")
        if base_url.rstrip("/").endswith("/chat/completions"):
            url = base_url.rstrip("/")
        else:
            model = quote(self._model(), safe="-_.~")
            url = (
                base_url.rstrip("/")
                + f"/openai/deployments/{model}/chat/completions"
            )
        api_version = self._api_version()
        if api_version:
            url = self._set_query_param(url, "api-version", api_version)
        headers = {
            "api-key": self._api_key(),
            "Content-Type": "application/json",
        }
        return ProviderHttpRequest(url=url, headers=headers, json=body)

    def _build_anthropic_request(
        self,
        payload: Dict[str, Any],
        *,
        use_max_tokens: bool,
        drop_temperature: bool,
    ) -> ProviderHttpRequest:
        """Build an Anthropic messages request from chat-style input."""
        system, messages = self._split_system_and_messages(payload.get("messages") or [])
        body: Dict[str, Any] = {
            "model": self._model(),
            "messages": messages,
            "max_tokens": self._extract_max_tokens(payload),
        }
        if system:
            body["system"] = system
        if not drop_temperature and payload.get("temperature") is not None:
            body["temperature"] = payload["temperature"]
        url = self._join_path(
            self._base_url() or self._provider_definition().default_base_url,
            "/v1/messages",
        )
        headers = {
            "x-api-key": self._api_key(),
            "anthropic-version": self._api_version() or "2023-06-01",
            "Content-Type": "application/json",
        }
        return ProviderHttpRequest(url=url, headers=headers, json=body)

    def _build_gemini_request(
        self,
        payload: Dict[str, Any],
        *,
        use_max_tokens: bool,
        drop_temperature: bool,
    ) -> ProviderHttpRequest:
        """Build a Gemini generateContent request from chat-style input."""
        system, messages = self._split_system_and_messages(payload.get("messages") or [])
        contents = self._gemini_contents(messages)
        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self._extract_max_tokens(payload),
            },
        }
        if system:
            body["systemInstruction"] = {
                "parts": [{"text": system}],
            }
        if not drop_temperature and payload.get("temperature") is not None:
            body["generationConfig"]["temperature"] = payload["temperature"]
        base_url = self._base_url() or self._provider_definition().default_base_url
        if base_url.rstrip("/").endswith(":generateContent"):
            url = base_url.rstrip("/")
        else:
            normalized = base_url.rstrip("/")
            if not re.search(r"/v\d(?:beta)?$", normalized):
                normalized = normalized + "/v1beta"
            url = normalized + f"/models/{quote(self._model(), safe='-_.~')}:generateContent"
        if self._api_key():
            url = self._set_query_param(url, "key", self._api_key())
        headers = {"Content-Type": "application/json"}
        return ProviderHttpRequest(url=url, headers=headers, json=body)

    def _build_ollama_request(
        self,
        payload: Dict[str, Any],
        *,
        use_max_tokens: bool,
        drop_temperature: bool,
    ) -> ProviderHttpRequest:
        """Build an Ollama chat request with optional image payloads."""
        messages = self._ollama_messages(payload.get("messages") or [])
        body: Dict[str, Any] = {
            "model": self._model(),
            "messages": messages,
            "stream": False,
        }
        options: Dict[str, Any] = {}
        if not drop_temperature and payload.get("temperature") is not None:
            options["temperature"] = payload["temperature"]
        max_tokens = self._extract_max_tokens(payload)
        if max_tokens:
            options["num_predict"] = max_tokens
        if options:
            body["options"] = options
        url = self._join_path(
            self._base_url() or self._provider_definition().default_base_url,
            "/api/chat",
        )
        headers = {"Content-Type": "application/json"}
        if self._api_key():
            headers["Authorization"] = f"Bearer {self._api_key()}"
        return ProviderHttpRequest(url=url, headers=headers, json=body)

    def _split_system_and_messages(
        self,
        messages: Iterable[Any],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Separate system text from chat turns for providers that need it."""
        system_parts: List[str] = []
        conversation: List[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "") or "").strip() or "user"
            content = message.get("content")
            if role == "system":
                system_parts.append(self._content_to_text(content))
                continue
            conversation.append({
                "role": "assistant" if role == "assistant" else "user",
                "content": self._content_to_blocks(content),
            })
        return "\n".join(part for part in system_parts if part).strip(), conversation

    def _content_to_blocks(self, content: Any) -> List[Dict[str, Any]]:
        """Convert OpenAI-style text and image content into Anthropic blocks."""
        if isinstance(content, list):
            blocks: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    text = str(part).strip()
                    if text:
                        blocks.append({"type": "text", "text": text})
                    continue
                part_type = str(part.get("type", "") or "").strip()
                if part_type == "text":
                    text = str(part.get("text", "") or "").strip()
                    if text:
                        blocks.append({"type": "text", "text": text})
                elif part_type == "image_url":
                    image = self._image_source_block(part.get("image_url"))
                    if image:
                        blocks.append(image)
            return blocks
        text = str(content or "").strip()
        return [{"type": "text", "text": text}] if text else []

    def _image_source_block(self, value: Any) -> Dict[str, Any]:
        url = ""
        if isinstance(value, dict):
            url = str(value.get("url", "") or "").strip()
        elif isinstance(value, str):
            url = value.strip()
        if not url:
            return {}
        mime_type, data = self._decode_data_url(url)
        if not mime_type or not data:
            return {}
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": data,
            },
        }

    def _gemini_contents(self, messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert chat messages into Gemini content parts."""
        contents: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "") or "").strip()
            content = message.get("content")
            parts: List[Dict[str, Any]] = []
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        text = str(part).strip()
                        if text:
                            parts.append({"text": text})
                        continue
                    part_type = str(part.get("type", "") or "").strip()
                    if part_type == "text":
                        text = str(part.get("text", "") or "").strip()
                        if text:
                            parts.append({"text": text})
                    elif part_type == "image_url":
                        image = self._gemini_inline_data(part.get("image_url"))
                        if image:
                            parts.append(image)
                    elif part_type == "image":
                        image = self._gemini_inline_data_from_source(part.get("source"))
                        if image:
                            parts.append(image)
            else:
                text = str(content or "").strip()
                if text:
                    parts.append({"text": text})
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": parts,
                }
            )
        return contents

    def _gemini_inline_data(self, value: Any) -> Dict[str, Any]:
        url = ""
        if isinstance(value, dict):
            url = str(value.get("url", "") or "").strip()
        elif isinstance(value, str):
            url = value.strip()
        if not url:
            return {}
        mime_type, data = self._decode_data_url(url)
        if not mime_type or not data:
            return {}
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": data,
            }
        }

    def _gemini_inline_data_from_source(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        if str(value.get("type", "") or "").strip() != "base64":
            return {}
        mime_type = str(value.get("media_type", "") or "").strip()
        data = str(value.get("data", "") or "").strip()
        if not mime_type or not data:
            return {}
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": data,
            }
        }

    def _ollama_messages(self, messages: Iterable[Any]) -> List[Dict[str, Any]]:
        """Convert chat messages into Ollama message objects."""
        converted: List[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "") or "").strip() or "user"
            content = message.get("content")
            message_body: Dict[str, Any] = {
                "role": "assistant" if role == "assistant" else role,
                "content": self._content_to_text(content),
            }
            images = self._ollama_images(content)
            if images:
                message_body["images"] = images
            converted.append(message_body)
        return converted

    def _ollama_images(self, content: Any) -> List[str]:
        images: List[str] = []
        if not isinstance(content, list):
            return images
        for part in content:
            if not isinstance(part, dict):
                continue
            if str(part.get("type", "") or "").strip() != "image_url":
                continue
            url = ""
            image_value = part.get("image_url")
            if isinstance(image_value, dict):
                url = str(image_value.get("url", "") or "").strip()
            elif isinstance(image_value, str):
                url = image_value.strip()
            if not url:
                continue
            _, data = self._decode_data_url(url)
            if data:
                images.append(data)
        return images

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if str(part.get("type", "") or "").strip() == "text":
                        text = str(part.get("text", "") or "").strip()
                        if text:
                            parts.append(text)
                else:
                    text = str(part).strip()
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content or "").strip()

    def _extract_max_tokens(self, payload: Dict[str, Any]) -> int:
        value = payload.get("max_completion_tokens")
        if value is None:
            value = payload.get("max_tokens")
        try:
            return max(1, int(value or 0))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _decode_data_url(url: str) -> Tuple[str, str]:
        text = str(url or "").strip()
        if not text.startswith("data:"):
            return "", ""
        match = re.match(
            r"^data:([^;,]+)?(?:;charset=[^;,]+)?;base64,(.+)$",
            text,
            re.I | re.S,
        )
        if not match:
            return "", ""
        mime_type = match.group(1) or "application/octet-stream"
        data = match.group(2).strip()
        return mime_type, data

    def _join_chat_completions_url(self, base_url: str) -> str:
        base = str(base_url or "").strip().rstrip("/")
        if not base:
            raise RuntimeError("未配置 AI 总结 Base URL")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    @staticmethod
    def _join_path(base_url: str, suffix: str) -> str:
        base = str(base_url or "").strip().rstrip("/")
        if not base:
            raise RuntimeError("未配置 AI 总结 Base URL")
        normalized_suffix = "/" + suffix.lstrip("/")
        if base.endswith(normalized_suffix):
            return base
        return base + normalized_suffix

    @staticmethod
    def _set_query_param(url: str, key: str, value: str) -> str:
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query[key] = value
        return urlunparse(parsed._replace(query=urlencode(query)))

    @staticmethod
    def _bearer_headers(api_key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _should_retry_with_max_tokens(message: str) -> bool:
        lowered = str(message or "").lower()
        return (
            "max_completion_tokens" in lowered
            or "max_tokens" in lowered
            or "unrecognized" in lowered
        )

    @staticmethod
    def _should_drop_temperature(message: str) -> bool:
        return "temperature" in str(message or "").lower()

    @staticmethod
    def _extract_openai_content(response: Dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("LLM 响应中没有 choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ).strip()
        else:
            text = ""
        if not text:
            raise RuntimeError("LLM 返回空总结")
        return text

    @staticmethod
    def _extract_anthropic_content(response: Dict[str, Any]) -> str:
        blocks = response.get("content") or []
        parts: List[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if str(block.get("type", "") or "").strip() != "text":
                continue
            text = str(block.get("text", "") or "").strip()
            if text:
                parts.append(text)
        text = "\n".join(parts).strip()
        if not text:
            raise RuntimeError("LLM 返回空总结")
        return text

    @staticmethod
    def _extract_gemini_content(response: Dict[str, Any]) -> str:
        candidates = response.get("candidates") or []
        if not candidates:
            raise RuntimeError("LLM 响应中没有 candidates")
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts: List[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = str(part.get("text", "") or "").strip()
            if text:
                texts.append(text)
        text = "".join(texts).strip()
        if not text:
            raise RuntimeError("LLM 返回空总结")
        return text

    @staticmethod
    def _extract_ollama_content(response: Dict[str, Any]) -> str:
        message = response.get("message") or {}
        content = message.get("content")
        text = str(content or "").strip()
        if not text:
            raise RuntimeError("LLM 返回空总结")
        return text
