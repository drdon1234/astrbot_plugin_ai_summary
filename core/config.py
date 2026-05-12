"""Configuration parsing for the standalone AI summary plugin."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from astrbot.api.star import StarTools

from .summary.prompts import (
    DEFAULT_STYLE_PROMPTS,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_VISUAL_ANALYSIS_PROMPT,
    DEFAULT_VISION_DECISION_PROMPT,
    SUMMARY_STYLE_OPTIONS,
)
from .summary.llm_provider_defs import (
    LLM_PROVIDER_DEFAULTS,
    LLM_PROVIDER_OPTIONS,
)


def _default_cache_dir() -> str:
    """Resolve the plugin-owned data directory for runtime files."""
    return str(StarTools.get_data_dir())


@dataclass
class PermissionConfig:
    admin_id: str = ""
    whitelist_enable: bool = False
    whitelist_user: List[str] = field(default_factory=list)
    whitelist_group: List[str] = field(default_factory=list)
    blacklist_enable: bool = False
    blacklist_user: List[str] = field(default_factory=list)
    blacklist_group: List[str] = field(default_factory=list)

    def is_admin(self, sender_id: Any) -> bool:
        """Return whether the sender matches the configured administrator."""
        sender = str(sender_id or "").strip()
        return bool(self.admin_id and sender == self.admin_id)

    def check(self, is_private: bool, sender_id: Any, group_id: Any) -> bool:
        """Evaluate administrator, whitelist, and blacklist rules for a message."""
        sender = str(sender_id or "").strip()
        group = "" if is_private else str(group_id or "").strip()

        if self.is_admin(sender):
            return True

        allowed = None
        if self.whitelist_enable and sender in self.whitelist_user:
            allowed = True
        elif self.blacklist_enable and sender in self.blacklist_user:
            allowed = False
        elif self.whitelist_enable and group and group in self.whitelist_group:
            allowed = True
        elif self.blacklist_enable and group and group in self.blacklist_group:
            allowed = False

        return (not self.whitelist_enable) if allowed is None else allowed


@dataclass
class AISummaryConfig:
    permission: PermissionConfig = field(default_factory=PermissionConfig)
    llm_provider_source: str = "astrbot"
    astrbot_provider_id: str = ""
    llm_provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-5.5"
    api_version: str = ""
    reply_keyword_trigger: bool = True
    keywords: List[str] = field(
        default_factory=lambda: ["总结视频", "视频总结", "总结一下"]
    )
    style: str = "auto"
    system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT
    prompts: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_STYLE_PROMPTS)
    )
    vision_decision_prompt: str = DEFAULT_VISION_DECISION_PROMPT
    visual_analysis_prompt: str = DEFAULT_VISUAL_ANALYSIS_PROMPT
    max_completion_tokens: int = 1800
    temperature: float = 0.2
    request_timeout_seconds: int = 180
    max_transcript_chars: int = 20000
    vision_max_frames: int = 8
    vision_frame_width: int = 512
    vision_jpeg_quality: int = 4
    vision_image_detail: str = "low"
    vision_batch_size: int = 4
    vision_max_concurrent: int = 2
    vision_request_timeout_seconds: int = 180
    vision_max_chars: int = 8000
    cache_dir: str = ""
    max_video_size_mb: float = 500.0
    download_timeout_seconds: int = 600
    show_error: bool = True
    enable_summary_repair: bool = True
    max_concurrent: int = 1
    asr_max_concurrent: int = 1
    max_videos_per_message: int = 1
    max_videos_per_link: int = 1
    status_message: bool = True
    asr_model: str = (
        "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    )
    vad_model: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    asr_model_dir: str = ""
    download_timeout_minutes: int = 60
    device: str = "cpu"
    batch_size_s: int = 300
    sample_rate: int = 16000
    asr_timeout_seconds: int = 900
    debug_mode: bool = False
    admin_test_keyword: str = "aiping"

    def has_keyword(self, text: str) -> bool:
        """Return whether a message contains any configured summary keyword."""
        return any(kw and kw in (text or "") for kw in self.keywords)

    def should_summarize_reply(self, text: str) -> bool:
        """Return whether a reply message should trigger video summarization."""
        return (
            self.reply_keyword_trigger and
            self.has_keyword(text)
        )

    @property
    def selected_prompt(self) -> str:
        """Return the summary prompt selected by the current style."""
        return (
            self.prompts.get(self.style)
            or DEFAULT_STYLE_PROMPTS["auto"]
        )


def parse_config(config: Dict[str, Any]) -> AISummaryConfig:
    """Normalize AstrBot plugin configuration into a runtime config object."""
    raw = config if isinstance(config, dict) else {}

    permissions = raw.get("permissions", {})
    if not isinstance(permissions, dict):
        permissions = {}
    whitelist = permissions.get("whitelist", {})
    blacklist = permissions.get("blacklist", {})
    if not isinstance(whitelist, dict):
        whitelist = {}
    if not isinstance(blacklist, dict):
        blacklist = {}
    admin_id = str(permissions.get("admin_id", "") or "").strip()
    whitelist_user = _normalize_list(whitelist.get("user", []))
    if admin_id and admin_id not in whitelist_user:
        whitelist_user.append(admin_id)

    permission = PermissionConfig(
        admin_id=admin_id,
        whitelist_enable=bool(whitelist.get("enable", False)),
        whitelist_user=whitelist_user,
        whitelist_group=_normalize_list(whitelist.get("group", [])),
        blacklist_enable=bool(blacklist.get("enable", False)),
        blacklist_user=_normalize_list(blacklist.get("user", [])),
        blacklist_group=_normalize_list(blacklist.get("group", [])),
    )

    llm_raw = _dict(raw.get("llm", {}))
    trigger_raw = _dict(raw.get("trigger", {}))
    basic_quality_raw = _dict(raw.get("基础质量", {}))
    advanced_quality_raw = _dict(raw.get("高级质量", {}))
    basic_summary_raw = _dict(basic_quality_raw.get("summary", {}))
    basic_vision_raw = _dict(basic_quality_raw.get("vision", {}))
    advanced_summary_raw = _dict(advanced_quality_raw.get("summary", {}))
    advanced_vision_raw = _dict(advanced_quality_raw.get("vision", {}))
    advanced_asr_raw = _dict(advanced_quality_raw.get("asr", {}))
    prompt_raw = _dict(raw.get("prompts", {}))
    output_raw = _dict(raw.get("output", {}))
    admin_raw = _dict(raw.get("admin", {}))

    style = _normalize_summary_style(basic_summary_raw.get("style", "自动"))
    provider_source = _normalize_llm_provider_source(
        llm_raw.get("provider_source", "AstrBot 内置提供商")
    )
    astrbot_provider_raw = _dict(llm_raw.get("astrbot_provider", {}))
    custom_provider_raw = _dict(llm_raw.get("custom_provider", {}))

    prompts = {
        "auto": _prompt_or_default(
            prompt_raw.get("auto_prompt"),
            DEFAULT_STYLE_PROMPTS["auto"],
        ),
        "brief": _prompt_or_default(
            prompt_raw.get("brief_prompt"),
            DEFAULT_STYLE_PROMPTS["brief"],
        ),
        "professional": _prompt_or_default(
            prompt_raw.get("professional_prompt"),
            DEFAULT_STYLE_PROMPTS["professional"],
        ),
    }
    system_prompt = _prompt_or_default(
        prompt_raw.get("system_prompt"),
        DEFAULT_SUMMARY_SYSTEM_PROMPT,
    )
    cache_dir = _default_cache_dir()
    model_dir = os.path.join(cache_dir, "models", "funasr")

    keywords = _normalize_list(
        trigger_raw.get("keywords", ["总结视频", "视频总结", "总结一下"])
    )
    if not keywords:
        keywords = ["总结视频", "视频总结", "总结一下"]

    provider = _normalize_llm_provider(
        custom_provider_raw.get("provider", "自定义 OpenAI 兼容")
    )
    provider_defaults = LLM_PROVIDER_DEFAULTS.get(
        provider,
        LLM_PROVIDER_DEFAULTS["openai_compatible"],
    )
    base_url = str(custom_provider_raw.get("base_url", "") or "").strip().rstrip("/")
    if not base_url:
        base_url = str(provider_defaults.get("base_url", "") or "").strip().rstrip("/")
    api_version = str(custom_provider_raw.get("api_version", "") or "").strip()
    if not api_version:
        api_version = str(provider_defaults.get("api_version", "") or "").strip()

    return AISummaryConfig(
        permission=permission,
        llm_provider_source=provider_source,
        astrbot_provider_id=str(
            astrbot_provider_raw.get("provider_id", "") or ""
        ).strip(),
        llm_provider=provider,
        base_url=base_url,
        api_key=str(custom_provider_raw.get("api_key", "") or "").strip(),
        model=str(custom_provider_raw.get("model", "gpt-5.5") or "gpt-5.5").strip(),
        api_version=api_version,
        reply_keyword_trigger=bool(
            trigger_raw.get("reply_keyword_trigger", True)
        ),
        keywords=keywords,
        style=style,
        system_prompt=system_prompt,
        prompts=prompts,
        vision_decision_prompt=_prompt_or_default(
            prompt_raw.get("vision_decision_prompt"),
            DEFAULT_VISION_DECISION_PROMPT,
        ),
        visual_analysis_prompt=_prompt_or_default(
            prompt_raw.get("visual_analysis_prompt"),
            DEFAULT_VISUAL_ANALYSIS_PROMPT,
        ),
        max_completion_tokens=_int_config(
            basic_summary_raw.get("max_completion_tokens"),
            1800,
            minimum=600,
            maximum=4000,
        ),
        temperature=_float_config(
            basic_summary_raw.get("temperature"),
            0.2,
            minimum=0.0,
            maximum=1.0,
        ),
        request_timeout_seconds=_int_config(
            advanced_summary_raw.get("request_timeout_seconds"),
            180,
            minimum=30,
            maximum=600,
        ),
        max_transcript_chars=_int_config(
            basic_summary_raw.get("max_transcript_chars"),
            20000,
            minimum=4000,
            maximum=60000,
        ),
        vision_max_frames=_int_config(
            basic_vision_raw.get("max_frames"),
            8,
            minimum=0,
            maximum=32,
        ),
        vision_frame_width=_vision_frame_width(
            basic_vision_raw.get("frame_size"),
            512,
        ),
        vision_jpeg_quality=_int_config(
            advanced_vision_raw.get("jpeg_quality"),
            4,
            minimum=2,
            maximum=31,
        ),
        vision_image_detail=_choice_config(
            basic_vision_raw.get("image_detail"),
            "low",
            {"auto", "low", "high"},
        ),
        vision_batch_size=_int_config(
            basic_vision_raw.get("batch_size"),
            4,
            minimum=1,
            maximum=8,
        ),
        vision_max_concurrent=_int_config(
            advanced_vision_raw.get("max_concurrent"),
            2,
            minimum=1,
            maximum=4,
        ),
        vision_request_timeout_seconds=_int_config(
            advanced_vision_raw.get("request_timeout_seconds"),
            180,
            minimum=30,
            maximum=600,
        ),
        vision_max_chars=_int_config(
            advanced_vision_raw.get("max_chars"),
            8000,
            minimum=1000,
            maximum=20000,
        ),
        cache_dir=cache_dir,
        max_video_size_mb=500.0,
        download_timeout_seconds=600,
        max_videos_per_message=1,
        show_error=bool(output_raw.get("show_error", True)),
        enable_summary_repair=bool(output_raw.get("enable_summary_repair", True)),
        status_message=bool(output_raw.get("status_message", True)),
        max_concurrent=_int_config(
            advanced_summary_raw.get("max_concurrent"),
            1,
            minimum=1,
            maximum=4,
        ),
        asr_max_concurrent=_int_config(
            advanced_asr_raw.get("max_concurrent"),
            1,
            minimum=1,
            maximum=2,
        ),
        asr_model=(
            "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        ),
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        asr_model_dir=model_dir,
        download_timeout_minutes=60,
        device=_choice_config(
            advanced_asr_raw.get("device"),
            "cpu",
            {"cpu", "cuda"},
        ),
        batch_size_s=_int_config(
            advanced_asr_raw.get("batch_size_s"),
            300,
            minimum=30,
            maximum=600,
        ),
        sample_rate=_int_config(
            advanced_asr_raw.get("sample_rate"),
            16000,
            minimum=8000,
            maximum=48000,
        ),
        asr_timeout_seconds=_int_config(
            advanced_asr_raw.get("asr_timeout_seconds"),
            900,
            minimum=60,
            maximum=3600,
        ),
        debug_mode=bool(admin_raw.get("debug_mode", False)),
        admin_test_keyword=str(
            admin_raw.get("test_keyword", "aiping") or "aiping"
        ).strip() or "aiping",
    )


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_list(values: Any) -> List[str]:
    """Return a deduplicated list of non-empty string values."""
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _prompt_or_default(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text if text else default


def _normalize_summary_style(value: Any) -> str:
    """Map display labels or raw keys to a supported summary style."""
    text = str(value or "").strip() or "自动"
    style = SUMMARY_STYLE_OPTIONS.get(text, text)
    return style if style in DEFAULT_STYLE_PROMPTS else "auto"


def _normalize_llm_provider_source(value: Any) -> str:
    """Normalize the configured LLM provider source to an internal key."""
    text = str(value or "").strip() or "AstrBot 内置提供商"
    mapping = {
        "AstrBot 内置提供商": "astrbot",
        "astrbot": "astrbot",
        "AstrBot": "astrbot",
        "插件自定义提供商": "custom",
        "custom": "custom",
    }
    return mapping.get(text, "astrbot")


def _int_config(
    value: Any,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(minimum, min(maximum, parsed))


def _float_config(
    value: Any,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return max(minimum, min(maximum, parsed))


def _choice_config(value: Any, default: str, choices: set[str]) -> str:
    text = str(value or "").strip()
    return text if text in choices else default


def _vision_frame_width(value: Any, default: int) -> int:
    text = str(value or "").strip().lower()
    if text in {"原始尺寸", "original", "origin", "none", "不压缩", "不缩放"}:
        return 0
    if text.endswith("px"):
        text = text[:-2].strip()
    try:
        width = int(text)
    except (TypeError, ValueError):
        width = int(default)
    allowed = {512, 768, 1024}
    return width if width in allowed else int(default)


def _normalize_llm_provider(value: Any) -> str:
    """Resolve display labels and aliases to a supported LLM provider key."""
    text = str(value or "").strip()
    if not text:
        return "openai_compatible"
    if text in LLM_PROVIDER_OPTIONS.values():
        return text
    if text in LLM_PROVIDER_OPTIONS:
        return LLM_PROVIDER_OPTIONS[text]
    lowered = text.lower()
    for label, key in LLM_PROVIDER_OPTIONS.items():
        if lowered == label.lower() or lowered == key.lower():
            return key
    return "openai_compatible"

