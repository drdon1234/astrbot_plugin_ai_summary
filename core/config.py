"""Configuration parsing for the standalone AI summary plugin."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from astrbot.api.star import StarTools

from .summary.llm_provider_defs import (
    LLM_PROVIDER_DEFAULTS,
    LLM_PROVIDER_OPTIONS,
)


PLUGIN_NAME = "astrbot_plugin_ai_summary"
DEFAULT_AUTO_KEYWORDS = ["总结一下", "总结视频", "自动总结"]
DEFAULT_BRIEF_KEYWORDS = ["简略总结", "简单总结"]
DEFAULT_PROFESSIONAL_KEYWORDS = ["专业总结", "详细总结"]
DEFAULT_ORAL_KEYWORDS = ["口语概述", "口语总结"]
DEFAULT_NEWS_KEYWORDS = ["新闻摘要", "事件摘要", "新闻总结"]
DEFAULT_NOTE_KEYWORDS = ["笔记总结", "专业总结", "详细总结"]
DEFAULT_QA_EXIT_COMMANDS = ["结束", "退出"]
DEFAULT_QA_CLEAR_COMMANDS = ["清理", "清空"]
DEFAULT_IMAGE_FONT_FAMILY = "noto_sans"


def _default_cache_dir() -> str:
    """Resolve the plugin-owned data directory for runtime files."""
    return str(StarTools.get_data_dir(PLUGIN_NAME))


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
    enable_persona: bool = False
    persona_id: str = ""
    llm_provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-5.5"
    api_version: str = ""
    reply_keyword_trigger: bool = True
    auto_keywords: List[str] = field(default_factory=lambda: list(DEFAULT_AUTO_KEYWORDS))
    brief_keywords: List[str] = field(default_factory=lambda: list(DEFAULT_BRIEF_KEYWORDS))
    professional_keywords: List[str] = field(
        default_factory=lambda: list(DEFAULT_PROFESSIONAL_KEYWORDS)
    )
    oral_keywords: List[str] = field(default_factory=lambda: list(DEFAULT_ORAL_KEYWORDS))
    news_keywords: List[str] = field(default_factory=lambda: list(DEFAULT_NEWS_KEYWORDS))
    note_keywords: List[str] = field(default_factory=lambda: list(DEFAULT_NOTE_KEYWORDS))
    qa_enabled: bool = True
    qa_record_ttl_minutes: int = 30
    qa_history_turns: int = 5
    qa_exit_commands: List[str] = field(
        default_factory=lambda: list(DEFAULT_QA_EXIT_COMMANDS)
    )
    qa_clear_commands: List[str] = field(
        default_factory=lambda: list(DEFAULT_QA_CLEAR_COMMANDS)
    )
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
    summary_format: str = "text"
    send_format: str = "text"
    qa_answer_format: str = "text"
    qa_send_format: str = "text"
    image_style: str = "fresh"
    image_font_family: str = DEFAULT_IMAGE_FONT_FAMILY
    image_font_size: int = 25
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
        return bool(self.summary_style_for_text(text))

    def should_summarize_reply(self, text: str) -> bool:
        """Return whether a reply message should trigger video summarization."""
        return (
            self.reply_keyword_trigger and
            self.has_keyword(text)
        )

    def summary_style_for_text(self, text: str) -> str:
        """Return the requested summary style from trigger commands."""
        value = str(text or "")
        matches: List[tuple[int, int, str]] = []
        for style, keywords in self.summary_keywords_by_style().items():
            for keyword in keywords:
                keyword_text = str(keyword or "").strip()
                if not keyword_text:
                    continue
                index = value.find(keyword_text)
                if index >= 0:
                    matches.append((index, -len(keyword_text), style))
        if not matches:
            return ""
        matches.sort()
        return matches[0][2]

    def summary_trigger_keywords(self) -> List[str]:
        """Return all summary trigger commands, longest first."""
        keywords: List[str] = []
        for values in self.summary_keywords_by_style().values():
            keywords.extend(values)
        return sorted(keywords, key=len, reverse=True)

    def summary_keywords_by_style(self) -> Dict[str, List[str]]:
        """Return trigger commands grouped by requested summary style."""
        return {
            "auto": self.auto_keywords,
            "note": self.note_keywords,
            "news": self.news_keywords,
            "oral": self.oral_keywords,
        }

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
    output_raw = _dict(raw.get("output", {}))
    qa_raw = _dict(raw.get("qa", {}))
    admin_raw = _dict(raw.get("admin", {}))

    provider_source = _normalize_llm_provider_source(
        llm_raw.get("provider_source", "AstrBot 内置提供商")
    )
    astrbot_provider_raw = _dict(llm_raw.get("astrbot_provider", {}))
    custom_provider_raw = _dict(llm_raw.get("custom_provider", {}))
    persona_raw_value = llm_raw.get("persona", {})
    has_persona_object = "persona" in llm_raw and isinstance(persona_raw_value, dict)
    persona_raw = _dict(persona_raw_value)
    legacy_persona_id = str(
        llm_raw.get("persona_id", "")
        or astrbot_provider_raw.get("persona_id", "")
        or ""
    ).strip()
    persona_id = str(
        persona_raw.get("persona_id", "") if has_persona_object else legacy_persona_id
    ).strip()
    enable_persona = bool(
        persona_raw.get(
            "enable",
            bool(legacy_persona_id) if not has_persona_object else False,
        )
    )

    cache_dir = _default_cache_dir()
    model_dir = os.path.join(cache_dir, "models", "funasr")

    auto_keywords = _normalize_trigger_keywords(
        trigger_raw.get("auto_keywords"),
        DEFAULT_AUTO_KEYWORDS,
    )
    brief_keywords = _normalize_trigger_keywords(
        trigger_raw.get("brief_keywords"),
        DEFAULT_BRIEF_KEYWORDS,
    )
    professional_keywords = _normalize_trigger_keywords(
        trigger_raw.get("professional_keywords"),
        DEFAULT_PROFESSIONAL_KEYWORDS,
    )
    oral_keywords = _merge_trigger_keywords(
        _normalize_trigger_keywords(
            trigger_raw.get("oral_keywords"),
            DEFAULT_ORAL_KEYWORDS,
        ),
        brief_keywords,
    )
    news_keywords = _normalize_trigger_keywords(
        trigger_raw.get("news_keywords"),
        DEFAULT_NEWS_KEYWORDS,
    )
    note_keywords = _merge_trigger_keywords(
        _normalize_trigger_keywords(
            trigger_raw.get("note_keywords"),
            DEFAULT_NOTE_KEYWORDS,
        ),
        professional_keywords,
    )

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
        enable_persona=enable_persona,
        persona_id=persona_id,
        llm_provider=provider,
        base_url=base_url,
        api_key=str(custom_provider_raw.get("api_key", "") or "").strip(),
        model=str(custom_provider_raw.get("model", "gpt-5.5") or "gpt-5.5").strip(),
        api_version=api_version,
        reply_keyword_trigger=bool(
            trigger_raw.get("reply_keyword_trigger", True)
        ),
        auto_keywords=auto_keywords,
        brief_keywords=brief_keywords,
        professional_keywords=professional_keywords,
        oral_keywords=oral_keywords,
        news_keywords=news_keywords,
        note_keywords=note_keywords,
        qa_enabled=bool(qa_raw.get("enable", True)),
        qa_record_ttl_minutes=_int_config(
            qa_raw.get("record_ttl_minutes"),
            30,
            minimum=0,
            maximum=1440,
        ),
        qa_history_turns=_int_config(
            qa_raw.get("history_turns"),
            5,
            minimum=0,
            maximum=20,
        ),
        qa_exit_commands=_normalize_trigger_keywords(
            qa_raw.get("exit_commands"),
            DEFAULT_QA_EXIT_COMMANDS,
        ),
        qa_clear_commands=_normalize_trigger_keywords(
            qa_raw.get("clear_commands"),
            DEFAULT_QA_CLEAR_COMMANDS,
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
        summary_format=_normalize_summary_format(
            output_raw.get("summary_format", "纯文本")
        ),
        send_format=_normalize_send_format(
            output_raw.get("send_format", "文本")
        ),
        qa_answer_format=_normalize_summary_format(
            output_raw.get(
                "qa_answer_format",
                output_raw.get("qa_format", "纯文本"),
            )
        ),
        qa_send_format=_normalize_send_format(
            output_raw.get(
                "qa_send_format",
                output_raw.get("qa_delivery_format", "文本"),
            )
        ),
        image_style=_normalize_image_style(
            output_raw.get("image_style", "清新")
        ),
        image_font_family=_normalize_image_font_family(
            output_raw.get("image_font_family", "默认黑体")
        ),
        image_font_size=_int_config(
            output_raw.get("image_font_size"),
            25,
            minimum=16,
            maximum=48,
        ),
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


def _normalize_trigger_keywords(
    values: Any,
    defaults: List[str],
) -> List[str]:
    """Return configured trigger commands or defaults."""
    normalized = _normalize_list(values)
    return normalized or list(defaults)


def _merge_trigger_keywords(*groups: List[str]) -> List[str]:
    """Merge trigger command groups without losing configured legacy commands."""
    merged: List[str] = []
    seen = set()
    for group in groups:
        for value in group:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return merged


def _normalize_summary_format(value: Any) -> str:
    """Normalize summary content format to text or markdown."""
    text = str(value or "").strip()
    lowered = text.casefold()
    mapping = {
        "text": "text",
        "plain": "text",
        "plain_text": "text",
        "纯文本": "text",
        "文本": "text",
        "markdown": "markdown",
        "md": "markdown",
    }
    return mapping.get(text, mapping.get(lowered, "text"))


def _normalize_send_format(value: Any) -> str:
    """Normalize final message delivery format to text or image."""
    text = str(value or "").strip()
    lowered = text.casefold()
    mapping = {
        "text": "text",
        "plain": "text",
        "纯文本": "text",
        "文本": "text",
        "文字": "text",
        "image": "image",
        "img": "image",
        "图片": "image",
        "图像": "image",
    }
    return mapping.get(text, mapping.get(lowered, "text"))


def _normalize_image_style(value: Any) -> str:
    """Normalize image rendering style to a stable internal key."""
    text = str(value or "").strip()
    lowered = text.casefold()
    mapping = {
        "清新": "fresh",
        "清新便签": "fresh",
        "便签": "fresh",
        "粉色便签": "fresh",
        "bilinote": "fresh",
        "note": "fresh",
        "fresh_note": "fresh",
        "科技感": "tech",
        "科技": "tech",
        "tech": "tech",
        "technology": "tech",
        "专业严肃": "serious",
        "严肃": "serious",
        "专业": "serious",
        "serious": "serious",
        "professional": "serious",
        "card": "card",
        "温和卡片": "card",
        "卡片": "card",
        "soft_card": "card",
        "default": "card",
    }
    return mapping.get(text, mapping.get(lowered, "fresh"))


def _normalize_image_font_family(value: Any) -> str:
    """Normalize image font family to a bundled font key."""
    text = str(value or "").strip()
    lowered = text.casefold()
    mapping = {
        "默认黑体": "noto_sans",
        "黑体": "noto_sans",
        "思源黑体": "noto_sans",
        "noto_sans": "noto_sans",
        "noto sans": "noto_sans",
        "default": "noto_sans",
        "专业宋体": "noto_serif",
        "宋体": "noto_serif",
        "思源宋体": "noto_serif",
        "noto_serif": "noto_serif",
        "noto serif": "noto_serif",
        "serif": "noto_serif",
        "清新文楷": "lxgw_wenkai",
        "文楷": "lxgw_wenkai",
        "霞鹜文楷": "lxgw_wenkai",
        "lxgw_wenkai": "lxgw_wenkai",
        "lxgw wenkai": "lxgw_wenkai",
        "wenkai": "lxgw_wenkai",
        "标题手札": "zcool_xiaowei",
        "站酷小薇": "zcool_xiaowei",
        "zcool_xiaowei": "zcool_xiaowei",
        "zcool xiaowei": "zcool_xiaowei",
        "xiaowei": "zcool_xiaowei",
        "科技窄体": "zcool_qingke",
        "站酷庆科黄油体": "zcool_qingke",
        "zcool_qingke": "zcool_qingke",
        "zcool qingke": "zcool_qingke",
        "qingke": "zcool_qingke",
    }
    return mapping.get(text, mapping.get(lowered, DEFAULT_IMAGE_FONT_FAMILY))


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

