"""Runtime helpers for summary-based QA."""
from __future__ import annotations

import re
from typing import Any


QA_RECORD_MARKER_LABEL = "问答ID"
QA_RECORD_ID_RE = re.compile(
    r"(?:问答ID|总结ID|记录ID|QA[-_ ]?ID)\s*[:：]\s*"
    r"([0-9]{14}-[0-9a-fA-F]{8})"
)


def qa_scope_id(is_private: bool, sender_id: Any, group_id: Any) -> str:
    """Return the knowledge scope for private or group QA records."""
    if is_private:
        value = str(sender_id or "").strip() or "unknown"
        return f"private:{value}"
    value = str(group_id or "").strip() or "unknown"
    return f"group:{value}"


def qa_missing_record_message() -> str:
    """Return a concise message when there is no usable summary knowledge."""
    return "未找到对应的总结知识库记录，可能已过期或已被清理。"


def qa_record_marker(record_id: str) -> str:
    """Return the stable marker users can quote to select one summary record."""
    value = str(record_id or "").strip()
    return f"{QA_RECORD_MARKER_LABEL}：{value}" if value else ""


def qa_record_id_from_text(text: Any) -> str:
    """Extract a saved summary record id from quoted AI reply text."""
    match = QA_RECORD_ID_RE.search(str(text or ""))
    return match.group(1).lower() if match else ""
