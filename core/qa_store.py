"""Lightweight summary-based QA record storage."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class QARecord:
    record_id: str
    scope_id: str
    summary: str
    source: str
    summary_style: str
    sender_id: str
    created_at: str
    last_accessed_at: str
    qa_history: list[dict[str, str]] = field(default_factory=list)


class QARecordStore:
    """Persist first-pass summaries as short-lived QA knowledge records."""

    def __init__(self, root_dir: Path | str, ttl_minutes: int = 30):
        self.root_dir = Path(root_dir).resolve()
        self.ttl_minutes = max(0, int(ttl_minutes or 0))
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_record(
        self,
        *,
        scope_id: str,
        summary: str,
        source: str,
        summary_style: str,
        sender_id: str,
        created_at: Optional[datetime] = None,
    ) -> QARecord:
        now = self._coerce_datetime(created_at)
        record = QARecord(
            record_id=self._new_record_id(now),
            scope_id=str(scope_id or "").strip(),
            summary=str(summary or "").strip(),
            source=str(source or "").strip(),
            summary_style=str(summary_style or "").strip(),
            sender_id=str(sender_id or "").strip(),
            created_at=self._format_datetime(now),
            last_accessed_at=self._format_datetime(now),
        )
        self._write_record(record)
        return record

    def get_record(
        self,
        scope_id: str,
        record_id: str,
        *,
        accessed_at: Optional[datetime] = None,
    ) -> Optional[QARecord]:
        wanted = str(record_id or "").strip()
        if not wanted:
            return None
        for record in self._records_for_scope(scope_id):
            if record.record_id == wanted or record.record_id.startswith(wanted):
                return self._touch_record(record, accessed_at)
        return None

    def cleanup_expired(self, *, now: Optional[datetime] = None) -> int:
        if self.ttl_minutes <= 0:
            return 0
        current = self._coerce_datetime(now)
        removed = 0
        for path in self.root_dir.glob("*/*.json"):
            data = self._read_json(path)
            if not data:
                continue
            last_accessed = self._parse_datetime(
                data.get("last_accessed_at") or data.get("created_at")
            )
            if (current - last_accessed).total_seconds() <= self.ttl_minutes * 60:
                continue
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        self._remove_empty_scope_dirs()
        return removed

    def delete_scope(self, scope_id: str) -> int:
        """Delete all records for one private or group QA scope."""
        scope_dir = self._scope_dir(scope_id)
        if not scope_dir.exists():
            return 0
        removed = 0
        for path in scope_dir.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        try:
            scope_dir.rmdir()
        except OSError:
            pass
        return removed

    def append_qa_turn(
        self,
        scope_id: str,
        record_id: str,
        *,
        question: str,
        answer: str,
        max_turns: int = 5,
        accessed_at: Optional[datetime] = None,
    ) -> Optional[QARecord]:
        """Append one question-answer turn to a record and keep only recent turns."""
        record = self.get_record(
            scope_id,
            record_id,
            accessed_at=accessed_at,
        )
        if record is None:
            return None

        limit = max(0, int(max_turns or 0))
        if limit <= 0:
            return record

        turn = {
            "question": str(question or "").strip(),
            "answer": str(answer or "").strip(),
        }
        if not turn["question"] or not turn["answer"]:
            return record

        history = self._normalize_history(getattr(record, "qa_history", []))
        history.append(turn)
        record.qa_history = history[-limit:]
        record.last_accessed_at = self._format_datetime(
            self._coerce_datetime(accessed_at)
        )
        self._write_record(record)
        return record

    def _records_for_scope(self, scope_id: str) -> list[QARecord]:
        scope_dir = self._scope_dir(scope_id)
        records: list[QARecord] = []
        for path in scope_dir.glob("*.json"):
            data = self._read_json(path)
            if not data:
                continue
            try:
                record = QARecord(**data)
            except TypeError:
                continue
            record.qa_history = self._normalize_history(
                getattr(record, "qa_history", [])
            )
            if record.summary:
                records.append(record)
        return records

    def _touch_record(
        self,
        record: QARecord,
        accessed_at: Optional[datetime],
    ) -> QARecord:
        record.last_accessed_at = self._format_datetime(
            self._coerce_datetime(accessed_at)
        )
        self._write_record(record)
        return record

    def _write_record(self, record: QARecord) -> None:
        path = self._record_path(record.scope_id, record.record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(asdict(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _record_path(self, scope_id: str, record_id: str) -> Path:
        return self._scope_dir(scope_id) / f"{record_id}.json"

    def _scope_dir(self, scope_id: str) -> Path:
        return self.root_dir / self._safe_name(scope_id)

    def _remove_empty_scope_dirs(self) -> None:
        for path in self.root_dir.iterdir():
            if not path.is_dir():
                continue
            try:
                next(path.iterdir())
            except StopIteration:
                path.rmdir()
            except OSError:
                continue

    @staticmethod
    def _new_record_id(now: datetime) -> str:
        return f"{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _safe_name(value: str) -> str:
        text = str(value or "unknown").strip() or "unknown"
        return re.sub(r"[^0-9A-Za-z_.-]+", "_", text)[:120]

    @staticmethod
    def _read_json(path: Path) -> Optional[dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _normalize_history(value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        history: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "") or "").strip()
            answer = str(item.get("answer", "") or "").strip()
            if question and answer:
                history.append({"question": question, "answer": answer})
        return history

    @staticmethod
    def _coerce_datetime(value: Optional[datetime]) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _format_datetime(value: datetime) -> str:
        return QARecordStore._coerce_datetime(value).isoformat()

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        try:
            parsed = datetime.fromisoformat(str(value or ""))
        except ValueError:
            return datetime.now(timezone.utc)
        return QARecordStore._coerce_datetime(parsed)
