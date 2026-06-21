"""Standalone AstrBot plugin for quoted-message AI summaries."""
from __future__ import annotations

import asyncio
import base64
import html
import json
import mimetypes
import os
import re
import shutil
import uuid
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import unquote, urlparse

import aiohttp

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, register
from astrbot.core.star.filter.event_message_type import EventMessageType

from .core.config import AISummaryConfig, parse_config
from .core.output_render import render_summary_image_file
from .core.qa_runtime import (
    qa_missing_record_message,
    qa_record_id_from_text,
    qa_record_marker,
    qa_scope_id,
)
from .core.qa_store import QARecordStore
from .core.summary import AISummaryManager


VIDEO_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".flv",
    ".ts",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
}


FORWARD_MAX_DEPTH = 1
FORWARD_MAX_NODES = 100
FORWARD_MAX_IMAGE_SOURCES = 16
FORWARD_MAX_VIDEO_SOURCES = 4
FORWARD_CARD_ID_KEYS = (
    "m_resid",
    "resid",
    "forward_id",
    "message_id",
    "file_id",
)


@dataclass
class SummaryCandidate:
    """Quoted message content carried into the summary pipeline."""

    source: str = ""
    text: str = ""
    video_sources: List[str] = field(default_factory=list)
    image_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_content(self) -> bool:
        return bool(
            self.text.strip()
            or self.video_sources
            or self.image_sources
        )


@dataclass(frozen=True)
class QARequest:
    """A question bound to one saved summary record."""

    question: str
    record_id: str


@register(
    "astrbot_plugin_ai_summary",
    "drdon1234",
    "支持引用视频、图片和文字的多模态 AI 总结工具",
    "0.4.0",
)
class AISummaryPlugin(Star):
    """AstrBot plugin entry point for reply-triggered summaries."""

    def __init__(self, context: Context, config: dict):
        """Initialize config, summary manager, task tracking, and concurrency guards."""
        super().__init__(context)
        self.config: AISummaryConfig = parse_config(config)
        logger.info(
            "AI 总结插件已载入: "
            f"cache_dir={self.config.cache_dir}, "
            f"runtime_dir={Path(self.config.cache_dir) / 'runtime'}, "
            f"qa_ttl_minutes={self.config.qa_record_ttl_minutes}, "
            f"qa_history_turns={self.config.qa_history_turns}, "
            f"model_dir={self.config.asr_model_dir}, "
            f"debug_mode={self.config.debug_mode}"
        )
        self.summary_manager = AISummaryManager(
            self.config,
            self.config.cache_dir,
            True,
            context,
        )
        self.qa_store = QARecordStore(
            Path(self.config.cache_dir) / "runtime" / "qa_records",
            self.config.qa_record_ttl_minutes,
        )
        self.summary_manager.start_background_prepare()
        self._shutdown_event = threading.Event()
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._qa_cleanup_task: Optional[asyncio.Task[Any]] = None
        self._qa_reply_bindings: Dict[str, Dict[str, str]] = {}
        self._semaphore = asyncio.Semaphore(max(1, self.config.max_concurrent))

    async def terminate(self):
        """Stop active summary tasks, release runtimes, and clear runtime files."""
        self._shutdown_event.set()
        shutdown_results = await asyncio.gather(
            self._cancel_active_tasks(),
            self._cancel_qa_cleanup_task(),
            self.summary_manager.shutdown(),
            return_exceptions=True,
        )
        for result in shutdown_results:
            if isinstance(result, Exception) and not isinstance(
                result,
                asyncio.CancelledError,
            ):
                logger.warning(f"AI 总结插件终止清理失败: {result}")
        self._clear_video_cache_dir()

    async def _cancel_active_tasks(self) -> None:
        """Cancel in-flight event tasks before plugin shutdown completes."""
        tasks = [task for task in self._active_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_tasks.clear()

    async def _cancel_qa_cleanup_task(self) -> None:
        """Cancel the periodic QA knowledge cleanup task."""
        task = self._qa_cleanup_task
        self._qa_cleanup_task = None
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _track_current_task(self) -> Optional[asyncio.Task[Any]]:
        task = asyncio.current_task()
        if task is None:
            return None
        self._active_tasks.add(task)
        return task

    def _untrack_current_task(
        self,
        task: Optional[asyncio.Task[Any]],
    ) -> None:
        if task is not None:
            self._active_tasks.discard(task)

    def _clear_video_cache_dir(self) -> None:
        """Remove runtime files owned by this plugin instance."""
        raw_cache_dir = str(getattr(self.config, "cache_dir", "") or "").strip()
        if not raw_cache_dir:
            return
        cache_dir = Path(raw_cache_dir).resolve()
        runtime_dirs = (
            cache_dir / "downloads",
            cache_dir / "runtime" / "images",
        )
        for target_dir in runtime_dirs:
            try:
                if target_dir.exists():
                    shutil.rmtree(target_dir, ignore_errors=True)
                    self._debug("已清空运行缓存目录: %r", str(target_dir))
            except Exception as exc:
                logger.warning(f"AI 总结运行缓存目录清理失败: {target_dir}, 错误: {exc}")

    @filter.event_message_type(EventMessageType.ALL)
    async def test_ai_config(self, event: AstrMessageEvent):
        """测试当前 AI 总结配置的连通性。"""
        if self._shutdown_event.is_set():
            return
        current_task = self._track_current_task()
        try:
            text = str(getattr(event, "message_str", "") or "").strip()
            if not self._is_admin_test_keyword(text):
                return

            if not event.is_private_chat():
                return

            sender_id = event.get_sender_id()
            if not self.config.permission.is_admin(sender_id):
                return

            yield event.plain_result("正在测试 AI 总结配置连通性，请稍候...")
            try:
                response = await self.summary_manager.test_llm_connectivity(
                    self._event_context(event)
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                message = self._format_ai_test_failure(exc)
                logger.warning(f"AI 总结配置连通性测试失败: {message}")
                yield event.plain_result(message)
                return

            if self._shutdown_event.is_set():
                return

            yield event.plain_result(self._format_ai_test_success(response))
        finally:
            self._untrack_current_task(current_task)

    @filter.event_message_type(EventMessageType.ALL)
    async def answer_summary_question(self, event: AstrMessageEvent):
        """Answer questions against the quoted summary knowledge record."""
        if self._shutdown_event.is_set():
            return
        current_task = self._track_current_task()
        try:
            cfg = self.config
            if not cfg.qa_enabled:
                return

            is_private = event.is_private_chat()
            sender_id = event.get_sender_id()
            group_id = None if is_private else event.get_group_id()
            if not cfg.permission.check(is_private, sender_id, group_id):
                return

            scope_id = qa_scope_id(is_private, sender_id, group_id)
            text = str(getattr(event, "message_str", "") or "").strip()
            handled, message = self._handle_qa_control_command(text, scope_id)
            if handled:
                await event.send(event.plain_result(message))
                self._stop_event(event)
                return

            command = self._qa_request_for_event(event, scope_id)
            if command is None:
                return

            self._ensure_qa_cleanup_task()
            self._cleanup_qa_records()
            record = self.qa_store.get_record(scope_id, command.record_id)

            if record is None:
                await event.send(event.plain_result(qa_missing_record_message()))
                self._stop_event(event)
                return

            try:
                answer = await self.summary_manager.answer_summary_question(
                    record,
                    command.question,
                    metadata=self._event_context(event),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"AI 总结问答失败: {exc}")
                if cfg.show_error:
                    await event.send(event.plain_result(f"AI问答失败：{exc}"))
                self._stop_event(event)
                return

            if answer and not self._shutdown_event.is_set():
                updated_record = self._append_qa_turn(
                    scope_id,
                    record,
                    command.question,
                    answer,
                )
                if updated_record is not None:
                    record = updated_record
                send_result = await self._send_qa_output(
                    event,
                    answer,
                    getattr(record, "record_id", ""),
                )
                self._remember_qa_reply(
                    scope_id,
                    getattr(record, "record_id", ""),
                    send_result,
                )
                self._stop_event(event)
        finally:
            self._untrack_current_task(current_task)

    @filter.event_message_type(EventMessageType.ALL)
    async def summarize_video(self, event: AstrMessageEvent):
        """Handle reply messages that request AI summarization."""
        if self._shutdown_event.is_set():
            self._debug("总结事件忽略: 插件正在关闭")
            return
        if self._is_event_stopped(event):
            self._debug("总结事件忽略: event 已被其他处理器停止")
            return
        current_task = self._track_current_task()
        try:
            cfg = self.config
            text = event.message_str or ""
            if "AI总结：" in text or "AI 总结：" in text:
                self._debug("总结事件忽略: 检测到插件自己的总结输出 text=%r", text[:120])
                return

            is_private = event.is_private_chat()
            sender_id = event.get_sender_id()
            group_id = None if is_private else event.get_group_id()
            self._debug(
                "总结事件入口: private=%s sender=%s group=%s text=%r",
                is_private,
                sender_id,
                group_id or "",
                text[:120],
            )
            if not cfg.permission.check(is_private, sender_id, group_id):
                self._debug(
                    "总结事件忽略: 权限未通过 private=%s sender=%s group=%s",
                    is_private,
                    sender_id,
                    group_id or "",
                )
                return

            scope_id = qa_scope_id(is_private, sender_id, group_id)
            if getattr(cfg, "qa_enabled", True) and self._record_id_for_replied_qa(
                scope_id,
                event,
            ):
                self._debug("总结事件忽略: 当前回复命中已有总结问答记录 scope=%s", scope_id)
                return

            requested_style = cfg.summary_style_for_text(text)
            summarize_reply = bool(cfg.reply_keyword_trigger and requested_style)
            self._debug(
                "触发检查: reply=%s style=%s text=%r",
                summarize_reply,
                requested_style or "none",
                text[:120],
            )
            if not summarize_reply:
                return

            self._ensure_qa_cleanup_task()
            self._cleanup_qa_records()
            candidates: List[SummaryCandidate] = []
            candidates.extend(await self._extract_reply_candidates(event))

            candidates = self._dedupe_candidates(candidates)
            self._debug(
                "抽取到引用总结候选: %s",
                [
                    {
                        "source": candidate.source,
                        "text_chars": len(candidate.text),
                        "videos": len(candidate.video_sources),
                        "images": len(candidate.image_sources),
                    }
                    for candidate in candidates
                ],
            )
            if not candidates:
                if requested_style:
                    await event.send(
                        event.plain_result(
                            "未找到可总结的引用内容，请引用包含视频、图片或文字的消息后再发送总结命令。"
                        )
                    )
                return

            candidates = candidates[: cfg.max_videos_per_message]
            user_hint = self._user_hint_from_text(text)
            self._attach_user_hint_to_candidates(
                candidates,
                user_hint,
                event,
                requested_style,
            )
            self._debug(
                "可选用户附加说明: style=%s hint=%r",
                requested_style,
                user_hint[:120],
            )
            if cfg.status_message:
                await event.send(event.plain_result("正在进行 AI 总结，请稍候..."))

            async with self._semaphore:
                results = await self._summarize_candidates(candidates)

            outputs: List[tuple[str, Optional[Any]]] = []
            for metadata in results:
                summary = str(metadata.get("ai_summary") or "").strip()
                error = str(metadata.get("ai_summary_error") or "").strip()
                if summary:
                    record = self._save_qa_record_from_summary(
                        scope_id,
                        metadata,
                        summary,
                        sender_id,
                    )
                    if record is not None:
                        outputs.append((self._format_summary_message(summary), record))
                    else:
                        outputs.append((self._format_summary_message(summary), None))
                elif cfg.show_error and error:
                    outputs.append((f"AI总结失败：{error}", None))

            for message, record in outputs:
                if self._shutdown_event.is_set():
                    break
                send_result = await self._send_summary_output(
                    event,
                    message,
                    getattr(record, "record_id", "") if record is not None else "",
                )
                if record is not None:
                    self._remember_qa_reply(
                        scope_id,
                        getattr(record, "record_id", ""),
                        send_result,
                    )
        finally:
            self._untrack_current_task(current_task)

    async def _send_summary_output(
        self,
        event: AstrMessageEvent,
        message: str,
        record_id: str = "",
    ) -> Any:
        """Send summary as text or rendered image according to output config."""
        marker = qa_record_marker(record_id)
        if str(getattr(self.config, "send_format", "text") or "text") != "image":
            return await event.send(
                event.plain_result(self._append_qa_marker(message, marker))
            )

        image_ref = await self._render_summary_image(message)
        if not marker:
            return await event.send(event.image_result(image_ref))
        return await event.send(
            event.chain_result([Image.fromFileSystem(image_ref), Plain(f"\n{marker}")])
        )

    async def _send_qa_output(
        self,
        event: AstrMessageEvent,
        answer: str,
        record_id: str = "",
    ) -> Any:
        """Send QA answers as text or rendered image according to output config."""
        message = self._format_qa_message(answer)
        marker = qa_record_marker(record_id)
        if str(getattr(self.config, "qa_send_format", "text") or "text") != "image":
            return await event.send(
                event.plain_result(self._append_qa_marker(message, marker))
            )

        image_ref = await self._render_qa_image(message)
        if not marker:
            return await event.send(event.image_result(image_ref))
        return await event.send(
            event.chain_result([Image.fromFileSystem(image_ref), Plain(f"\n{marker}")])
        )

    async def _render_summary_image(self, message: str) -> str:
        """Render final summary content with the plugin-owned local renderer."""
        return await self._render_output_image(
            message,
            getattr(self.config, "summary_format", "text"),
            "summary",
            "AI 总结",
        )

    async def _render_qa_image(self, message: str) -> str:
        """Render a QA answer with the plugin-owned local renderer."""
        return await self._render_output_image(
            message,
            getattr(self.config, "qa_answer_format", "text"),
            "qa",
            "AI 问答",
        )

    async def _render_output_image(
        self,
        message: str,
        content_format: str,
        prefix: str,
        title: str,
    ) -> str:
        """Render final chat content as a local PNG file."""
        image_dir = Path(self.config.cache_dir).resolve() / "runtime" / "images"
        image_path = image_dir / f"{prefix}_{uuid.uuid4().hex}.png"
        return await render_summary_image_file(
            message,
            content_format,
            str(image_path),
            title=title,
            font_size=getattr(self.config, "image_font_size", 25),
            style=getattr(self.config, "image_style", "fresh"),
            font_family=getattr(self.config, "image_font_family", "noto_sans"),
        )

    def _format_summary_message(self, summary: str) -> str:
        """Format one summary for the configured content format."""
        text = str(summary or "").strip()
        if str(getattr(self.config, "summary_format", "text") or "text") == "markdown":
            return text if text.startswith("# ") else f"# AI 总结\n\n{text}"
        return f"AI总结：\n{text}"

    def _format_qa_message(self, answer: str) -> str:
        """Format one QA answer for chat delivery."""
        text = str(answer or "").strip()
        if str(getattr(self.config, "qa_answer_format", "text") or "text") == "markdown":
            return text if text.startswith("# ") else f"# AI 问答\n\n{text}"
        return f"AI问答：\n{text}"

    @staticmethod
    def _append_qa_marker(message: str, marker: str) -> str:
        """Append a stable quoted-reply marker without changing stored summaries."""
        text = str(message or "").strip()
        marker_text = str(marker or "").strip()
        if not marker_text:
            return text
        return f"{text}\n\n{marker_text}" if text else marker_text

    def _qa_request_for_event(
        self,
        event: AstrMessageEvent,
        scope_id: str,
    ) -> Optional[QARequest]:
        """Return a QA request only when the user replies to a bound AI message."""
        text = str(getattr(event, "message_str", "") or "").strip()
        replied_record_id = self._record_id_for_replied_qa(scope_id, event)
        if not text or not replied_record_id:
            return None
        return QARequest(question=text, record_id=replied_record_id)

    def _record_id_for_replied_qa(
        self,
        scope_id: str,
        event: AstrMessageEvent,
    ) -> str:
        """Return the QA record id bound to the replied plugin message."""
        reply_ids = self._reply_message_ids(event)
        bindings = getattr(self, "_qa_reply_bindings", {}).get(scope_id, {})
        for reply_id in reply_ids:
            record_id = str(bindings.get(reply_id, "") or "").strip()
            if record_id:
                return record_id
        for reply_text in self._reply_message_texts(event):
            record_id = qa_record_id_from_text(reply_text)
            if record_id:
                return record_id
        return ""

    def _remember_qa_reply(
        self,
        scope_id: str,
        record_id: str,
        send_result: Any = None,
    ) -> None:
        """Bind a sent plugin reply id to the selected summary record."""
        scope = str(scope_id or "").strip()
        if not scope:
            return
        message_id = self._extract_message_id(send_result)
        if message_id:
            bindings = getattr(self, "_qa_reply_bindings", None)
            if bindings is None:
                self._qa_reply_bindings = {}
            self._qa_reply_bindings.setdefault(scope, {})[message_id] = str(
                record_id or ""
            ).strip()

    def _handle_qa_control_command(self, text: str, scope_id: str) -> tuple[bool, str]:
        """Handle QA context and knowledge cleanup commands."""
        command = str(text or "").strip().lstrip("/").strip()
        if command in set(getattr(self.config, "qa_exit_commands", ["结束", "退出"])):
            return True, "已结束当前问答。"
        if command in set(getattr(self.config, "qa_clear_commands", ["清理", "清空"])):
            removed = self._clear_qa_knowledge(scope_id)
            return True, f"已清理当前问答知识库（{removed} 条记录）。"
        return False, ""

    def _clear_qa_knowledge(self, scope_id: str) -> int:
        """Delete all QA records and reply bindings for one scope."""
        scope = str(scope_id or "").strip()
        if scope:
            getattr(self, "_qa_reply_bindings", {}).pop(scope, None)
        return self.qa_store.delete_scope(scope_id)

    def _reply_message_ids(self, event: AstrMessageEvent) -> set[str]:
        """Return reply component ids present on an incoming message."""
        ids: set[str] = set()
        for comp in self._safe_get_messages(event):
            if comp.__class__.__name__.lower() != "reply":
                continue
            for attr in ("id", "message_id", "msg_id"):
                value = str(getattr(comp, attr, "") or "").strip()
                if value:
                    ids.add(value)
        return ids

    def _reply_message_texts(self, event: AstrMessageEvent) -> List[str]:
        """Return quoted message text fragments that may contain QA record markers."""
        texts: List[str] = []
        for comp in self._safe_get_messages(event):
            if comp.__class__.__name__.lower() != "reply":
                continue
            for attr in ("message_str", "text"):
                value = str(getattr(comp, attr, "") or "").strip()
                if value:
                    texts.append(value)
            chain = getattr(comp, "chain", None)
            if isinstance(chain, list):
                for item in chain:
                    value = str(getattr(item, "text", "") or "").strip()
                    if value:
                        texts.append(value)
        return texts

    @staticmethod
    def _stop_event(event: AstrMessageEvent) -> None:
        """Stop later handlers after this plugin has handled a QA event."""
        stopper = getattr(event, "stop_event", None)
        if callable(stopper):
            stopper()

    @staticmethod
    def _is_event_stopped(event: AstrMessageEvent) -> bool:
        """Return whether an earlier handler already stopped this event."""
        checker = getattr(event, "is_stopped", None)
        if not callable(checker):
            return False
        try:
            return bool(checker())
        except Exception:
            return False

    @staticmethod
    def _extract_message_id(value: Any) -> str:
        """Best-effort extraction of a sent message id from platform results."""
        if value is None:
            return ""
        if isinstance(value, (str, int)):
            return str(value).strip()
        if isinstance(value, dict):
            for key in ("message_id", "msg_id", "id"):
                text = str(value.get(key, "") or "").strip()
                if text:
                    return text
            data = value.get("data")
            if data is not value:
                return AISummaryPlugin._extract_message_id(data)
            return ""
        for attr in ("message_id", "msg_id", "id"):
            text = str(getattr(value, attr, "") or "").strip()
            if text:
                return text
        if isinstance(value, (list, tuple)):
            for item in value:
                text = AISummaryPlugin._extract_message_id(item)
                if text:
                    return text
        return ""

    def _cleanup_qa_records(self) -> None:
        """Remove QA records that have not been accessed within the configured TTL."""
        if not getattr(self.config, "qa_enabled", True):
            return
        try:
            removed = self.qa_store.cleanup_expired()
        except Exception as exc:
            logger.warning(f"AI 总结问答知识库清理失败: {exc}")
            return
        if removed:
            self._debug("问答知识库清理完成: removed=%d", removed)

    def _ensure_qa_cleanup_task(self) -> None:
        """Start periodic QA cleanup once an event loop is available."""
        if not getattr(self.config, "qa_enabled", True):
            return
        if int(getattr(self.config, "qa_record_ttl_minutes", 30) or 0) <= 0:
            return
        if self._qa_cleanup_task is not None and not self._qa_cleanup_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._qa_cleanup_task = loop.create_task(self._qa_cleanup_loop())

    async def _qa_cleanup_loop(self) -> None:
        """Periodically enforce the idle TTL for saved summary knowledge records."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(60)
            self._cleanup_qa_records()

    def _save_qa_record_from_summary(
        self,
        scope_id: str,
        metadata: Dict[str, Any],
        summary: str,
        sender_id: Any,
    ) -> Optional[Any]:
        """Persist a successful first-pass summary as a QA knowledge record."""
        if not getattr(self.config, "qa_enabled", True):
            return None
        text = str(summary or "").strip()
        if not text:
            return None
        try:
            return self.qa_store.save_record(
                scope_id=scope_id,
                summary=text,
                source=str(metadata.get("url") or metadata.get("source") or "").strip(),
                summary_style=str(
                    metadata.get("_ai_summary_effective_style")
                    or metadata.get("summary_style")
                    or ""
                ).strip(),
                sender_id=str(sender_id or "").strip(),
            )
        except Exception as exc:
            logger.warning(f"AI 总结问答知识库保存失败: {exc}")
            return None

    def _append_qa_turn(
        self,
        scope_id: str,
        record: Any,
        question: str,
        answer: str,
    ) -> Optional[Any]:
        """Persist one completed QA pair onto the selected summary record."""
        record_id = str(getattr(record, "record_id", "") or "").strip()
        if not record_id:
            return None
        try:
            return self.qa_store.append_qa_turn(
                scope_id,
                record_id,
                question=question,
                answer=answer,
                max_turns=int(getattr(self.config, "qa_history_turns", 5) or 0),
            )
        except Exception as exc:
            logger.warning(f"AI 总结问答历史保存失败: {exc}")
            return None

    async def _summarize_candidates(
        self,
        candidates: List[SummaryCandidate],
    ) -> List[Dict[str, Any]]:
        """Prepare quoted content, run summaries, and clean temporary files."""
        metadata_list: List[Dict[str, Any]] = []
        started_at = time.perf_counter()
        self._debug("批处理开始: candidates=%d", len(candidates))
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=max(30, self.config.download_timeout_seconds)
                )
            ) as session:
                for index, candidate in enumerate(candidates, start=1):
                    self._debug(
                        "候选[%d]准备开始: source=%r text_chars=%d videos=%d images=%d",
                        index,
                        candidate.source,
                        len(candidate.text),
                        len(candidate.video_sources),
                        len(candidate.image_sources),
                    )
                    try:
                        metadata = await self._prepare_candidate(session, candidate)
                        self._debug(
                            "候选[%d]准备完成: error=%s",
                            index,
                            bool(metadata.get("ai_summary_error")),
                        )
                    except Exception as exc:
                        logger.warning(
                            f"AI 总结候选准备失败: source={candidate.source!r}, 错误: {exc}"
                        )
                        metadata = dict(candidate.metadata)
                        metadata["url"] = candidate.source
                        metadata["ai_summary_error"] = str(exc)
                    metadata_list.append(metadata)

            ready = [
                metadata for metadata in metadata_list
                if not metadata.get("ai_summary_error")
            ]
            self._debug(
                "候选准备汇总: total=%d ready=%d failed=%d",
                len(metadata_list),
                len(ready),
                len(metadata_list) - len(ready),
            )
            if ready:
                await self.summary_manager.summarize_metadata_list(ready)
        except asyncio.CancelledError:
            self._debug(
                "批处理被取消: elapsed=%.2fs",
                time.perf_counter() - started_at,
            )
            raise
        except Exception as exc:
            logger.warning(f"AI 总结批处理失败: {exc}")
            for metadata in metadata_list:
                metadata.setdefault("ai_summary_error", str(exc))
        finally:
            self._cleanup_downloaded_files(metadata_list)
            self._debug(
                "批处理结束: metadata=%d elapsed=%.2fs",
                len(metadata_list),
                time.perf_counter() - started_at,
            )
        return metadata_list

    def _debug(self, message: str, *args: Any) -> None:
        if not bool(getattr(self.config, "debug_mode", False)):
            return
        try:
            text = message % args if args else message
        except Exception:
            text = message
        logger.debug(f"AI 总结调试: {text}")

    @staticmethod
    def _payload_shape(value: Any) -> str:
        """Return a compact shape summary for debug logs without dumping content."""
        if value is None:
            return "None"
        if isinstance(value, dict):
            parts = [f"dict(keys={list(value.keys())[:8]})"]
            message = value.get("message") or value.get("messages")
            if isinstance(message, list):
                parts.append(f"message_len={len(message)}")
            elif isinstance(message, str):
                parts.append(f"message_chars={len(message)}")
            raw_message = value.get("raw_message")
            if isinstance(raw_message, str):
                parts.append(f"raw_chars={len(raw_message)}")
            data = value.get("data")
            if data is not None:
                parts.append(f"data_type={type(data).__name__}")
            return " ".join(parts)
        if isinstance(value, list):
            return f"list(len={len(value)})"
        if isinstance(value, str):
            return f"str(chars={len(value)})"
        return type(value).__name__

    @staticmethod
    def _segment_types(segments: Iterable[Any]) -> List[str]:
        """Return segment type names for debug logs."""
        types: List[str] = []
        for segment in list(segments)[:20]:
            if isinstance(segment, dict):
                segment_type = str(segment.get("type", "") or "").strip()
                types.append(segment_type or "dict")
            else:
                types.append(type(segment).__name__)
        return types

    @staticmethod
    def _text_has_forward_hint(text: str) -> bool:
        value = str(text or "")
        if not value:
            return False
        lowered = value.lower()
        return any(
            marker in lowered
            for marker in (
                "com.tencent.multimsg",
                "multimsg",
                "m_resid",
                "resid",
                "forward",
                "聊天记录",
                "转发消息",
                "合并转发",
            )
        )

    @staticmethod
    def _format_bytes(size: int) -> str:
        if size < 1024:
            return f"{size}B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f}KB"
        return f"{size / 1024 / 1024:.1f}MB"

    def _is_admin_test_keyword(self, text: str) -> bool:
        keyword = str(getattr(self.config, "admin_test_keyword", "") or "").strip()
        if not keyword:
            keyword = "aiping"
        normalized = str(text or "").strip()
        if not normalized:
            return False
        normalized = normalized.lstrip("/").strip()
        return normalized.casefold() == keyword.casefold()

    def _format_ai_test_success(self, response: str) -> str:
        reply = self._truncate_for_message(response, 300)
        lines = [
            "AI 配置连通性测试成功。",
        ]
        if getattr(self.config, "llm_provider_source", "astrbot") == "astrbot":
            provider_id = str(
                getattr(self.config, "astrbot_provider_id", "") or "当前会话 AI"
            ).strip()
            lines.extend([
                "AI 来源: AstrBot 内置提供商",
                f"AstrBot Provider: {provider_id}",
            ])
        else:
            provider = str(getattr(self.config, "llm_provider", "") or "未配置").strip()
            model = str(getattr(self.config, "model", "") or "未配置").strip()
            base_url = str(getattr(self.config, "base_url", "") or "默认").strip()
            lines.extend([
                "AI 来源: 插件自定义提供商",
                f"模型厂商: {provider}",
                f"模型: {model}",
                f"Base URL: {self._redact_sensitive_text(base_url)}",
            ])
        if reply:
            lines.append(f"模型响应: {reply}")
        return "\n".join(lines)

    def _format_ai_test_failure(self, exc: Exception) -> str:
        detail = self._redact_sensitive_text(str(exc))
        if not detail:
            detail = "未知错误"
        return (
            "AI 配置连通性测试失败：\n"
            f"{detail}\n\n"
            "请检查模型厂商、Base URL、API Key、模型名和网络连通性。"
        )

    def _redact_sensitive_text(self, text: str) -> str:
        redacted = str(text or "").strip()
        api_key = str(getattr(self.config, "api_key", "") or "").strip()
        if api_key:
            redacted = redacted.replace(api_key, "***")
        return self._truncate_for_message(redacted, 1200)

    @staticmethod
    def _truncate_for_message(text: str, limit: int) -> str:
        value = str(text or "").strip()
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + "...(已截断)"

    async def _prepare_candidate(
        self,
        session: aiohttp.ClientSession,
        candidate: SummaryCandidate,
    ) -> Dict[str, Any]:
        """Convert supported quoted content into summary-ready metadata."""
        source = candidate.source.strip()
        cleanup_paths: List[str] = []
        video_paths: List[str] = []
        video_urls: List[List[str]] = []
        image_paths: List[str] = []
        image_urls: List[List[str]] = []
        preparation_errors: List[str] = []
        try:
            for video_source in candidate.video_sources:
                try:
                    video_path, should_cleanup = await self._prepare_video_source(
                        session,
                        video_source,
                    )
                    if should_cleanup:
                        cleanup_paths.append(video_path)
                    video_paths.append(video_path)
                    video_urls.append([video_source])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    preparation_errors.append(f"视频准备失败: {exc}")
                    self._debug(
                        "视频候选准备失败: source=%r error=%s",
                        video_source,
                        exc,
                    )

            for image_source in candidate.image_sources:
                try:
                    image_path, should_cleanup = await self._prepare_image_source(
                        session,
                        image_source,
                    )
                    if should_cleanup:
                        cleanup_paths.append(image_path)
                    image_paths.append(image_path)
                    image_urls.append([image_source])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    preparation_errors.append(f"图片准备失败: {exc}")
                    self._debug(
                        "图片候选准备失败: source=%r error=%s",
                        image_source,
                        exc,
                    )

            metadata = dict(candidate.metadata)
            metadata.setdefault("url", source or "引用消息")
            metadata["source"] = source or "引用消息"
            metadata["content_text"] = candidate.text.strip()
            metadata["video_urls"] = video_urls
            metadata["file_paths"] = video_paths
            metadata["video_modes"] = ["local"] * len(video_paths)
            metadata["video_count"] = len(video_paths)
            metadata["image_urls"] = image_urls
            metadata["image_file_paths"] = image_paths
            metadata["image_modes"] = ["local"] * len(image_paths)
            metadata["image_count"] = len(image_paths)
            metadata["has_valid_media"] = bool(video_paths or image_paths)
            if preparation_errors:
                metadata["preparation_errors"] = preparation_errors
            if cleanup_paths:
                metadata["_cleanup_file_paths"] = cleanup_paths
            if not metadata["content_text"] and not video_paths and not image_paths:
                detail = "；".join(preparation_errors) if preparation_errors else "无可用内容"
                raise RuntimeError(detail)
            self._debug(
                "候选源准备完成: source=%r text_chars=%d videos=%d images=%d cleanup=%d",
                source,
                len(metadata["content_text"]),
                len(video_paths),
                len(image_paths),
                len(cleanup_paths),
            )
            return metadata
        except Exception:
            if cleanup_paths:
                self._cleanup_file_paths(cleanup_paths)
            raise

    async def _prepare_video_source(
        self,
        session: aiohttp.ClientSession,
        source: str,
    ) -> tuple[str, bool]:
        """Download a supported video source and return its local path."""
        normalized = source.strip()
        if not normalized.lower().startswith(("http://", "https://")):
            raise RuntimeError(f"不支持的视频来源: {normalized}")
        video_path = await self._download_video(session, normalized)
        try:
            file_size = os.path.getsize(video_path)
            self._debug(
                "视频源准备完成: source=%r path=%r size=%s",
                normalized,
                video_path,
                self._format_bytes(file_size),
            )
            max_bytes = int(self.config.max_video_size_mb * 1024 * 1024)
            if max_bytes > 0 and file_size > max_bytes:
                size_mb = file_size / 1024 / 1024
                raise RuntimeError(f"视频超过大小限制: {size_mb:.1f}MB")
            return video_path, True
        except Exception:
            self._cleanup_file_paths([video_path])
            raise

    async def _prepare_image_source(
        self,
        session: aiohttp.ClientSession,
        source: str,
    ) -> tuple[str, bool]:
        """Resolve a quoted image source to a local path usable by vision models."""
        normalized = source.strip()
        if not normalized:
            raise RuntimeError("空图片来源")
        if normalized.lower().startswith(("http://", "https://")):
            return await self._download_image(session, normalized), True
        if normalized.lower().startswith("data:image/"):
            return self._write_image_data_url(normalized), True

        local_path = self._local_path_from_source(normalized)
        if local_path and os.path.isfile(local_path):
            return local_path, False
        raise RuntimeError(f"不支持的图片来源: {normalized}")

    async def _download_video(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> str:
        """Download a remote video into the plugin cache with size checks."""
        cache_dir = Path(self.config.cache_dir).resolve() / "downloads"
        cache_dir.mkdir(parents=True, exist_ok=True)

        suffix = self._suffix_for_url(url)
        target = cache_dir / f"{uuid.uuid4().hex}{suffix}"
        max_bytes = int(self.config.max_video_size_mb * 1024 * 1024)
        started_at = time.perf_counter()
        self._debug(
            "下载开始: url=%r target=%r max_size=%s timeout=%ss",
            url,
            str(target),
            self._format_bytes(max_bytes) if max_bytes > 0 else "unlimited",
            self.config.download_timeout_seconds,
        )

        try:
            async with session.get(url) as response:
                self._debug(
                    "下载响应: status=%s content_length=%r content_type=%r",
                    response.status,
                    response.headers.get("Content-Length"),
                    response.headers.get("Content-Type"),
                )
                if response.status >= 400:
                    raise RuntimeError(f"下载视频失败: HTTP {response.status}")
                content_length = response.headers.get("Content-Length")
                if (
                    max_bytes > 0 and
                    content_length and
                    int(content_length) > max_bytes
                ):
                    raise RuntimeError(
                        "视频超过大小限制: "
                        f"{int(content_length) / 1024 / 1024:.1f}MB"
                    )

                total = 0
                with target.open("wb") as fh:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        total += len(chunk)
                        if max_bytes > 0 and total > max_bytes:
                            raise RuntimeError(
                                f"视频超过大小限制: {total / 1024 / 1024:.1f}MB"
                            )
                        fh.write(chunk)
        except asyncio.CancelledError:
            self._debug(
                "下载被取消: target=%r elapsed=%.2fs",
                str(target),
                time.perf_counter() - started_at,
            )
            try:
                if target.exists():
                    target.unlink()
            except Exception as exc:
                self._debug("下载取消后清理失败: path=%r error=%s", str(target), exc)
            raise
        except Exception as exc:
            self._debug(
                "下载失败: target=%r elapsed=%.2fs error=%s",
                str(target),
                time.perf_counter() - started_at,
                exc,
            )
            try:
                if target.exists():
                    target.unlink()
            except Exception as exc:
                self._debug("下载失败后清理失败: path=%r error=%s", str(target), exc)
            raise

        if target.stat().st_size <= 0:
            try:
                target.unlink()
            except Exception as exc:
                self._debug("空视频文件清理失败: path=%r error=%s", str(target), exc)
            raise RuntimeError("下载到空视频文件")
        self._debug(
            "下载完成: path=%r size=%s elapsed=%.2fs",
            str(target),
            self._format_bytes(target.stat().st_size),
            time.perf_counter() - started_at,
        )
        return str(target)

    async def _download_image(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> str:
        """Download a remote image into the plugin cache with size checks."""
        cache_dir = Path(self.config.cache_dir).resolve() / "downloads"
        cache_dir.mkdir(parents=True, exist_ok=True)

        suffix = self._suffix_for_image_url(url)
        target = cache_dir / f"{uuid.uuid4().hex}{suffix}"
        used_default_suffix = suffix == ".jpg"
        max_bytes = self._max_image_size_bytes()
        started_at = time.perf_counter()
        self._debug(
            "图片下载开始: url=%r target=%r max_size=%s timeout=%ss",
            url,
            str(target),
            self._format_bytes(max_bytes) if max_bytes > 0 else "unlimited",
            self.config.download_timeout_seconds,
        )

        try:
            async with session.get(url) as response:
                content_type = str(response.headers.get("Content-Type", "") or "")
                self._debug(
                    "图片下载响应: status=%s content_length=%r content_type=%r",
                    response.status,
                    response.headers.get("Content-Length"),
                    content_type,
                )
                if response.status >= 400:
                    raise RuntimeError(f"下载图片失败: HTTP {response.status}")
                lowered_content_type = content_type.lower()
                if (
                    lowered_content_type
                    and "image/" not in lowered_content_type
                    and "octet-stream" not in lowered_content_type
                ):
                    raise RuntimeError(f"下载目标不是图片: {content_type}")
                if used_default_suffix and "image/" in lowered_content_type:
                    response_suffix = mimetypes.guess_extension(
                        lowered_content_type.split(";", 1)[0].strip()
                    )
                    if response_suffix == ".jpe":
                        response_suffix = ".jpg"
                    if response_suffix and response_suffix.lower() in IMAGE_EXTENSIONS:
                        target = target.with_suffix(response_suffix.lower())
                content_length = response.headers.get("Content-Length")
                if (
                    max_bytes > 0 and
                    content_length and
                    int(content_length) > max_bytes
                ):
                    raise RuntimeError(
                        "图片超过大小限制: "
                        f"{int(content_length) / 1024 / 1024:.1f}MB"
                    )

                total = 0
                with target.open("wb") as fh:
                    async for chunk in response.content.iter_chunked(512 * 1024):
                        total += len(chunk)
                        if max_bytes > 0 and total > max_bytes:
                            raise RuntimeError(
                                f"图片超过大小限制: {total / 1024 / 1024:.1f}MB"
                            )
                        fh.write(chunk)
        except asyncio.CancelledError:
            self._unlink_download_target(target, "图片下载取消后清理失败")
            raise
        except Exception:
            self._unlink_download_target(target, "图片下载失败后清理失败")
            raise

        if target.stat().st_size <= 0:
            self._unlink_download_target(target, "空图片文件清理失败")
            raise RuntimeError("下载到空图片文件")
        self._debug(
            "图片下载完成: path=%r size=%s elapsed=%.2fs",
            str(target),
            self._format_bytes(target.stat().st_size),
            time.perf_counter() - started_at,
        )
        return str(target)

    def _write_image_data_url(self, data_url: str) -> str:
        """Persist a data URL image into the plugin download cache."""
        match = re.match(
            r"^data:(image/[^;,]+)(?:;charset=[^;,]+)?;base64,(.+)$",
            data_url.strip(),
            re.I | re.S,
        )
        if not match:
            raise RuntimeError("图片 data URL 格式无效")
        mime_type = match.group(1).lower()
        suffix = mimetypes.guess_extension(mime_type) or ".jpg"
        if suffix == ".jpe":
            suffix = ".jpg"
        if suffix.lower() not in IMAGE_EXTENSIONS:
            suffix = ".jpg"
        raw = base64.b64decode(match.group(2).strip(), validate=False)
        max_bytes = self._max_image_size_bytes()
        if max_bytes > 0 and len(raw) > max_bytes:
            raise RuntimeError(f"图片超过大小限制: {len(raw) / 1024 / 1024:.1f}MB")
        cache_dir = Path(self.config.cache_dir).resolve() / "downloads"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{uuid.uuid4().hex}{suffix}"
        target.write_bytes(raw)
        if target.stat().st_size <= 0:
            target.unlink(missing_ok=True)
            raise RuntimeError("写入空图片文件")
        return str(target)

    def _unlink_download_target(self, target: Path, debug_message: str) -> None:
        try:
            if target.exists():
                target.unlink()
        except Exception as exc:
            self._debug("%s: path=%r error=%s", debug_message, str(target), exc)

    @staticmethod
    def _max_image_size_bytes() -> int:
        return 25 * 1024 * 1024

    def _cleanup_downloaded_files(
        self,
        metadata_list: Iterable[Dict[str, Any]],
    ) -> None:
        """Delete downloaded files recorded on summary metadata."""
        for metadata in metadata_list:
            paths = metadata.get("_cleanup_file_paths")
            if isinstance(paths, list):
                self._cleanup_file_paths(paths)

    def _cleanup_file_paths(self, paths: Iterable[Any]) -> None:
        """Delete only files that live under the plugin downloads directory."""
        downloads_dir = Path(self.config.cache_dir).resolve() / "downloads"
        for raw_path in paths:
            if not raw_path:
                continue
            try:
                path = Path(str(raw_path)).resolve()
            except Exception:
                continue
            try:
                path.relative_to(downloads_dir)
            except ValueError:
                self._debug("跳过非插件下载文件清理: %r", str(path))
                continue
            try:
                if path.is_file():
                    path.unlink()
                    self._debug("已清理下载文件: %r", str(path))
            except Exception as exc:
                logger.warning(f"AI 总结下载文件清理失败: {path}, 错误: {exc}")

    async def _extract_reply_candidates(
        self,
        event: AstrMessageEvent,
    ) -> List[SummaryCandidate]:
        """Collect summary candidates from quoted messages and remote fallbacks."""
        candidates: List[SummaryCandidate] = []
        messages = self._safe_get_messages(event)
        reply_components = [
            comp for comp in messages
            if comp.__class__.__name__.lower() == "reply"
        ]
        self._debug(
            "引用组件扫描: total=%d replies=%d component_types=%s",
            len(messages),
            len(reply_components),
            [comp.__class__.__name__ for comp in messages[:10]],
        )
        for comp in reply_components:
            if comp.__class__.__name__.lower() != "reply":
                continue
            reply_id = str(getattr(comp, "id", "") or "").strip()
            context: Dict[str, Any] = {}
            candidate = SummaryCandidate(
                source=f"reply:{reply_id}" if reply_id else "引用消息",
                metadata=dict(context),
            )
            chain = getattr(comp, "chain", None) or []
            chain_list = chain if isinstance(chain, list) else []
            self._debug(
                "处理引用组件: reply_id=%r chain_type=%s chain_count=%d chain_types=%s",
                reply_id,
                type(chain).__name__,
                len(chain_list),
                [item.__class__.__name__ for item in chain_list[:10]],
            )
            local_candidate = self._extract_candidate_from_parts(chain, context)
            self._debug(
                "引用本地内容: reply_id=%r text_chars=%d videos=%d images=%d source=%r",
                reply_id,
                len(local_candidate.text),
                len(local_candidate.video_sources),
                len(local_candidate.image_sources),
                local_candidate.source,
            )
            self._merge_candidate_content(
                candidate,
                local_candidate,
            )
            forward_candidate = await self._extract_forward_candidates_from_parts(
                event,
                chain,
                context,
            )
            self._debug(
                "引用本地转发内容: reply_id=%r text_chars=%d videos=%d images=%d source=%r",
                reply_id,
                len(forward_candidate.text),
                len(forward_candidate.video_sources),
                len(forward_candidate.image_sources),
                forward_candidate.source,
            )
            self._merge_candidate_content(
                candidate,
                forward_candidate,
            )
            remote_candidate = await self._extract_reply_remote_candidate(
                event,
                comp,
                context,
            )
            self._merge_candidate_content(candidate, remote_candidate)
            self._debug(
                "引用候选汇总: reply_id=%r has_content=%s text_chars=%d videos=%d images=%d source=%r",
                reply_id,
                candidate.has_content(),
                len(candidate.text),
                len(candidate.video_sources),
                len(candidate.image_sources),
                candidate.source,
            )
            if candidate.has_content():
                if not candidate.source or candidate.source == "引用消息":
                    candidate.source = self._candidate_source_label(candidate)
                candidates.append(candidate)
        return candidates

    async def _extract_reply_remote_candidate(
        self,
        event: AstrMessageEvent,
        reply: Any,
        context: Dict[str, Any],
    ) -> SummaryCandidate:
        """Fetch quoted message details when local reply components are incomplete."""
        reply_id = str(getattr(reply, "id", "") or "").strip()
        candidate = SummaryCandidate(
            source=f"reply:{reply_id}" if reply_id else "引用消息",
            metadata=dict(context),
        )
        if not reply_id:
            return candidate
        payload = await self._call_platform_action_compat(
            event,
            "get_msg",
            reply_id,
        )
        if not payload:
            self._debug("引用消息远程回查为空: reply_id=%s", reply_id)
            return candidate
        self._debug(
            "引用消息远程回查命中: reply_id=%s payload=%s",
            reply_id,
            self._payload_shape(payload),
        )

        self._merge_candidate_content(
            candidate,
            await self._candidate_from_onebot_payload(event, payload),
        )
        self._debug(
            "引用消息远程内容: reply_id=%s text_chars=%d videos=%s images=%s",
            reply_id,
            len(candidate.text),
            candidate.video_sources,
            candidate.image_sources,
        )
        return candidate

    async def _candidate_from_onebot_payload(
        self,
        event: AstrMessageEvent,
        payload: Dict[str, Any],
        *,
        forward_depth: int = 0,
        forward_path: str = "",
    ) -> SummaryCandidate:
        """Extract a summary candidate from one OneBot payload, including forwards."""
        candidate = SummaryCandidate(metadata={})
        segments = self._onebot_payload_segments(payload)
        raw_message = str(payload.get("raw_message", "") or "")
        self._debug(
            "OneBot内容解析开始: depth=%d path=%r payload=%s segment_types=%s raw_chars=%d forward_hint=%s",
            forward_depth,
            forward_path,
            self._payload_shape(payload),
            self._segment_types(segments),
            len(raw_message),
            self._text_has_forward_hint(raw_message),
        )
        candidate.text = self._text_from_onebot_payload(payload)
        video_references = self._video_references_from_onebot_payload(payload)
        image_references = self._image_references_from_onebot_payload(payload)
        candidate.video_sources = self._sources_from_media_references(
            video_references
        )
        candidate.image_sources = self._sources_from_media_references(
            image_references
        )
        if len(candidate.video_sources) < len(video_references):
            remote_sources = await self._resolve_remote_file_sources(
                event,
                video_references,
                media_type="video",
            )
            for source in remote_sources:
                self._append_unique(candidate.video_sources, source)
        if len(candidate.image_sources) < len(image_references):
            remote_sources = await self._resolve_remote_file_sources(
                event,
                image_references,
                media_type="image",
            )
            for source in remote_sources:
                self._append_unique(candidate.image_sources, source)

        all_forward_ids = self._forward_ids_from_onebot_payload(payload)
        forward_ids = all_forward_ids if forward_depth < FORWARD_MAX_DEPTH else []
        skipped_forward_ids = [
            forward_id for forward_id in all_forward_ids
            if forward_id not in forward_ids
        ]
        self._debug(
            "OneBot内容解析结果: depth=%d path=%r text_chars=%d video_refs=%d image_refs=%d videos=%d images=%d forward_ids=%s skipped_forward_ids=%s",
            forward_depth,
            forward_path,
            len(candidate.text),
            len(video_references),
            len(image_references),
            len(candidate.video_sources),
            len(candidate.image_sources),
            forward_ids,
            skipped_forward_ids,
        )
        if skipped_forward_ids:
            self._debug(
                "OneBot内容包含子转发，按第一层平铺策略跳过: depth=%d path=%r ids=%s",
                forward_depth,
                forward_path,
                skipped_forward_ids,
            )
        for forward_index, forward_id in enumerate(forward_ids, start=1):
            forwarded = await self._resolve_forward_candidate(
                event,
                forward_id,
                forward_depth=forward_depth + 1,
                forward_path=self._forward_collection_path(
                    forward_path,
                    forward_index,
                ),
            )
            self._merge_candidate_content(candidate, forwarded)

        candidate.text = self._limit_forward_text(candidate.text)
        candidate.image_sources = candidate.image_sources[:FORWARD_MAX_IMAGE_SOURCES]
        candidate.video_sources = candidate.video_sources[:FORWARD_MAX_VIDEO_SOURCES]
        if candidate.has_content():
            candidate.source = self._candidate_source_label(candidate)
        self._debug(
            "OneBot候选完成: depth=%d path=%r has_content=%s text_chars=%d videos=%d images=%d source=%r",
            forward_depth,
            forward_path,
            candidate.has_content(),
            len(candidate.text),
            len(candidate.video_sources),
            len(candidate.image_sources),
            candidate.source,
        )
        return candidate

    async def _extract_forward_candidates_from_parts(
        self,
        event: AstrMessageEvent,
        parts: Iterable[Any],
        context: Dict[str, Any],
    ) -> SummaryCandidate:
        """Fetch forward collections referenced by local reply components."""
        candidate = SummaryCandidate(metadata=dict(context))
        forward_ids = self._forward_ids_from_components(parts)
        self._debug("本地组件转发识别: ids=%s", forward_ids)
        for forward_index, forward_id in enumerate(forward_ids, start=1):
            forwarded = await self._resolve_forward_candidate(
                event,
                forward_id,
                forward_depth=1,
                forward_path=self._forward_collection_path("", forward_index),
            )
            self._merge_candidate_content(candidate, forwarded)
        return candidate

    async def _resolve_forward_candidate(
        self,
        event: AstrMessageEvent,
        forward_id: str,
        *,
        forward_depth: int,
        forward_path: str = "",
    ) -> SummaryCandidate:
        """Fetch a OneBot merged-forward collection and flatten it for summary."""
        normalized_id = str(forward_id or "").strip()
        candidate = SummaryCandidate(
            source=f"forward:{normalized_id}" if normalized_id else "合并转发",
            metadata={"is_forward_collection": True},
        )
        collection_path = str(forward_path or "合并转发").strip() or "合并转发"
        if not normalized_id:
            self._debug("合并转发展开跳过: forward_id 为空 path=%s", collection_path)
            return candidate
        if forward_depth > FORWARD_MAX_DEPTH:
            candidate.text = f"（{collection_path} 层级过深，已停止继续展开）"
            self._debug(
                "合并转发展开停止: forward_id=%s depth=%d max_depth=%d path=%s",
                normalized_id,
                forward_depth,
                FORWARD_MAX_DEPTH,
                collection_path,
            )
            return candidate

        self._debug(
            "合并转发展开开始: forward_id=%s depth=%d path=%s",
            normalized_id,
            forward_depth,
            collection_path,
        )
        payload = await self._call_forward_msg_action(event, normalized_id)
        if payload is None:
            self._debug(
                "合并转发回查为空: forward_id=%s depth=%d path=%s",
                normalized_id,
                forward_depth,
                collection_path,
            )
            return candidate

        nodes = self._forward_nodes_from_payload(payload)
        self._debug(
            "合并转发回查命中: forward_id=%s depth=%d path=%s payload=%s nodes=%d",
            normalized_id,
            forward_depth,
            collection_path,
            self._payload_shape(payload),
            len(nodes),
        )
        if not nodes and isinstance(payload, dict):
            nested = await self._candidate_from_onebot_payload(
                event,
                payload,
                forward_depth=forward_depth,
                forward_path=collection_path,
            )
            nested.source = candidate.source
            nested.metadata["is_forward_collection"] = True
            self._debug(
                "合并转发无节点，按普通payload解析: forward_id=%s text_chars=%d videos=%d images=%d",
                normalized_id,
                len(nested.text),
                len(nested.video_sources),
                len(nested.image_sources),
            )
            return nested

        text_parts: List[str] = []
        max_nodes = self._max_forward_nodes()
        selected_nodes = nodes[:max_nodes]
        for index, node in enumerate(selected_nodes, start=1):
            node_payload = self._onebot_payload_from_forward_node(node)
            node_path = self._forward_node_path(collection_path, index)
            node_candidate = await self._candidate_from_onebot_payload(
                event,
                node_payload,
                forward_depth=forward_depth,
                forward_path=node_path,
            )
            self._debug(
                "合并转发节点解析: path=%s payload=%s text_chars=%d videos=%d images=%d",
                node_path,
                self._payload_shape(node_payload),
                len(node_candidate.text),
                len(node_candidate.video_sources),
                len(node_candidate.image_sources),
            )
            node_text = node_candidate.text.strip()
            if node_text:
                text_parts.append(
                    f"{self._forward_node_label(node, node_path)}：\n{node_text}"
                )
            for video_source in node_candidate.video_sources:
                self._append_unique(candidate.video_sources, video_source)
            for image_source in node_candidate.image_sources:
                self._append_unique(candidate.image_sources, image_source)

        if len(nodes) > len(selected_nodes):
            text_parts.append(
                f"（{collection_path} 共 {len(nodes)} 条，已截取前 {len(selected_nodes)} 条）"
            )
        candidate.text = self._limit_forward_text(
            self._join_text_fragments(text_parts)
        )
        candidate.video_sources = candidate.video_sources[:FORWARD_MAX_VIDEO_SOURCES]
        candidate.image_sources = candidate.image_sources[:FORWARD_MAX_IMAGE_SOURCES]
        candidate.metadata.update({
            "forward_id": normalized_id,
            "forward_depth": forward_depth,
            "forward_path": collection_path,
            "forward_node_count": len(nodes),
            "forward_nodes_used": len(selected_nodes),
        })
        self._debug(
            "合并转发展开完成: forward_id=%s nodes=%d used=%d text_chars=%d videos=%d images=%d",
            normalized_id,
            len(nodes),
            len(selected_nodes),
            len(candidate.text),
            len(candidate.video_sources),
            len(candidate.image_sources),
        )
        return candidate

    async def _call_forward_msg_action(
        self,
        event: AstrMessageEvent,
        forward_id: str,
    ) -> Optional[Any]:
        """Call compatible get_forward_msg action variants for one forward id."""
        params_list: List[Dict[str, Any]] = [
            {"id": forward_id},
            {"message_id": forward_id},
            {"forward_id": forward_id},
        ]
        if str(forward_id).isdigit():
            int_id = int(forward_id)
            params_list.extend([
                {"id": int_id},
                {"message_id": int_id},
                {"forward_id": int_id},
            ])
        return await self._call_platform_action_variants(
            event,
            "get_forward_msg",
            params_list,
        )

    def _extract_candidate_from_parts(
        self,
        parts: Iterable[Any],
        context: Dict[str, Any],
    ) -> SummaryCandidate:
        """Extract text, image, and video content from AstrBot message components."""
        text_parts: List[str] = []
        candidate = SummaryCandidate(metadata=dict(context))
        for comp in self._walk_components(parts):
            text = self._text_from_component(comp)
            if text:
                text_parts.append(text)
            video_source = self._video_source_from_component(comp)
            if video_source:
                self._append_unique(candidate.video_sources, video_source)
            image_source = self._image_source_from_component(comp)
            if image_source:
                self._append_unique(candidate.image_sources, image_source)

        candidate.text = self._join_text_fragments(text_parts)
        if candidate.has_content():
            candidate.source = self._candidate_source_label(candidate)
        return candidate

    def _merge_candidate_content(
        self,
        target: SummaryCandidate,
        source: Optional[SummaryCandidate],
    ) -> None:
        if source is None:
            return
        target.text = self._join_text_fragments([target.text, source.text])
        for video_source in source.video_sources:
            self._append_unique(target.video_sources, video_source)
        for image_source in source.image_sources:
            self._append_unique(target.image_sources, image_source)
        target.metadata.update(source.metadata)
        if (
            source.source
            and source.source != "引用消息"
            and (not target.source or target.source == "引用消息")
        ):
            target.source = source.source

    @staticmethod
    def _candidate_source_label(candidate: SummaryCandidate) -> str:
        for source in candidate.video_sources + candidate.image_sources:
            if source:
                return source
        text = candidate.text.strip()
        if text:
            return f"quoted-text:{text[:80]}"
        return "引用消息"

    @staticmethod
    def _append_unique(values: List[str], value: str) -> None:
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)

    @staticmethod
    def _join_text_fragments(values: Iterable[Any]) -> str:
        fragments: List[str] = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            normalized = re.sub(r"\s+", " ", text)
            if normalized in seen:
                continue
            seen.add(normalized)
            fragments.append(text)
        return "\n".join(fragments).strip()

    @staticmethod
    def _max_forward_nodes() -> int:
        return max(1, FORWARD_MAX_NODES)

    @staticmethod
    def _forward_collection_path(parent_path: str, index: int) -> str:
        return f"合并转发[{index}]"

    @staticmethod
    def _forward_node_path(collection_path: str, index: int) -> str:
        collection = str(collection_path or "").strip() or "合并转发"
        return f"{collection} > 消息[{index}]"

    def _limit_forward_text(self, text: str) -> str:
        value = str(text or "").strip()
        limit = max(
            2000,
            int(getattr(self.config, "max_transcript_chars", 20000) or 20000),
        )
        if len(value) <= limit:
            return value
        self._debug(
            "合并转发文本截断: original_chars=%d limit=%d",
            len(value),
            limit,
        )
        return value[:limit].rstrip() + "\n（合并转发文本因长度限制已截断）"

    @classmethod
    def _forward_ids_from_components(cls, parts: Iterable[Any]) -> List[str]:
        ids: List[str] = []
        for comp in cls._walk_components(parts):
            forward_id = cls._forward_id_from_component(comp)
            if forward_id and forward_id not in ids:
                ids.append(forward_id)
        return ids

    @classmethod
    def _forward_id_from_component(cls, comp: Any) -> str:
        if isinstance(comp, dict):
            return cls._forward_id_from_onebot_segment(comp)
        class_name = comp.__class__.__name__.lower()
        data = getattr(comp, "data", None)
        is_forward_like = (
            "forward" in class_name
            or "json" in class_name
            or "xml" in class_name
        )
        if isinstance(data, dict):
            segment_type = str(data.get("type", "") or "").lower()
            is_forward_like = is_forward_like or segment_type in {
                "forward",
                "forward_msg",
                "merged_forward",
                "json",
                "xml",
            }
        if not is_forward_like:
            return ""
        for attr in ("id", "message_id", "forward_id", "file_id"):
            value = str(getattr(comp, attr, "") or "").strip()
            if value:
                return value
        if isinstance(data, dict):
            forward_id = cls._first_mapping_text(
                data,
                ("id", "message_id", "forward_id", "file_id"),
            )
            if forward_id:
                return forward_id
            for key in ("data", "content", "message", "text"):
                value = data.get(key)
                if isinstance(value, str):
                    forward_id = cls._forward_id_from_card_text(value)
                    if forward_id:
                        return forward_id
        for attr in ("data", "content", "message", "raw_message", "text"):
            value = getattr(comp, attr, None)
            if isinstance(value, str):
                forward_id = cls._forward_id_from_card_text(value)
                if forward_id:
                    return forward_id
        return ""

    @classmethod
    def _forward_ids_from_onebot_payload(cls, payload: Dict[str, Any]) -> List[str]:
        ids: List[str] = []
        segments = cls._onebot_payload_segments(payload)
        for segment in cls._walk_onebot_segments(segments):
            forward_id = cls._forward_id_from_onebot_segment(segment)
            if forward_id and forward_id not in ids:
                ids.append(forward_id)
        raw_message = str(payload.get("raw_message", "") or "")
        for segment in cls._raw_cq_segments(raw_message):
            forward_id = cls._forward_id_from_onebot_segment(segment)
            if forward_id and forward_id not in ids:
                ids.append(forward_id)
        forward_id = cls._forward_id_from_card_text(raw_message)
        if forward_id and forward_id not in ids:
            ids.append(forward_id)
        return ids

    @staticmethod
    def _forward_id_from_onebot_segment(segment: Any) -> str:
        if not isinstance(segment, dict):
            return ""
        segment_type = str(segment.get("type", "") or "").lower()
        if segment_type not in {"forward", "forward_msg", "merged_forward"}:
            if segment_type in {"json", "xml"}:
                data = segment.get("data")
                if not isinstance(data, dict):
                    return ""
                for key in ("data", "content", "message", "text"):
                    value = data.get(key)
                    if isinstance(value, str):
                        forward_id = AISummaryPlugin._forward_id_from_card_text(value)
                        if forward_id:
                            return forward_id
            return ""
        data = segment.get("data")
        if not isinstance(data, dict):
            return ""
        return AISummaryPlugin._first_mapping_text(
            data,
            ("id", "message_id", "forward_id", "file_id"),
        )

    @staticmethod
    def _forward_id_from_card_text(text: str) -> str:
        """Extract merged-forward resid from QQ json/xml card payloads."""
        value = html.unescape(str(text or "")).strip()
        if not value:
            return ""
        for payload in AISummaryPlugin._json_objects_from_text(value):
            forward_id = AISummaryPlugin._forward_id_from_json_card(payload)
            if forward_id:
                return forward_id
        forward_id = AISummaryPlugin._forward_id_from_xml_card(value)
        return forward_id

    @staticmethod
    def _json_objects_from_text(text: str) -> List[Any]:
        value = str(text or "").strip()
        if not value:
            return []
        candidates = [value]
        match = re.search(r"(\{.*\})", value, re.S)
        if match and match.group(1) != value:
            candidates.append(match.group(1))
        objects: List[Any] = []
        for candidate in candidates:
            try:
                objects.append(json.loads(candidate))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return objects

    @staticmethod
    def _forward_id_from_json_card(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        if not AISummaryPlugin._json_card_looks_like_forward(payload):
            return ""
        for value in AISummaryPlugin._walk_json_values(payload):
            if not isinstance(value, dict):
                continue
            forward_id = AISummaryPlugin._first_mapping_text(
                value,
                FORWARD_CARD_ID_KEYS,
            )
            if forward_id:
                return forward_id
        return ""

    @staticmethod
    def _json_card_looks_like_forward(payload: Dict[str, Any]) -> bool:
        text = json.dumps(payload, ensure_ascii=False)
        lowered = text.lower()
        markers = (
            "com.tencent.multimsg",
            "multimsg",
            "forward",
            "聊天记录",
            "转发消息",
            "合并转发",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _walk_json_values(value: Any) -> Iterable[Any]:
        yield value
        if isinstance(value, dict):
            for item in value.values():
                yield from AISummaryPlugin._walk_json_values(item)
        elif isinstance(value, list):
            for item in value:
                yield from AISummaryPlugin._walk_json_values(item)

    @staticmethod
    def _forward_id_from_xml_card(text: str) -> str:
        value = str(text or "")
        lowered = value.lower()
        if not any(
            marker in lowered
            for marker in ("multimsg", "forward", "聊天记录", "转发消息", "合并转发")
        ):
            return ""
        for key in FORWARD_CARD_ID_KEYS:
            patterns = (
                rf'{re.escape(key)}\s*=\s*"([^"]+)"',
                rf"{re.escape(key)}\s*=\s*'([^']+)'",
                rf"<{re.escape(key)}>\s*([^<]+)\s*</{re.escape(key)}>",
            )
            for pattern in patterns:
                match = re.search(pattern, value, re.I)
                if match:
                    return html.unescape(match.group(1)).strip()
        return ""

    @staticmethod
    def _forward_nodes_from_payload(payload: Any) -> List[Any]:
        """Return forward nodes from common OneBot get_forward_msg shapes."""
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        for key in ("messages", "message", "nodes", "news", "content"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        data = payload.get("data")
        if isinstance(data, dict):
            return AISummaryPlugin._forward_nodes_from_payload(data)
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _onebot_payload_from_forward_node(node: Any) -> Dict[str, Any]:
        if not isinstance(node, dict):
            return {"message": [node] if isinstance(node, str) else []}
        for key in ("content", "message", "messages"):
            value = node.get(key)
            if isinstance(value, list):
                return {"message": value}
            if isinstance(value, str):
                return {"message": value, "raw_message": value}
        data = node.get("data")
        if isinstance(data, dict):
            for key in ("content", "message", "messages"):
                value = data.get(key)
                if isinstance(value, list):
                    return {"message": value}
                if isinstance(value, str):
                    return {"message": value, "raw_message": value}
        if str(node.get("type", "") or "").strip():
            return {"message": [node]}
        return {"message": []}

    @staticmethod
    def _forward_node_label(node: Any, node_path: str) -> str:
        path = str(node_path or "").strip() or "合并转发 > 消息"
        if not isinstance(node, dict):
            return path
        data = node.get("data")
        data = data if isinstance(data, dict) else {}
        sender = node.get("sender") or data.get("sender")
        name = ""
        if isinstance(sender, dict):
            name = AISummaryPlugin._first_mapping_text(
                sender,
                ("card", "nickname", "name", "user_id", "uin"),
            )
        if not name:
            name = AISummaryPlugin._first_mapping_text(
                node,
                ("nickname", "name", "sender_name", "user_id", "uin"),
            )
        if not name and data:
            name = AISummaryPlugin._first_mapping_text(
                data,
                ("nickname", "name", "sender_name", "user_id", "uin"),
            )
        return f"{path} {name}" if name else path

    def _safe_get_messages(self, event: AstrMessageEvent) -> List[Any]:
        try:
            return list(event.get_messages() or [])
        except Exception as exc:
            self._debug("读取消息组件失败: error=%s", exc)
            return []

    async def _call_platform_action_compat(
        self,
        event: AstrMessageEvent,
        action: str,
        message_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Call platform actions with common message id parameter variants."""
        params_list: List[Dict[str, Any]] = [
            {"message_id": message_id},
            {"id": message_id},
        ]
        if message_id.isdigit():
            int_id = int(message_id)
            params_list.extend([{"message_id": int_id}, {"id": int_id}])

        result = await self._call_platform_action_variants(
            event,
            action,
            params_list,
        )
        return result if isinstance(result, dict) else None

    async def _call_platform_action_variants(
        self,
        event: AstrMessageEvent,
        action: str,
        params_list: Iterable[Dict[str, Any]],
    ) -> Optional[Any]:
        """Try compatible platform action payloads and return the first result."""
        call_action = self._resolve_call_action(event)
        if not call_action:
            self._debug("平台不支持 action 回查: %s", action)
            return None

        last_error: Optional[Exception] = None
        for params in params_list:
            try:
                self._debug("平台 action 调用: action=%s params=%s", action, params)
                result = call_action(action, **params)
                if asyncio.iscoroutine(result):
                    result = await result
                self._debug(
                    "平台 action 返回: action=%s params=%s result=%s",
                    action,
                    params,
                    self._payload_shape(result),
                )
                if result is None:
                    self._debug(
                        "平台 action 空返回，尝试下一组参数: action=%s params=%s",
                        action,
                        params,
                    )
                    continue
                if isinstance(result, dict):
                    data = result.get("data")
                    normalized = data if data is not None else result
                    self._debug(
                        "平台 action 命中: action=%s params=%s normalized=%s",
                        action,
                        params,
                        self._payload_shape(normalized),
                    )
                    return normalized
                self._debug(
                    "平台 action 命中: action=%s params=%s normalized=%s",
                    action,
                    params,
                    self._payload_shape(result),
                )
                return result
            except Exception as exc:
                last_error = exc
                self._debug(
                    "平台 action 失败: action=%s params=%s error=%s",
                    action,
                    params,
                    exc,
                )
        if last_error:
            logger.warning(
                f"AI 总结平台 action 失败: action={action}, 错误: {last_error}"
            )
        return None

    @staticmethod
    def _resolve_call_action(event: AstrMessageEvent) -> Any:
        bot = getattr(event, "bot", None)
        api = getattr(bot, "api", None)
        call_action = getattr(api, "call_action", None)
        if callable(call_action):
            return call_action
        call_action = getattr(bot, "call_action", None)
        return call_action if callable(call_action) else None

    @staticmethod
    def _walk_components(parts: Iterable[Any]) -> Iterable[Any]:
        """Yield message components recursively from common nested attributes."""
        for comp in parts:
            yield comp
            if isinstance(comp, dict):
                data = comp.get("data")
                if isinstance(data, dict):
                    for key in ("chain", "content", "message", "messages", "nodes"):
                        nested = data.get(key)
                        if isinstance(nested, list):
                            yield from AISummaryPlugin._walk_components(nested)
                for key in ("chain", "content", "message", "messages", "nodes"):
                    nested = comp.get(key)
                    if isinstance(nested, list):
                        yield from AISummaryPlugin._walk_components(nested)
                continue
            for attr in ("chain", "content", "nodes"):
                nested = getattr(comp, attr, None)
                if isinstance(nested, list):
                    yield from AISummaryPlugin._walk_components(nested)
            data = getattr(comp, "data", None)
            if isinstance(data, dict):
                for key in ("chain", "content", "nodes"):
                    nested = data.get(key)
                    if isinstance(nested, list):
                        yield from AISummaryPlugin._walk_components(nested)

    @staticmethod
    def _text_from_component(comp: Any) -> str:
        """Extract plain text from text-like AstrBot components."""
        if isinstance(comp, str):
            return AISummaryPlugin._plain_text_from_raw_message(comp)
        if isinstance(comp, dict):
            return AISummaryPlugin._text_from_onebot_segment(comp)
        class_name = comp.__class__.__name__.lower()
        if "reply" in class_name or "image" in class_name or "video" in class_name:
            return ""
        data = getattr(comp, "data", None)
        is_text_like = (
            "plain" in class_name
            or class_name == "text"
            or "text" in class_name
        )
        if isinstance(data, dict):
            segment_type = str(data.get("type", "") or "").lower()
            is_text_like = is_text_like or segment_type in {"text", "plain"}
        if not is_text_like:
            return ""

        for attr in ("text", "message_str", "message", "content"):
            value = getattr(comp, attr, None)
            if isinstance(value, str):
                text = AISummaryPlugin._plain_text_from_raw_message(value)
                if text:
                    return text
        if isinstance(data, dict):
            for key in ("text", "content", "message", "raw_message"):
                value = data.get(key)
                if isinstance(value, str):
                    text = AISummaryPlugin._plain_text_from_raw_message(value)
                    if text:
                        return text
        return ""

    @staticmethod
    def _text_from_onebot_payload(payload: Dict[str, Any]) -> str:
        """Extract visible plain text from OneBot message payloads."""
        texts: List[str] = []
        for segment in AISummaryPlugin._walk_onebot_segments(
            AISummaryPlugin._onebot_payload_segments(payload)
        ):
            text = AISummaryPlugin._text_from_onebot_segment(segment)
            if text:
                texts.append(text)

        raw_message = str(payload.get("raw_message", "") or "")
        if raw_message:
            texts.append(AISummaryPlugin._plain_text_from_raw_message(raw_message))
        message = payload.get("message")
        if isinstance(message, str):
            texts.append(AISummaryPlugin._plain_text_from_raw_message(message))
        for key in ("text", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                texts.append(AISummaryPlugin._plain_text_from_raw_message(value))
        return AISummaryPlugin._join_text_fragments(texts)

    @staticmethod
    def _text_from_onebot_segment(segment: Any) -> str:
        if isinstance(segment, str):
            return AISummaryPlugin._plain_text_from_raw_message(segment)
        if not isinstance(segment, dict):
            return ""
        segment_type = str(segment.get("type", "") or "").lower()
        data = segment.get("data")
        if not isinstance(data, dict):
            return ""
        if segment_type in {"text", "plain"}:
            for key in ("text", "content", "message"):
                value = data.get(key)
                if isinstance(value, str):
                    return AISummaryPlugin._plain_text_from_raw_message(value)
        if segment_type == "node":
            value = data.get("content")
            if isinstance(value, str):
                return AISummaryPlugin._plain_text_from_raw_message(value)
        return ""

    @staticmethod
    def _plain_text_from_raw_message(raw_message: str) -> str:
        """Strip CQ media codes while preserving visible text."""
        if not raw_message:
            return ""
        text = re.sub(r"\[CQ:[^\]]+\]", "", str(raw_message))
        text = html.unescape(text)
        return re.sub(r"[ \t\f\v]+", " ", text).strip()

    @staticmethod
    def _video_sources_from_onebot_payload(payload: Dict[str, Any]) -> List[str]:
        return AISummaryPlugin._sources_from_media_references(
            AISummaryPlugin._video_references_from_onebot_payload(payload)
        )

    @staticmethod
    def _video_references_from_onebot_payload(
        payload: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Collect video and video-like file references from OneBot payloads."""
        segments = AISummaryPlugin._onebot_payload_segments(payload)
        references: List[Dict[str, str]] = []
        for segment in AISummaryPlugin._walk_onebot_segments(segments):
            if not isinstance(segment, dict):
                continue
            reference = AISummaryPlugin._video_reference_from_onebot_segment(
                segment
            )
            AISummaryPlugin._append_media_reference(references, reference)

        raw_message = str(payload.get("raw_message", "") or "")
        for segment in AISummaryPlugin._raw_cq_segments(raw_message):
            reference = AISummaryPlugin._video_reference_from_onebot_segment(
                segment
            )
            AISummaryPlugin._append_media_reference(references, reference)
        return references

    @staticmethod
    def _image_references_from_onebot_payload(
        payload: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Collect image references from OneBot payloads."""
        segments = AISummaryPlugin._onebot_payload_segments(payload)
        references: List[Dict[str, str]] = []
        for segment in AISummaryPlugin._walk_onebot_segments(segments):
            if not isinstance(segment, dict):
                continue
            reference = AISummaryPlugin._image_reference_from_onebot_segment(
                segment
            )
            AISummaryPlugin._append_media_reference(references, reference)

        raw_message = str(payload.get("raw_message", "") or "")
        for segment in AISummaryPlugin._raw_cq_segments(raw_message):
            reference = AISummaryPlugin._image_reference_from_onebot_segment(
                segment
            )
            AISummaryPlugin._append_media_reference(references, reference)
        return references

    @staticmethod
    def _sources_from_media_references(
        references: Iterable[Dict[str, str]],
    ) -> List[str]:
        sources: List[str] = []
        for reference in references:
            source = str(reference.get("source", "") or "").strip()
            if source and source not in sources:
                sources.append(source)
        return sources

    @staticmethod
    def _append_media_reference(
        references: List[Dict[str, str]],
        reference: Dict[str, str],
    ) -> None:
        if not reference:
            return
        if not reference.get("source") and not reference.get("file_id"):
            return
        key = (
            reference.get("source", ""),
            reference.get("file_id", ""),
            reference.get("file_name", ""),
            reference.get("segment_type", ""),
        )
        for existing in references:
            existing_key = (
                existing.get("source", ""),
                existing.get("file_id", ""),
                existing.get("file_name", ""),
                existing.get("segment_type", ""),
            )
            if existing_key == key:
                return
        references.append(reference)

    @staticmethod
    def _append_video_reference(
        references: List[Dict[str, str]],
        reference: Dict[str, str],
    ) -> None:
        AISummaryPlugin._append_media_reference(references, reference)

    @staticmethod
    def _video_reference_from_onebot_segment(
        segment: Dict[str, Any],
    ) -> Dict[str, str]:
        """Convert a OneBot video or file segment into a normalized reference."""
        segment_type = str(segment.get("type", "") or "").lower()
        if segment_type not in {"video", "file"}:
            return {}
        data = segment.get("data")
        if not isinstance(data, dict):
            return {}

        source = AISummaryPlugin._source_from_mapping(data)
        file_id = AISummaryPlugin._first_mapping_text(
            data,
            ("file_id", "fileid", "id"),
        )
        file_name = AISummaryPlugin._first_mapping_text(
            data,
            ("file_name", "name", "file"),
        )
        is_video = segment_type == "video" or AISummaryPlugin._is_video_filename(
            file_name
        )
        if not is_video and source:
            is_video = AISummaryPlugin._is_video_filename(source)
        if not is_video:
            return {}
        return {
            "source": source,
            "file_id": file_id,
            "file_name": file_name,
            "segment_type": segment_type,
        }

    @staticmethod
    def _image_reference_from_onebot_segment(
        segment: Dict[str, Any],
    ) -> Dict[str, str]:
        """Convert a OneBot image or image-like file segment into a reference."""
        segment_type = str(segment.get("type", "") or "").lower()
        if segment_type not in {"image", "file"}:
            return {}
        data = segment.get("data")
        if not isinstance(data, dict):
            return {}

        source = AISummaryPlugin._image_source_from_mapping(data)
        file_id = AISummaryPlugin._first_mapping_text(
            data,
            ("file_id", "fileid", "id", "file"),
        )
        file_name = AISummaryPlugin._first_mapping_text(
            data,
            ("file_name", "name", "filename", "file"),
        )
        is_image = segment_type == "image" or AISummaryPlugin._is_image_filename(
            file_name
        )
        if not is_image and source:
            is_image = AISummaryPlugin._is_image_filename(source)
        if not is_image:
            return {}
        return {
            "source": source,
            "file_id": file_id,
            "file_name": file_name,
            "segment_type": segment_type,
        }

    @staticmethod
    def _raw_cq_segments(raw_message: str) -> Iterable[Dict[str, Any]]:
        """Parse raw CQ code segments into OneBot-like segment mappings."""
        if not raw_message:
            return []
        segments: List[Dict[str, Any]] = []
        for match in re.finditer(r"\[CQ:([^,\]]+)(?:,([^\]]*))?\]", raw_message):
            segment_type = html.unescape(match.group(1)).strip()
            data: Dict[str, str] = {}
            params = match.group(2) or ""
            for item in params.split(","):
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                key = html.unescape(key).strip()
                value = html.unescape(value).strip()
                if key:
                    data[key] = value
            segments.append({"type": segment_type, "data": data})
        return segments

    @staticmethod
    def _onebot_payload_segments(payload: Dict[str, Any]) -> List[Any]:
        segments = payload.get("message") or payload.get("messages")
        if isinstance(segments, list):
            return segments
        if isinstance(segments, str):
            return [segments]
        return []

    @staticmethod
    def _walk_onebot_segments(segments: Iterable[Any]) -> Iterable[Any]:
        for segment in segments:
            yield segment
            if not isinstance(segment, dict):
                continue
            data = segment.get("data")
            if isinstance(data, dict):
                for key in ("content", "message", "messages", "nodes"):
                    nested = data.get(key)
                    if isinstance(nested, list):
                        yield from AISummaryPlugin._walk_onebot_segments(nested)

    @staticmethod
    def _video_source_from_component(comp: Any) -> str:
        """Resolve a usable video URL from an AstrBot component."""
        if isinstance(comp, dict):
            return AISummaryPlugin._video_reference_from_onebot_segment(comp).get(
                "source",
                "",
            )
        class_name = comp.__class__.__name__.lower()
        if (
            "video" not in class_name and
            not AISummaryPlugin._component_looks_like_video_file(comp)
        ):
            return ""

        data = getattr(comp, "data", None)
        value = AISummaryPlugin._source_from_mapping(data)
        if value:
            return value

        for attr in ("url", "path", "file_path", "src", "file", "file_"):
            value = getattr(comp, attr, None)
            source = AISummaryPlugin._source_from_value(value)
            if source:
                return source
        return ""

    @staticmethod
    def _image_source_from_component(comp: Any) -> str:
        """Resolve a usable image source from an AstrBot component."""
        if isinstance(comp, dict):
            return AISummaryPlugin._image_reference_from_onebot_segment(comp).get(
                "source",
                "",
            )
        class_name = comp.__class__.__name__.lower()
        if (
            "image" not in class_name and
            not AISummaryPlugin._component_looks_like_image_file(comp)
        ):
            return ""

        data = getattr(comp, "data", None)
        value = AISummaryPlugin._image_source_from_mapping(data)
        if value:
            return value

        for attr in ("url", "path", "file_path", "src", "file", "file_"):
            value = getattr(comp, attr, None)
            source = AISummaryPlugin._image_source_from_value(value)
            if source:
                return source
        return ""

    @staticmethod
    def _source_from_mapping(value: Any) -> str:
        return AISummaryPlugin._media_source_from_mapping(value, "video")

    @staticmethod
    def _source_from_value(value: Any) -> str:
        return AISummaryPlugin._media_source_from_value(value, "video")

    @staticmethod
    def _image_source_from_mapping(value: Any) -> str:
        return AISummaryPlugin._media_source_from_mapping(value, "image")

    @staticmethod
    def _image_source_from_value(value: Any) -> str:
        return AISummaryPlugin._media_source_from_value(value, "image")

    @staticmethod
    def _media_source_from_mapping(value: Any, media_type: str) -> str:
        if not isinstance(value, dict):
            return ""
        common_keys = ("url", "src", "path", "file_path", "file", "file_")
        media_keys = (
            ("video_url", "video")
            if media_type == "video"
            else ("image_url", "image", "pic", "photo")
        )
        for key in (*common_keys, *media_keys):
            item = value.get(key)
            source = AISummaryPlugin._media_source_from_value(item, media_type)
            if source:
                return source
        return ""

    @staticmethod
    def _media_source_from_value(value: Any, media_type: str) -> str:
        if isinstance(value, str):
            source = AISummaryPlugin._strip_media_prefixes(value.strip())
            if media_type == "image" and source.startswith("base64://"):
                source = "data:image/jpeg;base64," + source[len("base64://"):]
            if media_type == "image":
                return source if AISummaryPlugin._is_usable_image_source(source) else ""
            return source if AISummaryPlugin._is_usable_video_source(source) else ""
        if isinstance(value, dict):
            return AISummaryPlugin._media_source_from_mapping(value, media_type)
        return ""

    @staticmethod
    def _first_mapping_text(
        value: Dict[str, Any],
        keys: Iterable[str],
    ) -> str:
        for key in keys:
            item = value.get(key)
            if item is None:
                continue
            text = str(item).strip()
            if text:
                return text
        return ""

    @staticmethod
    def _component_looks_like_video_file(comp: Any) -> bool:
        data = getattr(comp, "data", None)
        if isinstance(data, dict):
            for key in ("file_name", "name", "file", "path", "file_path", "url"):
                if AISummaryPlugin._is_video_filename(str(data.get(key, "") or "")):
                    return True
        for attr in ("name", "file", "file_", "path", "file_path", "url"):
            if AISummaryPlugin._is_video_filename(str(getattr(comp, attr, "") or "")):
                return True
        return False

    @staticmethod
    def _component_looks_like_image_file(comp: Any) -> bool:
        data = getattr(comp, "data", None)
        if isinstance(data, dict):
            for key in ("file_name", "name", "file", "path", "file_path", "url"):
                if AISummaryPlugin._is_image_filename(str(data.get(key, "") or "")):
                    return True
        for attr in ("name", "file", "file_", "path", "file_path", "url"):
            if AISummaryPlugin._is_image_filename(str(getattr(comp, attr, "") or "")):
                return True
        return False

    @staticmethod
    def _is_video_filename(value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if text.lower().startswith(("http://", "https://")):
            text = urlparse(text).path
        return Path(text).suffix.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def _is_image_filename(value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if lowered.startswith("data:image/"):
            return True
        if lowered.startswith(("http://", "https://")):
            text = urlparse(text).path
        local_path = AISummaryPlugin._local_path_from_source(text)
        suffix = Path(local_path or text).suffix.lower()
        return suffix in IMAGE_EXTENSIONS

    @staticmethod
    def _is_usable_video_source(source: str) -> bool:
        if not source:
            return False
        return source.lower().startswith(("http://", "https://"))

    @staticmethod
    def _is_usable_image_source(source: str) -> bool:
        if not source:
            return False
        lowered = source.lower()
        if lowered.startswith(("http://", "https://", "data:image/")):
            return True
        local_path = AISummaryPlugin._local_path_from_source(source)
        return bool(local_path and os.path.isfile(local_path))

    async def _resolve_remote_file_sources(
        self,
        event: AstrMessageEvent,
        references: Iterable[Dict[str, str]],
        media_type: str = "video",
    ) -> List[str]:
        """Resolve file identifiers into downloadable media URLs."""
        sources: List[str] = []
        for reference in references:
            if reference.get("source"):
                continue
            file_id = str(reference.get("file_id", "") or "").strip()
            if not file_id:
                continue
            for source in await self._resolve_one_remote_file_source(
                event,
                file_id,
                media_type,
            ):
                if source not in sources:
                    sources.append(source)
        return sources

    async def _resolve_one_remote_file_source(
        self,
        event: AstrMessageEvent,
        file_id: str,
        media_type: str = "video",
    ) -> List[str]:
        """Resolve a single remote file id through private or group APIs."""
        sources: List[str] = []
        if media_type == "image":
            for params in ({"file": file_id}, {"file_id": file_id}):
                payload = await self._call_platform_action_variants(
                    event,
                    "get_image",
                    [params],
                )
                source = self._source_from_action_payload(payload, media_type)
                if source:
                    return [source]

        payload = await self._call_platform_action_variants(
            event,
            "get_file",
            [{"file_id": file_id}],
        )
        source = self._source_from_action_payload(payload, media_type)
        if source:
            return [source]

        if self._event_is_private_chat(event):
            payload = await self._call_platform_action_variants(
                event,
                "get_private_file_url",
                [{"file_id": file_id}],
            )
            source = self._source_from_action_payload(payload, media_type)
            if source:
                sources.append(source)
            return sources

        group_id = self._event_group_id(event)
        if group_id:
            params_list: List[Dict[str, Any]] = [
                {"group_id": group_id, "file_id": file_id}
            ]
            if str(group_id).isdigit():
                params_list.append({"group_id": int(group_id), "file_id": file_id})
            payload = await self._call_platform_action_variants(
                event,
                "get_group_file_url",
                params_list,
            )
            source = self._source_from_action_payload(payload, media_type)
            if source:
                sources.append(source)
        return sources

    @staticmethod
    def _source_from_action_payload(
        payload: Any,
        media_type: str = "video",
    ) -> str:
        source = AISummaryPlugin._media_source_from_value(payload, media_type)
        if source:
            return source
        if isinstance(payload, dict):
            data = payload.get("data")
            source = AISummaryPlugin._media_source_from_value(data, media_type)
            if source:
                return source
        return ""

    @staticmethod
    def _event_is_private_chat(event: AstrMessageEvent) -> bool:
        is_private_chat = getattr(event, "is_private_chat", None)
        if not callable(is_private_chat):
            return False
        try:
            return bool(is_private_chat())
        except Exception:
            return False

    @staticmethod
    def _event_group_id(event: AstrMessageEvent) -> str:
        get_group_id = getattr(event, "get_group_id", None)
        if not callable(get_group_id):
            return ""
        try:
            return str(get_group_id() or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _strip_media_prefixes(value: str) -> str:
        text = value.strip()
        for prefix in ("range:", "proxy:", "cache:"):
            if text.startswith(prefix):
                text = text[len(prefix):]
        return text

    @staticmethod
    def _local_path_from_source(source: str) -> str:
        text = str(source or "").strip()
        if not text:
            return ""
        lowered = text.lower()
        if lowered.startswith("file://"):
            parsed = urlparse(text)
            path = unquote(parsed.path or "")
            if parsed.netloc and not path:
                path = unquote(parsed.netloc)
            elif parsed.netloc and path:
                path = f"//{parsed.netloc}{path}"
            if re.match(r"^/[A-Za-z]:/", path):
                path = path[1:]
            return path
        if lowered.startswith("file:"):
            return unquote(text[5:])
        return text

    @staticmethod
    def _suffix_for_url(url: str) -> str:
        suffix = Path(urlparse(url).path).suffix.lower()
        return suffix if suffix in VIDEO_EXTENSIONS else ".mp4"

    @staticmethod
    def _suffix_for_image_url(url: str) -> str:
        suffix = Path(urlparse(url).path).suffix.lower()
        return suffix if suffix in IMAGE_EXTENSIONS else ".jpg"

    @staticmethod
    def _dedupe_candidates(
        candidates: List[SummaryCandidate],
    ) -> List[SummaryCandidate]:
        """Preserve the first candidate for each distinct quoted content item."""
        seen = set()
        deduped: List[SummaryCandidate] = []
        for candidate in candidates:
            key = AISummaryPlugin._candidate_dedupe_key(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _candidate_dedupe_key(candidate: SummaryCandidate) -> str:
        source = str(candidate.source or "").strip()
        if source and source != "引用消息":
            return source
        media = "|".join(candidate.video_sources + candidate.image_sources)
        if media:
            return media
        return re.sub(r"\s+", " ", candidate.text.strip())[:500]

    def _attach_user_hint_to_candidates(
        self,
        candidates: List[SummaryCandidate],
        user_hint: str,
        event: Optional[AstrMessageEvent] = None,
        summary_style: str = "auto",
    ) -> None:
        """Attach user-provided summary hints and event context to candidates."""
        context = self._candidate_context(user_hint, event, summary_style)
        if not context:
            return
        for candidate in candidates:
            candidate.metadata.update(context)

    def _candidate_context(
        self,
        user_hint: str,
        event: Optional[AstrMessageEvent] = None,
        summary_style: str = "auto",
    ) -> Dict[str, Any]:
        context = self._event_context(event)
        style = str(summary_style or "").strip()
        if style:
            context["summary_style"] = style
        hint = str(user_hint or "").strip()
        if hint:
            context["user_hint"] = hint
        return context

    @staticmethod
    def _event_context(event: Optional[AstrMessageEvent]) -> Dict[str, Any]:
        if event is None:
            return {}
        umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
        return {"_astrbot_unified_msg_origin": umo} if umo else {}

    def _user_hint_from_text(self, text: str) -> str:
        """Remove trigger commands and keep the remaining text as user intent."""
        hint = str(text or "").strip()
        for keyword in self.config.summary_trigger_keywords():
            keyword_text = str(keyword or "").strip()
            if keyword_text:
                hint = hint.replace(keyword_text, "")
        return hint.strip(" \t\r\n，。；、,.!！?？:：")
