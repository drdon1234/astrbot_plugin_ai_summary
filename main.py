"""Standalone AstrBot plugin for video AI summaries."""
from __future__ import annotations

import asyncio
import html
import os
import re
import shutil
import uuid
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import aiohttp

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.star.filter.event_message_type import EventMessageType

from .core.config import AISummaryConfig, parse_config
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


@dataclass
class VideoCandidate:
    """Video source plus metadata carried into the summary pipeline."""

    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@register(
    "astrbot_plugin_ai_summary",
    "drdon1234",
    "基于本地语音转写，结合用户提示词与视觉信息的多模态 AI 视频总结工具",
    "0.1.3",
)
class AISummaryPlugin(Star):
    """AstrBot plugin entry point for reply-triggered video summaries."""

    def __init__(self, context: Context, config: dict):
        """Initialize config, summary manager, task tracking, and concurrency guards."""
        super().__init__(context)
        self.config: AISummaryConfig = parse_config(config)
        logger.info(
            "AI 总结插件已载入: "
            f"cache_dir={self.config.cache_dir}, "
            f"runtime_dir={Path(self.config.cache_dir) / 'runtime'}, "
            f"model_dir={self.config.asr_model_dir}"
        )
        self.summary_manager = AISummaryManager(
            self.config,
            self.config.cache_dir,
            True,
            context,
        )
        self.summary_manager.start_background_prepare()
        self._shutdown_event = threading.Event()
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._semaphore = asyncio.Semaphore(max(1, self.config.max_concurrent))

    async def terminate(self):
        """Stop active summary tasks, release runtimes, and clear downloaded videos."""
        self._shutdown_event.set()
        shutdown_results = await asyncio.gather(
            self._cancel_active_tasks(),
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
        """Remove downloaded videos owned by this plugin instance."""
        raw_cache_dir = str(getattr(self.config, "cache_dir", "") or "").strip()
        if not raw_cache_dir:
            return
        downloads_dir = Path(raw_cache_dir).resolve() / "downloads"
        try:
            if downloads_dir.exists():
                shutil.rmtree(downloads_dir, ignore_errors=True)
                self._debug("已清空视频缓存目录: %r", str(downloads_dir))
        except Exception as exc:
            logger.warning(f"AI 总结视频缓存目录清理失败: {exc}")

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
    async def summarize_video(self, event: AstrMessageEvent):
        """Handle reply messages that request AI video summarization."""
        if self._shutdown_event.is_set():
            return
        current_task = self._track_current_task()
        try:
            cfg = self.config
            text = event.message_str or ""
            if "AI总结：" in text or "AI 总结：" in text:
                return

            is_private = event.is_private_chat()
            sender_id = event.get_sender_id()
            group_id = None if is_private else event.get_group_id()
            if not cfg.permission.check(is_private, sender_id, group_id):
                return

            summarize_reply = cfg.should_summarize_reply(text)
            self._debug(
                "触发检查: reply=%s text=%r",
                summarize_reply,
                text[:120],
            )
            if not summarize_reply:
                return

            candidates: List[VideoCandidate] = []
            candidates.extend(await self._extract_reply_candidates(event))

            candidates = self._dedupe_candidates(candidates)
            self._debug(
                "抽取到引用视频候选: %s",
                [candidate.source for candidate in candidates],
            )
            if not candidates:
                if cfg.has_keyword(text):
                    await event.send(
                        event.plain_result(
                            "未找到可总结的视频，请引用包含视频的消息后再发送总结关键词。"
                        )
                    )
                return

            candidates = candidates[: cfg.max_videos_per_message]
            user_hint = self._user_hint_from_text(text)
            self._attach_user_hint_to_candidates(candidates, user_hint, event)
            self._debug("可选用户附加说明: %r", user_hint[:120])
            if cfg.status_message:
                await event.send(event.plain_result("正在进行 AI 总结，请稍候..."))

            async with self._semaphore:
                results = await self._summarize_candidates(candidates)

            messages: List[str] = []
            for metadata in results:
                summary = str(metadata.get("ai_summary") or "").strip()
                error = str(metadata.get("ai_summary_error") or "").strip()
                if summary:
                    messages.append(f"AI总结：\n{summary}")
                elif cfg.show_error and error:
                    messages.append(f"AI总结失败：{error}")

            if messages and not self._shutdown_event.is_set():
                await event.send(event.plain_result("\n\n".join(messages)))
        finally:
            self._untrack_current_task(current_task)

    async def _summarize_candidates(
        self,
        candidates: List[VideoCandidate],
    ) -> List[Dict[str, Any]]:
        """Download candidate videos, run summaries, and clean temporary files."""
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
                        "候选[%d]准备开始: source=%r",
                        index,
                        candidate.source,
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
        if not self.config.debug_mode:
            return
        try:
            text = message % args if args else message
        except Exception:
            text = message
        logger.info(f"AI 总结调试: {text}")

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
        candidate: VideoCandidate,
    ) -> Dict[str, Any]:
        """Convert a supported video source into summary-ready metadata."""
        source = candidate.source.strip()
        downloaded_path = ""
        try:
            if source.lower().startswith(("http://", "https://")):
                video_path = await self._download_video(session, source)
                downloaded_path = video_path
            else:
                raise RuntimeError(f"不支持的视频来源: {source}")
            file_size = os.path.getsize(video_path)
            self._debug(
                "候选源准备完成: source=%r path=%r size=%s",
                source,
                video_path,
                self._format_bytes(file_size),
            )

            max_bytes = int(self.config.max_video_size_mb * 1024 * 1024)
            if max_bytes > 0 and file_size > max_bytes:
                size_mb = file_size / 1024 / 1024
                raise RuntimeError(f"视频超过大小限制: {size_mb:.1f}MB")

            metadata = dict(candidate.metadata)
            metadata.setdefault("url", source)
            metadata["video_urls"] = [[source]]
            metadata["file_paths"] = [video_path]
            metadata["video_modes"] = ["local"]
            metadata["video_count"] = 1
            metadata["has_valid_media"] = True
            if downloaded_path:
                metadata["_cleanup_file_paths"] = [downloaded_path]
            return metadata
        except Exception:
            if downloaded_path:
                self._cleanup_file_paths([downloaded_path])
            raise

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
                    self._debug("已清理下载视频: %r", str(path))
            except Exception as exc:
                logger.warning(f"AI 总结下载视频清理失败: {path}, 错误: {exc}")

    async def _extract_reply_candidates(
        self,
        event: AstrMessageEvent,
    ) -> List[VideoCandidate]:
        """Collect video candidates from the quoted message and remote fallbacks."""
        candidates: List[VideoCandidate] = []
        for comp in self._safe_get_messages(event):
            if comp.__class__.__name__.lower() != "reply":
                continue
            context: Dict[str, Any] = {}
            chain = getattr(comp, "chain", None) or []
            candidates.extend(
                self._extract_candidates_from_parts(chain, context)
            )
            remote_candidates = await self._extract_reply_remote_candidates(
                event,
                comp,
                context,
            )
            if remote_candidates:
                candidates.extend(remote_candidates)
        return candidates

    async def _extract_reply_remote_candidates(
        self,
        event: AstrMessageEvent,
        reply: Any,
        context: Dict[str, Any],
    ) -> List[VideoCandidate]:
        """Fetch quoted message details when local reply components lack video URLs."""
        reply_id = str(getattr(reply, "id", "") or "").strip()
        if not reply_id:
            return []
        payload = await self._call_platform_action_compat(
            event,
            "get_msg",
            reply_id,
        )
        if not payload:
            self._debug("引用消息远程回查为空: reply_id=%s", reply_id)
            return []
        references = self._video_references_from_onebot_payload(payload)
        sources = self._sources_from_video_references(references)
        if len(sources) < len(references):
            remote_sources = await self._resolve_remote_file_sources(
                event,
                references,
            )
            for source in remote_sources:
                if source not in sources:
                    sources.append(source)
        self._debug(
            "引用消息远程视频源: reply_id=%s sources=%s",
            reply_id,
            sources,
        )
        return [
            VideoCandidate(source, dict(context))
            for source in sources
        ]

    def _extract_candidates_from_parts(
        self,
        parts: Iterable[Any],
        context: Dict[str, Any],
    ) -> List[VideoCandidate]:
        """Extract video candidates from nested AstrBot message components."""
        candidates: List[VideoCandidate] = []
        for comp in self._walk_components(parts):
            source = self._video_source_from_component(comp)
            if source:
                candidates.append(VideoCandidate(source, dict(context)))

        return candidates

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
                result = call_action(action, **params)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, dict):
                    data = result.get("data")
                    return data if data is not None else result
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
    def _video_sources_from_onebot_payload(payload: Dict[str, Any]) -> List[str]:
        return AISummaryPlugin._sources_from_video_references(
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
            AISummaryPlugin._append_video_reference(references, reference)

        raw_message = str(payload.get("raw_message", "") or "")
        for segment in AISummaryPlugin._raw_cq_segments(raw_message):
            reference = AISummaryPlugin._video_reference_from_onebot_segment(
                segment
            )
            AISummaryPlugin._append_video_reference(references, reference)
        return references

    @staticmethod
    def _sources_from_video_references(
        references: Iterable[Dict[str, str]],
    ) -> List[str]:
        sources: List[str] = []
        for reference in references:
            source = str(reference.get("source", "") or "").strip()
            if source and source not in sources:
                sources.append(source)
        return sources

    @staticmethod
    def _append_video_reference(
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
    def _source_from_mapping(value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        for key in ("url", "video_url", "src", "path", "file_path", "file", "video"):
            item = value.get(key)
            source = AISummaryPlugin._source_from_value(item)
            if source:
                return source
        return ""

    @staticmethod
    def _source_from_value(value: Any) -> str:
        if isinstance(value, str):
            source = AISummaryPlugin._strip_media_prefixes(value.strip())
            return source if AISummaryPlugin._is_usable_video_source(source) else ""
        if isinstance(value, dict):
            return AISummaryPlugin._source_from_mapping(value)
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
    def _is_video_filename(value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if text.lower().startswith(("http://", "https://")):
            text = urlparse(text).path
        return Path(text).suffix.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def _is_usable_video_source(source: str) -> bool:
        if not source:
            return False
        return source.lower().startswith(("http://", "https://"))

    async def _resolve_remote_file_sources(
        self,
        event: AstrMessageEvent,
        references: Iterable[Dict[str, str]],
    ) -> List[str]:
        """Resolve file identifiers into downloadable video URLs."""
        sources: List[str] = []
        for reference in references:
            if reference.get("source"):
                continue
            file_id = str(reference.get("file_id", "") or "").strip()
            if not file_id:
                continue
            for source in await self._resolve_one_remote_file_source(event, file_id):
                if source not in sources:
                    sources.append(source)
        return sources

    async def _resolve_one_remote_file_source(
        self,
        event: AstrMessageEvent,
        file_id: str,
    ) -> List[str]:
        """Resolve a single remote file id through private or group APIs."""
        sources: List[str] = []
        payload = await self._call_platform_action_variants(
            event,
            "get_file",
            [{"file_id": file_id}],
        )
        source = self._source_from_action_payload(payload)
        if source:
            return [source]

        if self._event_is_private_chat(event):
            payload = await self._call_platform_action_variants(
                event,
                "get_private_file_url",
                [{"file_id": file_id}],
            )
            source = self._source_from_action_payload(payload)
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
            source = self._source_from_action_payload(payload)
            if source:
                sources.append(source)
        return sources

    @staticmethod
    def _source_from_action_payload(payload: Any) -> str:
        source = AISummaryPlugin._source_from_value(payload)
        if source:
            return source
        if isinstance(payload, dict):
            data = payload.get("data")
            source = AISummaryPlugin._source_from_value(data)
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
    def _suffix_for_url(url: str) -> str:
        suffix = Path(urlparse(url).path).suffix.lower()
        return suffix if suffix in VIDEO_EXTENSIONS else ".mp4"

    @staticmethod
    def _dedupe_candidates(
        candidates: List[VideoCandidate],
    ) -> List[VideoCandidate]:
        """Preserve the first candidate for each distinct source."""
        seen = set()
        deduped: List[VideoCandidate] = []
        for candidate in candidates:
            key = candidate.source.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _attach_user_hint_to_candidates(
        self,
        candidates: List[VideoCandidate],
        user_hint: str,
        event: Optional[AstrMessageEvent] = None,
    ) -> None:
        """Attach user-provided summary hints and event context to candidates."""
        context = self._candidate_context(user_hint, event)
        if not context:
            return
        for candidate in candidates:
            candidate.metadata.update(context)

    def _candidate_context(
        self,
        user_hint: str,
        event: Optional[AstrMessageEvent] = None,
    ) -> Dict[str, Any]:
        context = self._event_context(event)
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
        """Remove trigger keywords and keep the remaining text as user intent."""
        hint = str(text or "").strip()
        for keyword in getattr(self.config, "keywords", []) or []:
            keyword_text = str(keyword or "").strip()
            if keyword_text:
                hint = hint.replace(keyword_text, "")
        return hint.strip(" \t\r\n，。；、,.!！?？:：")
