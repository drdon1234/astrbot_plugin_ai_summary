"""AI summary orchestration: ASR transcription + LLM summary."""
from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from astrbot.api import logger
from .asr_runtime import (
    AsrModelRuntimeManager,
    AsrPythonRuntimeManager,
    AsrRuntimeStateStore,
)
from .llm_client import LLMClient
from .prompts import (
    DEFAULT_QA_SYSTEM_PROMPT,
    DEFAULT_QA_USER_PROMPT,
    DEFAULT_SUMMARY_SYSTEM_PROMPT,
    DEFAULT_VISION_DECISION_PROMPT,
    DEFAULT_VISUAL_ANALYSIS_PROMPT,
    build_summary_prompt,
    normalize_summary_style,
)


def _log_cleanup_error(action: str, exc: Exception) -> None:
    """Record cleanup failures without interrupting cancellation paths."""
    logger.debug(f"AI 总结清理忽略异常: action={action}, error={exc}")


class AISummaryManager:
    """Generate summaries for prepared video records."""

    _AUTO_PROFESSIONAL_KEYWORDS = (
        "财经",
        "金融",
        "投资",
        "上市",
        "融资",
        "股权",
        "债务",
        "估值",
        "商业",
        "公司",
        "企业",
        "政策",
        "案例",
        "复盘",
        "争议",
        "新闻",
        "科普",
        "教程",
        "方法",
        "技术",
        "模型",
        "开发者",
        "成本",
        "风险",
        "数据",
        "研究",
        "market",
        "model",
        "developer",
        "cost",
        "benchmark",
        "api",
        "company",
        "business",
        "risk",
        "research",
    )
    _AUTO_NEWS_KEYWORDS = (
        "警方",
        "通报",
        "公安",
        "案件",
        "违法",
        "犯罪",
        "嫌疑",
        "调查",
        "处置",
        "伤情",
        "受伤",
        "死亡",
        "事故",
        "事件",
        "发布通报",
        "news",
        "police",
        "case",
    )
    _AUTO_STRONG_NEWS_KEYWORDS = (
        "警方",
        "通报",
        "公安",
        "违法",
        "犯罪",
        "嫌疑",
        "伤情",
        "受伤",
        "死亡",
        "事故",
        "发布通报",
        "police",
    )
    _AUTO_LOW_INFO_KEYWORDS = (
        "bgm",
        "歌词",
        "纯音乐",
        "舞蹈",
        "跳舞",
        "游戏对局",
        "擦边",
        "展示",
        "水印",
    )

    _REPAIR_SYSTEM_PROMPT = (
        "你是视频总结格式修复器。只修复已有总结的输出格式，"
        "删除不应展示的原始转写内容，不新增事实，不改变主要判断。"
    )

    def __init__(
        self,
        config: Any,
        cache_dir: str,
        cache_dir_available: bool,
        astrbot_context: Any = None,
    ):
        """Wire config, runtime managers, LLM client, and concurrency controls."""
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir_available = bool(cache_dir_available)
        self.astrbot_context = astrbot_context
        self.runtime_state = AsrRuntimeStateStore.from_config(config)
        self.python_runtime = AsrPythonRuntimeManager(config, self.runtime_state)
        self.asr_runtime = AsrModelRuntimeManager(
            config,
            self.python_runtime,
            self.runtime_state,
        )
        self.llm_client = LLMClient(config)
        self._summary_semaphore = asyncio.Semaphore(
            max(1, int(getattr(config, "max_concurrent", 1) or 1))
        )
        self._asr_semaphore = asyncio.Semaphore(
            max(1, int(getattr(config, "asr_max_concurrent", 1) or 1))
        )
        self._vision_semaphore = asyncio.Semaphore(
            max(1, int(getattr(config, "vision_max_concurrent", 2) or 2))
        )

    def _debug(self, message: str, *args: Any) -> None:
        if not bool(getattr(self.config, "debug_mode", False)):
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

    @staticmethod
    def _path_size(path: str) -> str:
        try:
            return AISummaryManager._format_bytes(os.path.getsize(path))
        except OSError:
            return "unknown"

    @staticmethod
    def _format_command(command: List[str]) -> str:
        text = " ".join(str(part) for part in command)
        return text if len(text) <= 500 else text[:500] + "...(已截断)"

    def start_background_prepare(self) -> None:
        """Kick off background preparation for ASR dependencies and models."""
        logger.info("AI 总结后台 ASR 准备调度")
        self.python_runtime.ensure_background_prepare_started()
        self.asr_runtime.ensure_background_download_started()

    async def shutdown(self) -> None:
        """Stop background runtime preparation tasks."""
        await asyncio.gather(
            self.asr_runtime.shutdown(),
            self.python_runtime.shutdown(),
        )

    async def summarize_metadata_list(
        self,
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """Summarize all eligible prepared video metadata records."""
        started_at = time.perf_counter()
        self.python_runtime.ensure_background_prepare_started()
        if self.python_runtime.get_status().state == "READY":
            self.asr_runtime.ensure_background_download_started()

        eligible = [
            metadata for metadata in metadata_list
            if self._metadata_can_try_summary(metadata)
        ]
        self._debug(
            "总结批次开始: total=%d eligible=%d dependency_state=%s model_state=%s",
            len(metadata_list),
            len(eligible),
            self.python_runtime.get_status().state,
            self.asr_runtime.get_status().state,
        )
        tasks = [
            asyncio.create_task(self._summarize_one(metadata))
            for metadata in eligible
        ]
        if not tasks:
            self._debug("总结批次跳过: 无可总结视频")
            return

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self._debug(
                "总结批次被取消: eligible=%d elapsed=%.2fs",
                len(eligible),
                time.perf_counter() - started_at,
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        self._debug(
            "总结批次完成: eligible=%d elapsed=%.2fs",
            len(eligible),
            time.perf_counter() - started_at,
        )

    def _metadata_can_try_summary(self, metadata: Dict[str, Any]) -> bool:
        """Return whether metadata has enough information for a summary attempt."""
        if metadata.get("error"):
            return False
        return bool(metadata.get("video_urls"))

    async def _summarize_one(self, metadata: Dict[str, Any]) -> None:
        """Run ASR, optional visual analysis, and LLM summarization for one record."""
        async with self._summary_semaphore:
            started_at = time.perf_counter()
            source = str(metadata.get("url", "") or "")
            try:
                self._debug("总结任务开始: url=%r", source)
                dependency_status = self.python_runtime.get_status()
                model_status = self.asr_runtime.get_status()
                self._debug(
                    "运行时状态: dependency=%s message=%r model=%s message=%r",
                    dependency_status.state,
                    dependency_status.message,
                    model_status.state,
                    model_status.message,
                )
                await self._ensure_python_runtime_ready_for_request()
                await self._ensure_asr_ready_for_request()
                video_paths = self._local_video_paths(metadata)
                if not video_paths:
                    raise RuntimeError(
                        "没有可用于总结的本地视频文件；请确认视频下载目录可用，"
                        "且视频未超过大小限制或下载失败"
                    )
                self._debug(
                    "本地视频路径: count=%d paths=%s",
                    len(video_paths),
                    video_paths,
                )

                summaries: List[str] = []
                for index, video_path in enumerate(video_paths, start=1):
                    transcript_started_at = time.perf_counter()
                    self._debug(
                        "视频[%d] ASR开始: path=%r size=%s",
                        index,
                        video_path,
                        self._path_size(video_path),
                    )
                    transcript = await self._transcribe_video(video_path)
                    self._debug(
                        "视频[%d] ASR完成: transcript_chars=%d elapsed=%.2fs",
                        index,
                        len(transcript),
                        time.perf_counter() - transcript_started_at,
                    )
                    if len(transcript) > self.config.max_transcript_chars:
                        self._debug(
                            "视频[%d] 转写截断: original_chars=%d limit=%d",
                            index,
                            len(transcript),
                            self.config.max_transcript_chars,
                        )
                        transcript = transcript[: self.config.max_transcript_chars]
                    decision_started_at = time.perf_counter()
                    visual_decision = await self._decide_visual_fallback(
                        metadata,
                        transcript,
                    )
                    self._debug(
                        "视频[%d] 视觉判定完成: need_visual=%s quality=%r reason=%r elapsed=%.2fs",
                        index,
                        visual_decision.get("need_visual"),
                        visual_decision.get("transcript_quality"),
                        visual_decision.get("reason"),
                        time.perf_counter() - decision_started_at,
                    )
                    visual = ""
                    if visual_decision.get("need_visual"):
                        try:
                            visual_started_at = time.perf_counter()
                            self._debug("视频[%d] 视觉分析开始", index)
                            visual = await self._analyze_video_visuals(
                                video_path,
                                metadata,
                                transcript,
                            )
                            self._debug(
                                "视频[%d] 视觉分析完成: chars=%d elapsed=%.2fs",
                                index,
                                len(visual),
                                time.perf_counter() - visual_started_at,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logger.warning(
                                f"AI 总结视觉兜底失败: {video_path}, 错误: {e}"
                            )
                            visual = f"视觉兜底失败：{e}"

                    llm_started_at = time.perf_counter()
                    self._debug("视频[%d] LLM总结开始", index)
                    summary = await self._call_llm_summary(
                        metadata,
                        transcript,
                        visual,
                        visual_decision,
                    )
                    self._debug(
                        "视频[%d] LLM总结完成: summary_chars=%d elapsed=%.2fs",
                        index,
                        len(summary),
                        time.perf_counter() - llm_started_at,
                    )
                    summaries.append(summary)

                metadata["ai_summary"] = "\n\n".join(summaries).strip()
                self._debug(
                    "总结任务完成: url=%r videos=%d summary_chars=%d elapsed=%.2fs",
                    source,
                    len(video_paths),
                    len(metadata["ai_summary"]),
                    time.perf_counter() - started_at,
                )
            except asyncio.CancelledError:
                self._debug(
                    "总结任务被取消: url=%r elapsed=%.2fs",
                    source,
                    time.perf_counter() - started_at,
                )
                raise
            except Exception as e:
                self._debug(
                    "总结任务失败: url=%r elapsed=%.2fs error=%s",
                    source,
                    time.perf_counter() - started_at,
                    e,
                )
                logger.warning(
                    f"AI 总结失败: {metadata.get('url', '')}, 错误: {e}"
                )
                if self.config.show_error:
                    metadata["ai_summary_error"] = str(e)

    async def _ensure_python_runtime_ready_for_request(self) -> None:
        """Fail fast when FunASR Python dependencies are not ready for requests."""
        status = self.python_runtime.get_status()
        if status.state == "READY":
            return
        if status.state in {"UNKNOWN", "CHECKING"}:
            self.python_runtime.ensure_background_prepare_started()
            raise RuntimeError(
                "ASR 依赖正在后台检查，请稍后重试"
            )
        if status.state == "INSTALLING":
            raise RuntimeError("ASR 依赖正在后台自动安装，请稍后重试")
        if status.state == "FAILED":
            raise RuntimeError(f"ASR runtime 准备失败: {status.message}")
        raise RuntimeError(f"ASR runtime 未就绪: {status.message}")

    async def _ensure_asr_ready_for_request(self) -> None:
        """Fail fast when local ASR/VAD model files are not ready for requests."""
        status = self.asr_runtime.get_status()
        if status.state == "READY":
            return
        if status.state in {"PREPARING", "DOWNLOADING"}:
            raise RuntimeError(f"本地语音模型正在后台准备: {status.message}")
        if status.state == "FAILED":
            raise RuntimeError(f"本地语音模型准备失败: {status.message}")
        self.asr_runtime.ensure_background_download_started()
        raise RuntimeError("本地语音模型已开始后台准备，请稍后重试")

    def _local_video_paths(self, metadata: Dict[str, Any]) -> List[str]:
        """Return existing local video paths selected from prepared metadata."""
        file_paths = metadata.get("file_paths") or []
        video_modes = metadata.get("video_modes") or []
        video_count = len(metadata.get("video_urls") or [])
        paths: List[str] = []
        for idx in range(video_count):
            if idx >= len(file_paths):
                continue
            if idx < len(video_modes) and video_modes[idx] != "local":
                continue
            path = str(file_paths[idx] or "").strip()
            if path and os.path.isfile(path):
                paths.append(path)
            if len(paths) >= self.config.max_videos_per_link:
                break
        return paths

    async def _transcribe_video(self, video_path: str) -> str:
        """Extract audio, run the ASR worker, and return transcript text."""
        tmp_parent = self._summary_tmp_parent()
        with tempfile.TemporaryDirectory(
            prefix="ai_summary_",
            dir=tmp_parent,
        ) as temp_dir:
            wav_path = os.path.join(temp_dir, "audio.wav")
            result_path = os.path.join(temp_dir, "transcript.json")
            try:
                audio_started_at = time.perf_counter()
                self._debug(
                    "音频提取开始: video=%r wav=%r",
                    video_path,
                    wav_path,
                )
                await self._extract_audio(video_path, wav_path)
                self._debug(
                    "音频提取完成: wav=%r size=%s elapsed=%.2fs",
                    wav_path,
                    self._path_size(wav_path),
                    time.perf_counter() - audio_started_at,
                )
            except RuntimeError as exc:
                if self._is_no_audio_error(str(exc)):
                    self._debug("音频提取结果: 无音频流 video=%r", video_path)
                    return ""
                raise
            try:
                asr_started_at = time.perf_counter()
                self._debug(
                    "ASR worker开始: wav=%r result=%r",
                    wav_path,
                    result_path,
                )
                await self._run_asr_worker(wav_path, result_path)
                self._debug(
                    "ASR worker完成: result=%r size=%s elapsed=%.2fs",
                    result_path,
                    self._path_size(result_path),
                    time.perf_counter() - asr_started_at,
                )
            except RuntimeError as exc:
                if self._is_empty_asr_error(str(exc)):
                    self._debug("ASR worker结果: 空转写 wav=%r", wav_path)
                    return ""
                raise
            data = json.loads(
                Path(result_path).read_text(encoding="utf-8")
            )
            text = str(data.get("text", "") or "").strip()
            self._debug("转写读取完成: chars=%d", len(text))
            return text

    @staticmethod
    def _is_no_audio_error(message: str) -> bool:
        lowered = message.lower()
        return (
            "does not contain any stream" in lowered
            or "stream map" in lowered and "matches no streams" in lowered
            or "no audio" in lowered
        )

    @staticmethod
    def _is_empty_asr_error(message: str) -> bool:
        return (
            "空转写" in message
            or "empty transcription" in message.lower()
            or "返回了空" in message
        )

    def _summary_tmp_parent(self) -> str:
        """Return the parent directory used for temporary summary artifacts."""
        if self.cache_dir and os.path.isdir(self.cache_dir):
            path = os.path.join(self.cache_dir, "runtime", "summary_tmp")
            os.makedirs(path, exist_ok=True)
            return path
        return tempfile.gettempdir()

    async def _extract_audio(self, video_path: str, wav_path: str) -> None:
        """Extract mono WAV audio from a video with ffmpeg."""
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.config.sample_rate),
            "-f",
            "wav",
            wav_path,
        ]
        await self._run_subprocess(
            command,
            timeout=max(30, int(self.config.asr_timeout_seconds)),
            error_prefix="音频提取失败",
        )

    async def _run_asr_worker(self, wav_path: str, output_path: str) -> None:
        """Run the isolated FunASR worker process for one WAV file."""
        worker_path = Path(__file__).with_name("asr_worker.py")
        python_path = self.python_runtime.get_python_path()
        asr_model_ref = self._effective_model_ref("asr")
        vad_model_ref = self._effective_model_ref("vad")
        command = [
            python_path,
            str(worker_path),
            "--input",
            wav_path,
            "--output",
            output_path,
            "--model",
            asr_model_ref,
            "--vad-model",
            vad_model_ref,
            "--models-dir",
            self.config.asr_model_dir,
            "--device",
            self.config.device,
            "--batch-size-s",
            str(self.config.batch_size_s),
        ]
        self._debug(
            "ASR worker参数: python=%r device=%r batch_size_s=%s asr_model=%r vad_model=%r",
            python_path,
            self.config.device,
            self.config.batch_size_s,
            asr_model_ref,
            vad_model_ref,
        )
        async with self._asr_semaphore:
            await self._run_subprocess(
                command,
                timeout=max(60, int(self.config.asr_timeout_seconds)),
                error_prefix="本地 ASR 转写失败",
            )

    def _effective_model_ref(self, kind: str) -> str:
        """Prefer downloaded model paths and fall back to configured model refs."""
        paths = self.asr_runtime.get_model_paths()
        path = paths.get(kind)
        if path and os.path.isdir(path):
            return path
        return self.config.asr_model if kind == "asr" else self.config.vad_model

    async def _run_subprocess(
        self,
        command: List[str],
        *,
        timeout: int,
        error_prefix: str,
    ) -> None:
        """Run a subprocess and convert failures into user-facing runtime errors."""
        started_at = time.perf_counter()
        self._debug(
            "子进程启动: prefix=%r timeout=%ss command=%s",
            error_prefix,
            timeout,
            self._format_command(command),
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
        )
        except FileNotFoundError as exc:
            self._debug(
                "子进程启动失败: prefix=%r executable=%r elapsed=%.2fs",
                error_prefix,
                command[0],
                time.perf_counter() - started_at,
            )
            raise RuntimeError(f"{error_prefix}: 找不到可执行文件 {command[0]}") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.CancelledError:
            self._debug(
                "子进程被取消: prefix=%r elapsed=%.2fs",
                error_prefix,
                time.perf_counter() - started_at,
            )
            try:
                process.kill()
            except Exception as exc:
                _log_cleanup_error("kill subprocess after cancellation", exc)
            await process.communicate()
            raise
        except asyncio.TimeoutError as exc:
            self._debug(
                "子进程超时: prefix=%r timeout=%ss elapsed=%.2fs",
                error_prefix,
                timeout,
                time.perf_counter() - started_at,
            )
            try:
                process.kill()
            except Exception as kill_exc:
                _log_cleanup_error("kill subprocess after timeout", kill_exc)
            await process.communicate()
            raise RuntimeError(f"{error_prefix}: 执行超时") from exc

        if process.returncode != 0:
            self._debug(
                "子进程失败: prefix=%r returncode=%s elapsed=%.2fs",
                error_prefix,
                process.returncode,
                time.perf_counter() - started_at,
            )
            detail = (
                stderr.decode("utf-8", errors="replace").strip()
                or stdout.decode("utf-8", errors="replace").strip()
                or f"退出码 {process.returncode}"
            )
            raise RuntimeError(f"{error_prefix}: {detail}")
        self._debug(
            "子进程完成: prefix=%r elapsed=%.2fs",
            error_prefix,
            time.perf_counter() - started_at,
        )

    async def _call_llm_summary(
        self,
        metadata: Dict[str, Any],
        transcript: str,
        visual: str = "",
        visual_decision: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render the summary prompt, call the LLM, and clean the result."""
        missing = self._missing_llm_fields(metadata)
        if missing:
            raise RuntimeError("未配置 AI 总结接口: " + "、".join(missing))

        effective_style = self._effective_summary_style(
            transcript,
            visual,
            visual_decision or {},
            metadata,
        )
        metadata["_ai_summary_effective_style"] = effective_style
        prompt = self._render_prompt(
            metadata,
            transcript,
            visual,
            visual_decision or {},
            effective_style,
        )
        self._debug(
            "总结Prompt准备完成: configured_style=%s effective_style=%s transcript_chars=%d visual_chars=%d prompt_chars=%d",
            self._configured_summary_style(metadata),
            effective_style,
            len(transcript),
            len(visual),
            len(prompt),
        )
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self._summary_system_prompt(effective_style)},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_completion_tokens,
        }
        summary = await self._post_chat_completion(
            payload,
            timeout_seconds=self.config.request_timeout_seconds,
            metadata=metadata,
        )
        return await self._postprocess_summary(summary, transcript, metadata)

    async def answer_summary_question(
        self,
        record: Any,
        question: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Answer a question using a saved summary plus recent QA turns."""
        metadata = metadata or {}
        missing = self._missing_llm_fields(metadata)
        if missing:
            raise RuntimeError("未配置 AI 问答接口: " + "、".join(missing))

        summary = str(getattr(record, "summary", "") or "").strip()
        question_text = str(question or "").strip()
        history_text = self._format_qa_history(
            getattr(record, "qa_history", []),
            int(getattr(self.config, "qa_history_turns", 5) or 0),
        )
        if not summary:
            return "当前没有可用的视频上下文，请先完成一次视频总结。"
        if not question_text:
            return "请提供要询问的问题。"

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self._qa_system_prompt()},
                {
                    "role": "user",
                    "content": DEFAULT_QA_USER_PROMPT.format(
                        summary=summary,
                        history=history_text,
                        question=question_text,
                    ),
                },
            ],
            "temperature": min(
                float(getattr(self.config, "temperature", 0.2) or 0.2),
                0.3,
            ),
            "max_completion_tokens": min(
                max(
                    int(getattr(self.config, "max_completion_tokens", 600) or 600),
                    256,
                ),
                1000,
            ),
        }
        answer = await self._post_chat_completion(
            payload,
            timeout_seconds=self.config.request_timeout_seconds,
            metadata=metadata,
        )
        text = str(answer or "").strip()
        return text or "我暂时无法生成有效回答，请稍后重试。"

    def _qa_system_prompt(self) -> str:
        """Return the QA system prompt with output-format constraints."""
        return f"{DEFAULT_QA_SYSTEM_PROMPT}\n\n{self._qa_format_instruction()}".strip()

    def _qa_format_instruction(self) -> str:
        """Describe the configured final QA answer format for the LLM."""
        if self._qa_answer_format() == "markdown":
            return (
                "本次问答回答必须输出简洁 Markdown。可以使用短标题、列表、加粗或引用，"
                "但不要输出代码块或 HTML；除非问题需要结构化拆分，否则优先用短段落直接回答。"
                "不要在答案末尾添加与问题无关的固定栏目。"
            )
        return (
            "本次问答回答必须输出纯文本。不要使用 Markdown 标题、表格、代码块或 HTML。"
        )

    def _qa_answer_format(self) -> str:
        value = str(getattr(self.config, "qa_answer_format", "text") or "text")
        return "markdown" if value.casefold() == "markdown" else "text"

    @staticmethod
    def _format_qa_history(history: Any, max_turns: int) -> str:
        """Format recent question-answer turns for the QA prompt."""
        limit = max(0, min(int(max_turns or 0), 20))
        if limit <= 0 or not isinstance(history, list):
            return "（无）"
        normalized: List[Dict[str, str]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "") or "").strip()
            answer = str(item.get("answer", "") or "").strip()
            if question and answer:
                normalized.append({"question": question, "answer": answer})
        if not normalized:
            return "（无）"
        lines: List[str] = []
        for index, turn in enumerate(normalized[-limit:], start=1):
            lines.append(f"Q{index}：{turn['question']}")
            lines.append(f"A{index}：{turn['answer']}")
        return "\n".join(lines)

    async def test_llm_connectivity(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a small ping request through the configured LLM path."""
        metadata = metadata or {}
        missing = self._missing_llm_fields(metadata)
        if missing:
            raise RuntimeError("AI 配置不完整: 缺少 " + "、".join(missing))

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是 AI 配置连通性测试助手。请只回复 pong。",
                },
                {"role": "user", "content": "ping"},
            ],
            "temperature": 0,
            "max_completion_tokens": 16,
        }
        response = await self._post_chat_completion(
            payload,
            timeout_seconds=self._llm_connectivity_timeout_seconds(),
            metadata=metadata,
        )
        text = str(response or "").strip()
        if not text:
            raise RuntimeError("AI 返回空响应")
        return text

    def _llm_connectivity_timeout_seconds(self) -> int:
        """Clamp connectivity test timeout to a short interactive range."""
        try:
            configured = int(
                getattr(self.config, "request_timeout_seconds", 30) or 30
            )
        except (TypeError, ValueError):
            configured = 30
        return min(max(10, configured), 60)

    def _summary_system_prompt(self, summary_style: Optional[str] = None) -> str:
        """Return the final summary system prompt with output-format constraints."""
        prompt = DEFAULT_SUMMARY_SYSTEM_PROMPT
        prompt = self._adapt_prompt_to_summary_format(prompt)
        return f"{prompt}\n\n{self._summary_format_instruction(summary_style)}".strip()

    def _summary_format_instruction(self, summary_style: Optional[str] = None) -> str:
        """Describe the configured final summary content format for the LLM."""
        if self._summary_format() == "markdown":
            style = summary_style or self._configured_summary_style()
            if style == "oral":
                return (
                    "本次最终总结必须输出简洁 Markdown。第一行必须是唯一的 h1 标题，"
                    "标题应直接概括视频主题；标题后直接输出 1-2 个自然段，信息密度很高时最多 3 个自然段。"
                    "不要使用“##”二级章节、表格、固定栏目、代码块或 HTML；不要输出“关键总结”“事件脉络”"
                    "“视频脉络”“主体关系”“经验启示”“应对建议”等章节。"
                )
            if style == "news":
                return (
                    "本次最终总结必须输出新闻摘要 Markdown。第一行必须是唯一的 h1 标题，"
                    "标题直接概括事件；后续优先使用“## 事件背景概述”“## 核心事件经过”"
                    "“## 关键数据与身份信息”“## 总结与当前处置进展”四个章节。"
                    "短段落和 bullet 都要适合图片卡片渲染；不要输出表格、代码块或 HTML。"
                )
            return (
                "本次最终总结必须输出笔记总结 Markdown。第一行必须是唯一的 h1 标题，"
                "标题本身承担主题概括，不要再输出“主题”章节；后续使用"
                "“## 章节标题”拆分内容板块，每个板块聚焦一个内容面向。笔记总结优先包含"
                "“关键总结”“事件脉络”或“视频脉络”“主体关系”“AI 总结”等板块。"
                "不要输出“主题”“总览”或“关键数字”章节；重要背景和数字必须在相关结论或脉络中解释清楚。"
                "“经验启示”只有在视频确实提供可迁移经验、风险教训或决策启发时才输出，"
                "不要为了固定格式强行添加。如果语音转写中带有 [mm:ss-mm:ss] 时间段，"
                "可以在事件脉络或视频脉络中保留对应起始时间点；没有时间段时不要编造时间戳。"
                "可以使用列表、加粗、引用和表格，但不要把整段结果包裹在代码块中，不要输出 HTML。"
            )
        return (
            "本次最终总结必须输出纯文本。不要使用 Markdown 标题、表格、代码块或 HTML。"
        )

    def _summary_format(self) -> str:
        value = str(getattr(self.config, "summary_format", "text") or "text")
        return "markdown" if value.casefold() == "markdown" else "text"

    def _configured_summary_style(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if isinstance(metadata, dict):
            value = str(metadata.get("summary_style", "") or "").casefold()
            if value == "auto":
                return "auto"
            if value:
                return normalize_summary_style(value)
        return "oral"

    def _effective_summary_style(
        self,
        transcript: str,
        visual: str = "",
        visual_decision: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the actual prompt style used for this summary request."""
        configured = self._configured_summary_style(metadata)
        if configured != "auto":
            return configured
        return self._auto_summary_style(transcript, visual, visual_decision or {})

    def _auto_summary_style(
        self,
        transcript: str,
        visual: str,
        visual_decision: Dict[str, Any],
    ) -> str:
        """Choose oral, news, or note mode from input density and structure."""
        quality = str(visual_decision.get("transcript_quality", "") or "").casefold()
        plain_transcript = self._strip_transcript_timestamps(transcript)
        combined = f"{plain_transcript}\n{visual}".casefold()
        transcript_chars = len(plain_transcript.strip())
        segment_count = len(re.findall(r"\[\d{2}:\d{2}-\d{2}:\d{2}\]", transcript))
        number_count = len(re.findall(r"\d+(?:\.\d+)?(?:%|％)?", plain_transcript))
        professional_hits = sum(
            1 for keyword in self._AUTO_PROFESSIONAL_KEYWORDS
            if keyword.casefold() in combined
        )
        news_hits = sum(
            1 for keyword in self._AUTO_NEWS_KEYWORDS
            if keyword.casefold() in combined
        )
        strong_news_hits = sum(
            1 for keyword in self._AUTO_STRONG_NEWS_KEYWORDS
            if keyword.casefold() in combined
        )
        low_info_hits = sum(
            1 for keyword in self._AUTO_LOW_INFO_KEYWORDS
            if keyword.casefold() in combined
        )
        visual_high_density = "信息密度：高" in visual or "信息密度: 高" in visual

        if quality in {"empty", "low"} and transcript_chars < 1200 and not visual_high_density:
            return "oral"
        if strong_news_hits >= 2 and transcript_chars >= 80:
            return "news"
        if news_hits >= 2 and 80 <= transcript_chars < 1200 and professional_hits < 2:
            return "news"
        if transcript_chars >= 1800 or segment_count >= 6:
            return "note"
        if transcript_chars >= 900 and (number_count >= 5 or professional_hits >= 2):
            return "note"
        if visual_high_density and transcript_chars >= 500:
            return "note"
        if low_info_hits and professional_hits == 0:
            return "oral"
        return "oral"

    @staticmethod
    def _strip_transcript_timestamps(transcript: str) -> str:
        return re.sub(r"\[\d{2}:\d{2}-\d{2}:\d{2}\]\s*", "", str(transcript or ""))

    def _summary_prompt_for_style(self, summary_style: str) -> str:
        return build_summary_prompt(summary_style)

    def _adapt_prompt_to_summary_format(self, prompt: str) -> str:
        """Remove default plain-text wording when Markdown output is selected."""
        if self._summary_format() != "markdown":
            return prompt
        text = str(prompt or "")
        replacements = {
            "输出纯文本，适合在聊天消息中阅读。": "",
            "输出纯文本，只输出": "输出 Markdown，只输出",
            "输出纯文本": "输出 Markdown",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text

    async def _decide_visual_fallback(
        self,
        metadata: Dict[str, Any],
        transcript: str,
    ) -> Dict[str, Any]:
        """Ask the LLM whether visual frames are needed to compensate for ASR."""
        decision = {
            "need_visual": False,
            "reason": "视觉兜底前置条件不足",
            "transcript_quality": "sufficient",
        }
        if int(getattr(self.config, "vision_max_frames", 8) or 0) <= 0:
            decision["reason"] = "视觉帧分析已关闭"
            self._debug("视觉判定跳过: max_frames<=0")
            return decision
        if not self._llm_can_try(metadata):
            self._debug(
                "视觉判定跳过: LLM配置不足 missing=%s",
                self._missing_llm_fields(metadata),
            )
            return decision

        prompt = self._render_template(
            DEFAULT_VISION_DECISION_PROMPT,
            metadata,
            transcript,
            "",
            {},
        )
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": DEFAULT_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_completion_tokens": 300,
        }
        try:
            started_at = time.perf_counter()
            self._debug(
                "视觉判定请求开始: transcript_chars=%d timeout=%ss",
                len(transcript),
                self.config.request_timeout_seconds,
            )
            content = await self._post_chat_completion(
                payload,
                timeout_seconds=self.config.request_timeout_seconds,
                metadata=metadata,
            )
            parsed = self._extract_json_object(content)
            need_visual = parsed.get("need_visual")
            if isinstance(need_visual, str):
                need_visual = need_visual.strip().lower() in (
                    "true",
                    "yes",
                    "1",
                    "需要",
                    "是",
                )
            decision = {
                "need_visual": bool(need_visual),
                "reason": str(parsed.get("reason", "") or "").strip(),
                "transcript_quality": str(
                    parsed.get("transcript_quality", "") or ""
                ).strip(),
            }
            if not decision["reason"]:
                decision["reason"] = "模型判定"
            metadata["ai_summary_visual_decision"] = decision
            self._debug(
                "视觉判定请求完成: need_visual=%s quality=%r reason=%r response_chars=%d elapsed=%.2fs",
                decision.get("need_visual"),
                decision.get("transcript_quality"),
                decision.get("reason"),
                len(content),
                time.perf_counter() - started_at,
            )
            return decision
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"AI 总结视觉兜底判定失败: {e}")
            if not transcript.strip():
                decision = {
                    "need_visual": True,
                    "reason": "判定请求失败，但 ASR 为空，启用视觉兜底",
                    "transcript_quality": "empty",
                }
            else:
                decision = {
                    "need_visual": False,
                    "reason": f"判定请求失败: {e}",
                    "transcript_quality": "unknown",
                }
            metadata["ai_summary_visual_decision"] = decision
            self._debug("视觉判定降级: decision=%s", decision)
            return decision

    async def _post_chat_completion(
        self,
        payload: Dict[str, Any],
        *,
        timeout_seconds: int,
        metadata: Optional[Dict[str, Any]] = None,
        image_paths: Optional[List[str]] = None,
    ) -> str:
        """Route a chat payload through AstrBot's provider or the custom client."""
        route = (
            "astrbot"
            if self._use_astrbot_provider()
            else f"custom:{getattr(self.config, 'llm_provider', '') or 'unknown'}"
        )
        image_count = len(image_paths or [])
        if image_count <= 0:
            image_count = self._payload_image_count(payload)
        started_at = time.perf_counter()
        self._debug(
            "LLM请求开始: route=%s model=%r messages=%d images=%d timeout=%ss",
            route,
            payload.get("model"),
            len(payload.get("messages") or []),
            image_count,
            timeout_seconds,
        )
        try:
            if self._use_astrbot_provider():
                text = await self._post_astrbot_completion(
                    payload,
                    metadata=metadata or {},
                    image_paths=image_paths or [],
                )
            else:
                text = await self.llm_client.complete(
                    payload,
                    timeout_seconds=timeout_seconds,
                )
        except Exception as exc:
            self._debug(
                "LLM请求失败: route=%s elapsed=%.2fs error=%s",
                route,
                time.perf_counter() - started_at,
                exc,
            )
            raise
        self._debug(
            "LLM请求完成: route=%s response_chars=%d elapsed=%.2fs",
            route,
            len(text),
            time.perf_counter() - started_at,
        )
        return text

    @staticmethod
    def _payload_image_count(payload: Dict[str, Any]) -> int:
        count = 0
        for message in payload.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    count += 1
        return count

    def _use_astrbot_provider(self) -> bool:
        """Return whether summaries should use AstrBot's built-in AI provider."""
        return getattr(self.config, "llm_provider_source", "astrbot") == "astrbot"

    def _missing_llm_fields(self, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """List missing fields for the active LLM route."""
        if self._use_astrbot_provider():
            missing: List[str] = []
            if self.astrbot_context is None:
                missing.append("AstrBot Context")
            configured_provider = str(
                getattr(self.config, "astrbot_provider_id", "") or ""
            ).strip()
            current_origin = str(
                (metadata or {}).get("_astrbot_unified_msg_origin", "") or ""
            ).strip()
            if not configured_provider and not current_origin:
                missing.append("AstrBot AI Provider")
            return missing
        return self.llm_client.missing_fields()

    def _llm_can_try(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Return whether the active LLM route has enough configuration to run."""
        return not self._missing_llm_fields(metadata)

    async def _post_astrbot_completion(
        self,
        payload: Dict[str, Any],
        *,
        metadata: Dict[str, Any],
        image_paths: List[str],
    ) -> str:
        """Call AstrBot's selected provider with prompt text and local image paths."""
        provider_id = await self._astrbot_provider_id(metadata)
        system_prompt, prompt, payload_images = self._payload_to_astrbot_chat(payload)
        image_urls = image_paths or payload_images
        self._debug(
            "AstrBot LLM调用: provider_id=%r prompt_chars=%d system_chars=%d images=%d",
            provider_id,
            len(prompt),
            len(system_prompt),
            len(image_urls),
        )
        response = await self._call_astrbot_llm_generate(
            provider_id=provider_id,
            prompt=prompt,
            system_prompt=system_prompt,
            image_urls=image_urls,
        )
        text = self._extract_astrbot_response_text(response)
        if not text:
            raise RuntimeError("AstrBot AI 返回空总结")
        return text

    async def _astrbot_provider_id(self, metadata: Dict[str, Any]) -> str:
        """Resolve the explicit or current-session AstrBot provider id."""
        configured_provider = str(
            getattr(self.config, "astrbot_provider_id", "") or ""
        ).strip()
        if configured_provider:
            return configured_provider
        if self.astrbot_context is None:
            raise RuntimeError("未接入 AstrBot Context，无法使用 AstrBot 内置提供商")

        umo = str(metadata.get("_astrbot_unified_msg_origin", "") or "").strip()
        provider_id = ""
        if umo and hasattr(self.astrbot_context, "get_current_chat_provider_id"):
            provider_id = str(
                await self._maybe_await(
                    self.astrbot_context.get_current_chat_provider_id(umo)
                )
                or ""
            ).strip()
        if provider_id:
            return provider_id

        provider = None
        if hasattr(self.astrbot_context, "get_using_provider"):
            provider = await self._maybe_await(
                self.astrbot_context.get_using_provider(umo or None)
            )
        if provider is not None and hasattr(provider, "meta"):
            meta = provider.meta()
            provider_id = str(getattr(meta, "id", "") or "").strip()
            if provider_id:
                return provider_id

        raise RuntimeError("未选择 AstrBot AI，且当前会话没有可用的 AstrBot AI")

    async def _call_astrbot_llm_generate(
        self,
        *,
        provider_id: str,
        prompt: str,
        system_prompt: str,
        image_urls: List[str],
    ) -> Any:
        """Invoke AstrBot LLM APIs across supported context versions."""
        if self.astrbot_context is None:
            raise RuntimeError("未接入 AstrBot Context，无法使用 AstrBot 内置提供商")
        if hasattr(self.astrbot_context, "llm_generate"):
            return await self._maybe_await(
                self.astrbot_context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    image_urls=image_urls or None,
                    system_prompt=system_prompt or None,
                )
            )

        provider = None
        if hasattr(self.astrbot_context, "get_provider_by_id"):
            provider = await self._maybe_await(
                self.astrbot_context.get_provider_by_id(provider_id)
            )
        if provider is None or not hasattr(provider, "text_chat"):
            raise RuntimeError(f"未找到 AstrBot AI Provider: {provider_id}")
        return await self._maybe_await(
            provider.text_chat(
                prompt=prompt,
                image_urls=image_urls or None,
                system_prompt=system_prompt or None,
            )
        )

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        """Await values only when the AstrBot API returns an awaitable."""
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _payload_to_astrbot_chat(payload: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """Convert chat completion payloads into AstrBot prompt arguments."""
        system_parts: List[str] = []
        user_parts: List[str] = []
        image_urls: List[str] = []
        for message in payload.get("messages") or []:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "") or "").strip()
            content = message.get("content")
            text, images = AISummaryManager._astrbot_content_parts(content)
            if role == "system":
                system_parts.append(text)
            elif role == "user":
                user_parts.append(text)
                image_urls.extend(images)
        return (
            "\n\n".join(part for part in system_parts if part).strip(),
            "\n\n".join(part for part in user_parts if part).strip(),
            image_urls,
        )

    @staticmethod
    def _astrbot_content_parts(content: Any) -> Tuple[str, List[str]]:
        """Split mixed text and image content into AstrBot-compatible parts."""
        if isinstance(content, str):
            return content.strip(), []
        if not isinstance(content, list):
            return str(content or "").strip(), []
        text_parts: List[str] = []
        image_urls: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    text_parts.append(part.strip())
                continue
            if not isinstance(part, dict):
                value = str(part or "").strip()
                if value:
                    text_parts.append(value)
                continue
            part_type = str(part.get("type", "") or "").strip()
            if part_type == "text":
                text = str(part.get("text", "") or "").strip()
                if text:
                    text_parts.append(text)
            elif part_type == "image_url":
                image_value = part.get("image_url")
                if isinstance(image_value, dict):
                    url = str(image_value.get("url", "") or "").strip()
                else:
                    url = str(image_value or "").strip()
                if url:
                    image_urls.append(url)
        return "\n".join(text_parts).strip(), image_urls

    @staticmethod
    def _extract_astrbot_response_text(response: Any) -> str:
        """Extract text from AstrBot response objects and error on provider failures."""
        if response is None:
            return ""
        role = str(getattr(response, "role", "") or "").strip()
        if role == "err":
            detail = (
                getattr(response, "completion_text", "")
                or getattr(response, "_completion_text", "")
                or str(response)
            )
            raise RuntimeError(f"AstrBot AI 返回错误: {detail}")
        text = str(
            getattr(response, "completion_text", "")
            or getattr(response, "_completion_text", "")
            or ""
        ).strip()
        if text:
            return text
        result_chain = getattr(response, "result_chain", None)
        if result_chain is not None:
            getter = getattr(result_chain, "get_plain_text", None)
            if callable(getter):
                text = str(getter() or "").strip()
                if text:
                    return text
        if isinstance(response, str):
            return response.strip()
        return str(response or "").strip()

    async def _postprocess_summary(
        self,
        summary: str,
        transcript: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Remove raw transcript leakage and optionally repair summary format."""
        cleaned, issues = self._clean_summary_output(summary, transcript)
        self._debug(
            "总结后处理开始: raw_chars=%d cleaned_chars=%d issues=%s",
            len(summary),
            len(cleaned),
            issues,
        )
        if self._summary_repair_enabled() and self._summary_needs_llm_repair(
            cleaned,
            transcript,
            issues,
        ):
            repair_input = cleaned or str(summary or "").strip()
            try:
                started_at = time.perf_counter()
                self._debug(
                    "总结格式修复开始: input_chars=%d issues=%s",
                    len(repair_input),
                    issues,
                )
                repaired = await self._repair_summary_with_llm(
                    repair_input,
                    issues,
                    metadata or {},
                )
                repaired_cleaned, _ = self._clean_summary_output(repaired, transcript)
                self._debug(
                    "总结格式修复完成: repaired_chars=%d cleaned_chars=%d elapsed=%.2fs",
                    len(repaired),
                    len(repaired_cleaned),
                    time.perf_counter() - started_at,
                )
                if repaired_cleaned:
                    cleaned = repaired_cleaned
                    if not self._summary_needs_llm_repair(
                        cleaned,
                        transcript,
                        [],
                    ):
                        return cleaned
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"AI 总结格式修复失败，使用确定性清理结果: {exc}")
                self._debug("总结格式修复失败: error=%s", exc)

        if cleaned:
            self._debug("总结后处理完成: final_chars=%d", len(cleaned))
            return cleaned
        raise RuntimeError("LLM 总结只包含原始转写或无效内容，已被过滤")

    def _summary_repair_enabled(self) -> bool:
        """Return whether LLM-based summary format repair is enabled."""
        return bool(getattr(self.config, "enable_summary_repair", True))

    async def _repair_summary_with_llm(
        self,
        summary: str,
        issues: List[str],
        metadata: Dict[str, Any],
    ) -> str:
        """Ask the LLM to repair formatting without adding new facts."""
        issue_text = "\n".join(f"- {issue}" for issue in issues) or "- 输出格式不符合要求"
        prompt = (
            "请只修复下面这段视频总结，不要重新总结视频，不要新增事实。\n\n"
            "修复要求：\n"
            "1. 删除语音转写原文、raw 片段、证据摘录、原文引用和“转写中说……”之类内容。\n"
            "2. 保留整理后的总结判断，不改变主要结论。\n"
            "3. 不确定内容必须在正文对应句子后使用“〔疑1〕”“〔疑2〕”等编号标记。\n"
            "4. 如果正文使用了编号标记，末尾添加“注释：”并逐条说明每个编号需要核对的原因；没有编号标记则不要写注释。\n"
            f"5. {self._summary_repair_format_instruction()}\n\n"
            "需要修复的问题：\n"
            f"{issue_text}\n\n"
            "待修复总结：\n"
            f"{summary.strip()}"
        )
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self._REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_completion_tokens": min(
                1200,
                max(300, int(self.config.max_completion_tokens or 1800)),
            ),
        }
        return await self._post_chat_completion(
            payload,
            timeout_seconds=min(
                60,
                max(10, int(getattr(self.config, "request_timeout_seconds", 180) or 180)),
            ),
            metadata=metadata,
        )

    def _summary_repair_format_instruction(self) -> str:
        if self._summary_format() == "markdown":
            return "输出 Markdown，只输出修复后的总结，不要包裹代码块，不要输出 HTML。"
        return "输出纯文本，只输出修复后的总结。"

    @classmethod
    def _clean_summary_output(
        cls,
        summary: str,
        transcript: str,
    ) -> Tuple[str, List[str]]:
        """Strip wrappers and raw-transcript sections from a summary."""
        text = cls._strip_summary_wrappers(str(summary or ""))
        lines = text.splitlines()
        cleaned_lines: List[str] = []
        issues: List[str] = []
        skip_raw_section = False

        for line in lines:
            stripped = line.strip()
            if cls._is_raw_section_heading(stripped):
                issues.append("输出包含原始转写、raw 或证据摘录段落")
                skip_raw_section = True
                continue
            if skip_raw_section:
                if not stripped:
                    skip_raw_section = False
                    continue
                if cls._is_summary_heading(stripped):
                    skip_raw_section = False
                else:
                    continue
            if cls._line_looks_like_raw_transcript(stripped, transcript):
                issues.append("输出中存在与语音转写高度重合的长句")
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned.strip(), cls._dedupe_text_list(issues)

    @classmethod
    def _strip_summary_wrappers(cls, text: str) -> str:
        stripped = str(text or "").strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[\w-]*\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        wrapper_patterns = [
            r"^\s*视频\[\d+\]\s*(?:总结)?\s*[:：]?\s*",
            r"^\s*AI\s*总结\s*[:：]\s*",
            r"^\s*AI总结\s*[:：]\s*",
            r"^\s*以下是(?:修复后的)?(?:视频)?总结\s*[:：]?\s*",
        ]
        changed = True
        while changed:
            changed = False
            for pattern in wrapper_patterns:
                new_value = re.sub(pattern, "", stripped, count=1, flags=re.I)
                if new_value != stripped:
                    stripped = new_value.strip()
                    changed = True
        return stripped

    @staticmethod
    def _is_raw_section_heading(line: str) -> bool:
        if not line:
            return False
        return bool(
            re.match(
                r"^(?:语音转写|原始转写|转写原文|转写片段|语音原文|原文|raw|asr|证据摘录|转写引用)\s*[:：]",
                line,
                re.I,
            )
            or re.match(r"^以下是.*(?:语音转写|原始转写|raw|证据摘录)", line, re.I)
        )

    @staticmethod
    def _is_summary_heading(line: str) -> bool:
        return bool(
            re.match(
                r"^(?:主题|概要|关键总结|关键事件|事件脉络|视频脉络|主体关系|经验启示|结论|注释|总结)\s*[:：]",
                line,
            )
        )

    @classmethod
    def _line_looks_like_raw_transcript(cls, line: str, transcript: str) -> bool:
        if not line or not transcript:
            return False
        normalized_line = cls._normalize_for_overlap(line)
        if len(normalized_line) < 60:
            return False
        normalized_transcript = cls._normalize_for_overlap(transcript)
        if not normalized_transcript:
            return False
        if normalized_line in normalized_transcript:
            return True
        window = 80
        if len(normalized_line) <= window:
            return False
        return any(
            normalized_line[index:index + window] in normalized_transcript
            for index in range(0, len(normalized_line) - window + 1, 20)
        )

    @staticmethod
    def _normalize_for_overlap(text: str) -> str:
        return re.sub(r"[\s\W_]+", "", str(text or "").lower(), flags=re.U)

    @classmethod
    def _summary_needs_llm_repair(
        cls,
        summary: str,
        transcript: str,
        issues: List[str],
    ) -> bool:
        """Return whether deterministic cleanup still leaves repair-worthy issues."""
        text = str(summary or "")
        if not text.strip():
            return True
        if issues:
            return True
        if re.search(r"(?:语音转写|原始转写|转写原文|转写片段|raw|证据摘录)\s*[:：]", text, re.I):
            return True
        if "需核对" in text:
            return True
        markers = [int(value) for value in re.findall(r"〔疑(\d+)〕", text)]
        has_notes = "注释：" in text
        if markers:
            expected = set(range(1, max(markers) + 1))
            if set(markers) != expected:
                return True
            note_numbers = {
                int(value)
                for value in re.findall(r"(?:疑|〔疑)(\d+)(?:〕)?\s*[:：]", text)
            }
            if not has_notes or not set(markers).issubset(note_numbers):
                return True
        elif has_notes:
            return True
        if re.search(r"^(?:不确定|存疑|待核对|需要核对)(?:的)?(?:内容|部分|信息)?\s*[:：]", text, re.M):
            return True
        return any(
            cls._line_looks_like_raw_transcript(line.strip(), transcript)
            for line in text.splitlines()
        )

    @staticmethod
    def _dedupe_text_list(values: List[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    async def _analyze_video_visuals(
        self,
        video_path: str,
        metadata: Dict[str, Any],
        transcript: str,
    ) -> str:
        """Sample video frames and summarize visual evidence in batches."""
        with tempfile.TemporaryDirectory(
            prefix="ai_visual_",
            dir=self._summary_tmp_parent(),
        ) as temp_dir:
            started_at = time.perf_counter()
            self._debug(
                "视觉抽帧开始: video=%r temp_dir=%r max_frames=%s width=%s",
                video_path,
                temp_dir,
                self.config.vision_max_frames,
                self.config.vision_frame_width,
            )
            frames = await self._extract_visual_frames(
                video_path,
                Path(temp_dir),
            )
            if not frames:
                raise RuntimeError("未能从视频中抽取视觉帧")
            self._debug(
                "视觉抽帧完成: frames=%d elapsed=%.2fs",
                len(frames),
                time.perf_counter() - started_at,
            )

            batch_size = max(1, int(self.config.vision_batch_size or 4))
            batches = [
                frames[index:index + batch_size]
                for index in range(0, len(frames), batch_size)
            ]
            self._debug(
                "视觉分析批次: frames=%d batch_size=%d batches=%d",
                len(frames),
                batch_size,
                len(batches),
            )

            async def run_batch(
                batch_index: int,
                batch: List[Tuple[Path, float]],
            ) -> str:
                async with self._vision_semaphore:
                    return await self._call_visual_batch(
                        metadata,
                        transcript,
                        batch,
                        batch_index,
                    )

            results = await asyncio.gather(
                *[
                    run_batch(batch_index, batch)
                    for batch_index, batch in enumerate(batches, start=1)
                ]
            )

        visual = "\n\n".join(
            f"视觉片段[{index}]：\n{text.strip()}"
            for index, text in enumerate(results, start=1)
            if text.strip()
        ).strip()
        max_chars = max(1000, int(self.config.vision_max_chars or 8000))
        if len(visual) > max_chars:
            visual = visual[:max_chars] + "\n（视觉观察因长度限制已截断）"
            self._debug("视觉观察截断: max_chars=%d", max_chars)
        self._debug("视觉分析汇总完成: chars=%d", len(visual))
        return visual

    async def _extract_visual_frames(
        self,
        video_path: str,
        output_dir: Path,
    ) -> List[Tuple[Path, float]]:
        """Extract representative JPEG frames for visual fallback analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        duration = await self._probe_duration(video_path)
        max_frames = int(self.config.vision_max_frames or 0)
        if max_frames <= 0:
            self._debug("视觉抽帧跳过: max_frames=%d", max_frames)
            return []
        width = int(self.config.vision_frame_width or 0)
        quality = max(2, min(31, int(self.config.vision_jpeg_quality or 4)))
        frames: List[Tuple[Path, float]] = []

        times = self._sample_frame_times(duration, max_frames)
        self._debug(
            "视觉抽帧参数: duration=%s sample_times=%s width=%d quality=%d",
            f"{duration:.2f}s" if duration else "unknown",
            [round(value, 2) for value in times],
            width,
            quality,
        )
        if times:
            for index, timestamp in enumerate(times, start=1):
                frame_path = output_dir / f"frame_{index:03d}.jpg"
                command = [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                ]
                frame_filter = self._visual_frame_filter(width)
                if frame_filter:
                    command.extend(["-vf", frame_filter])
                command.extend([
                    "-q:v",
                    str(quality),
                    str(frame_path),
                ])
                try:
                    await self._run_subprocess(
                        command,
                        timeout=60,
                        error_prefix="视觉帧抽取失败",
                    )
                except RuntimeError as exc:
                    self._debug(
                        "跳过视觉帧: index=%d timestamp=%.2fs error=%s",
                        index,
                        timestamp,
                        exc,
                    )
                    continue
                if frame_path.exists() and frame_path.stat().st_size > 0:
                    frames.append((frame_path, timestamp))
            self._debug("视觉定点抽帧结果: frames=%d", len(frames))
            return frames

        pattern = output_dir / "frame_%03d.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            self._visual_frame_filter(width, prefix="fps=1/5"),
            "-frames:v",
            str(max_frames),
            "-q:v",
            str(quality),
            str(pattern),
        ]
        await self._run_subprocess(
            command,
            timeout=max(60, int(self.config.asr_timeout_seconds)),
            error_prefix="视觉帧抽取失败",
        )
        for index, frame_path in enumerate(sorted(output_dir.glob("frame_*.jpg")), start=1):
            if frame_path.stat().st_size > 0:
                frames.append((frame_path, float(index - 1) * 5.0))
        self._debug("视觉fallback抽帧结果: frames=%d", len(frames))
        return frames

    @staticmethod
    def _visual_frame_filter(width: int, prefix: str = "") -> str:
        parts = [prefix.strip()] if str(prefix or "").strip() else []
        try:
            frame_width = int(width or 0)
        except (TypeError, ValueError):
            frame_width = 0
        if frame_width > 0:
            parts.append(f"scale={max(64, frame_width)}:-2")
        return ",".join(parts)

    async def _probe_duration(self, video_path: str) -> Optional[float]:
        """Return video duration from ffprobe when it is available."""
        ffprobe = self._ffprobe_path()
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return None

        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.CancelledError:
            try:
                process.kill()
            except Exception as exc:
                _log_cleanup_error("kill ffprobe after cancellation", exc)
            await process.communicate()
            raise
        except asyncio.TimeoutError:
            try:
                process.kill()
            except Exception as exc:
                _log_cleanup_error("kill ffprobe after timeout", exc)
            await process.communicate()
            return None
        if process.returncode != 0:
            return None
        try:
            duration = float(stdout.decode("utf-8", errors="replace").strip())
        except (TypeError, ValueError):
            return None
        if duration <= 0:
            return None
        return duration

    def _ffprobe_path(self) -> str:
        return "ffprobe"

    @staticmethod
    def _sample_frame_times(
        duration: Optional[float],
        max_frames: int,
    ) -> List[float]:
        """Choose evenly distributed frame timestamps inside the video duration."""
        if not duration or duration <= 0:
            return []
        count = max(1, int(max_frames))
        if count == 1:
            return [max(0.0, duration / 2.0)]
        margin = min(1.0, duration * 0.08)
        start = margin
        end = max(start, duration - margin)
        if end <= start:
            return [duration / 2.0]
        return [
            start + (end - start) * index / (count - 1)
            for index in range(count)
        ]

    async def _call_visual_batch(
        self,
        metadata: Dict[str, Any],
        transcript: str,
        frames: List[Tuple[Path, float]],
        batch_index: int,
    ) -> str:
        """Send one batch of sampled frames to the configured visual-capable LLM."""
        started_at = time.perf_counter()
        frame_notes = "\n".join(
            f"帧{index}: 约 {timestamp:.1f}s"
            for index, (_, timestamp) in enumerate(frames, start=1)
        )
        self._debug(
            "视觉批次请求开始: batch=%d frames=%d timestamps=%s",
            batch_index,
            len(frames),
            [round(timestamp, 2) for _, timestamp in frames],
        )
        prompt = self._render_template(
            DEFAULT_VISUAL_ANALYSIS_PROMPT,
            metadata,
            transcript[: min(len(transcript), 4000)],
            "",
            {"frame_notes": frame_notes},
        )
        prompt = f"这是第 {batch_index} 批抽样帧。\n\n{prompt}"

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        detail = str(self.config.vision_image_detail or "low")
        for frame_path, _ in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self._image_data_url(frame_path),
                    "detail": detail,
                },
            })

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": DEFAULT_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
            "max_completion_tokens": min(
                1200,
                max(300, int(self.config.max_completion_tokens or 1800)),
            ),
        }
        result = await self._post_chat_completion(
            payload,
            timeout_seconds=self.config.vision_request_timeout_seconds,
            metadata=metadata,
            image_paths=[str(frame_path) for frame_path, _ in frames],
        )
        self._debug(
            "视觉批次请求完成: batch=%d chars=%d elapsed=%.2fs",
            batch_index,
            len(result),
            time.perf_counter() - started_at,
        )
        return result

    @staticmethod
    def _image_data_url(path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, re.S)
            if not match:
                raise
            return json.loads(match.group(0))

    def _chat_completions_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def _render_prompt(
        self,
        metadata: Dict[str, Any],
        transcript: str,
        visual: str = "",
        visual_decision: Optional[Dict[str, Any]] = None,
        summary_style: Optional[str] = None,
    ) -> str:
        """Render the selected summary prompt with transcript and visual context."""
        effective_style = summary_style or self._effective_summary_style(
            transcript,
            visual,
            visual_decision or {},
            metadata,
        )
        prompt = self._render_template(
            self._summary_prompt_for_style(effective_style),
            metadata,
            transcript,
            visual,
            visual_decision or {},
        )
        prompt = self._adapt_prompt_to_summary_format(prompt)
        return f"{prompt.rstrip()}\n\n{self._summary_format_instruction(effective_style)}"

    def _render_template(
        self,
        template: str,
        metadata: Dict[str, Any],
        transcript: str,
        visual: str,
        extra: Dict[str, Any],
    ) -> str:
        """Fill prompt template placeholders with normalized summary context."""
        transcript_text = transcript.strip() or "（无可用语音转写）"
        visual_text = visual.strip() or "（未使用视觉兜底或无视觉观察）"
        decision = extra if isinstance(extra, dict) else {}
        if "need_visual" in decision or "transcript_quality" in decision:
            visual_text = (
                f"视觉兜底判定：{decision.get('reason', '') or '未说明'}\n"
                f"转写质量：{decision.get('transcript_quality', '') or 'unknown'}\n"
                f"视觉观察：\n{visual_text}"
            )
        values = {
            "user_hint": str(metadata.get("user_hint", "") or "（无）"),
            "transcript": transcript_text,
            "visual": visual_text,
            "frame_notes": str(extra.get("frame_notes", "") or ""),
        }
        prompt = template
        for key, value in values.items():
            prompt = prompt.replace("{" + key + "}", value)
        return prompt

    @staticmethod
    def _extract_chat_content(response: Dict[str, Any]) -> str:
        """Extract OpenAI-style chat response content."""
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
