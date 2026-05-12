"""FunASR dependency and model preparation for the standalone plugin."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import re
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logger import logger


DEFAULT_REQUIRED_MODULES = ["funasr", "modelscope", "torch", "torchaudio"]
ASR_INSTALL_TIMEOUT_SECONDS = 3600
ASR_PROGRESS_LOG_INTERVAL_SECONDS = 20

MODEL_ALIASES = {
    "asr": "paraformer-zh",
    "vad": "fsmn-vad",
}


@dataclass
class AsrRuntimeStatus:
    state: str
    message: str = ""


class RuntimeStopRequested(RuntimeError):
    """Raised when a background runtime has been asked to stop."""


class AsrRuntimeStateStore:
    """Small JSON state store shared across plugin reloads."""

    VERSION = 1

    def __init__(self, runtime_dir: Path):
        self.runtime_dir = runtime_dir.resolve()
        self.path = self.runtime_dir / "asr_state.json"
        self.instance_id = uuid.uuid4().hex
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: Any) -> "AsrRuntimeStateStore":
        """Create the state store under the configured plugin runtime directory."""
        runtime_dir = Path(getattr(config, "cache_dir", "") or ".") / "runtime"
        return cls(runtime_dir)

    def read(self) -> Dict[str, Any]:
        """Read persisted runtime state and fill missing schema fields."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        return self._normalize(data)

    def update_section(self, section: str, payload: Dict[str, Any]) -> None:
        """Merge one runtime section into the shared state file atomically."""
        with self._lock:
            data = self.read()
            data["version"] = self.VERSION
            data["instance_id"] = self.instance_id
            data["updated_at"] = int(time.time())
            merged = dict(data.get(section) or {})
            merged.update(payload)
            data[section] = merged
            if str(payload.get("state", "")).upper() == "FAILED":
                data["last_error"] = str(payload.get("message", "") or "")
            self._write(data)

    def _normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            data = {}
        normalized = {
            "version": self.VERSION,
            "instance_id": str(data.get("instance_id", "") or ""),
            "updated_at": int(data.get("updated_at", 0) or 0),
            "dependencies": dict(data.get("dependencies") or {}),
            "model_files": dict(data.get("model_files") or {}),
            "worker": dict(data.get("worker") or {}),
            "last_error": str(data.get("last_error", "") or ""),
        }
        normalized["worker"].setdefault("state", "STOPPED")
        normalized["worker"].setdefault("pid", None)
        return normalized

    def _write(self, data: Dict[str, Any]) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)


def _subprocess_group_kwargs() -> Dict[str, Any]:
    """Return platform-specific options for killing subprocess trees."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _terminate_subprocess(proc: subprocess.Popen[Any]) -> None:
    """Best-effort terminate a subprocess and its children."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def _python_has_modules(
    python_path: str,
    modules: List[str],
    stop_event: Optional[threading.Event] = None,
) -> bool:
    """Check whether a Python interpreter can import all required modules."""
    checks = "; ".join(f"import {module}" for module in modules)
    proc: Optional[subprocess.Popen[Any]] = None
    try:
        proc = subprocess.Popen(
            [python_path, "-c", checks],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_subprocess_group_kwargs(),
        )
        while True:
            if stop_event and stop_event.is_set():
                _terminate_subprocess(proc)
                raise RuntimeStopRequested("ASR 依赖检查已停止")
            return_code = proc.poll()
            if return_code is not None:
                try:
                    proc.communicate(timeout=1)
                except Exception:
                    pass
                return return_code == 0
            time.sleep(0.2)
    except RuntimeStopRequested:
        raise
    except Exception:
        return False
    finally:
        if proc and proc.poll() is None:
            _terminate_subprocess(proc)


def _asr_requirements_path() -> Path:
    return Path(__file__).resolve().parents[2] / "requirements-asr.txt"


def _last_output_lines(text: str, max_lines: int = 20) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text or "")


def _progress_snippet(text: str, max_lines: int = 3, max_length: int = 240) -> str:
    cleaned = _strip_ansi(text).replace("\r", "\n")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    keywords = (
        "%",
        "Downloading",
        "download",
        "Installing",
        "Collecting",
        "Resolving",
        "Processing",
        "Using cached",
        "Preparing",
        "正在下载",
        "下载中",
        "安装中",
        "处理中",
        "剩余",
        "已下载",
        "进度",
        "it/s",
        "MB/s",
        "KB/s",
        "bytes",
        "Files",
    )
    progress_lines: List[str] = []
    for line in reversed(lines):
        if any(token in line for token in keywords):
            progress_lines.append(line[:max_length])
            if len(progress_lines) >= max_lines:
                break
    if progress_lines:
        progress_lines.reverse()
        return " | ".join(progress_lines)
    return lines[-1][:max_length]


def _collected_output_text(chunks: List[str]) -> str:
    if not chunks:
        return ""
    return "".join(chunks)


def _run_subprocess_with_progress(
    command: List[str],
    *,
    timeout_seconds: int,
    progress_label: str,
    heartbeat_seconds: int = ASR_PROGRESS_LOG_INTERVAL_SECONDS,
    stop_event: Optional[threading.Event] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess while logging heartbeat progress and honoring stop signals."""
    started_at = time.time()
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []
    stdout_lock = threading.Lock()
    stderr_lock = threading.Lock()

    def _reader(stream: Any, chunks: List[str], lock: threading.Lock) -> None:
        try:
            while True:
                chunk = stream.read(1024)
                if not chunk:
                    break
                with lock:
                    chunks.append(chunk)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **_subprocess_group_kwargs(),
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    stdout_thread = threading.Thread(
        target=_reader,
        args=(proc.stdout, stdout_chunks, stdout_lock),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_reader,
        args=(proc.stderr, stderr_chunks, stderr_lock),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    last_heartbeat = started_at
    try:
        while True:
            if stop_event and stop_event.is_set():
                logger.info(f"{progress_label} 收到停止信号，正在终止子进程")
                _terminate_subprocess(proc)
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
                with stdout_lock:
                    stdout_text = _collected_output_text(stdout_chunks)
                with stderr_lock:
                    stderr_text = _collected_output_text(stderr_chunks)
                raise RuntimeStopRequested(f"{progress_label} 已停止")
            return_code = proc.poll()
            if return_code is not None:
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
                with stdout_lock:
                    stdout_text = _collected_output_text(stdout_chunks)
                with stderr_lock:
                    stderr_text = _collected_output_text(stderr_chunks)
                return subprocess.CompletedProcess(
                    command,
                    return_code,
                    stdout_text,
                    stderr_text,
                )

            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                with stdout_lock:
                    stdout_text = _collected_output_text(stdout_chunks)
                with stderr_lock:
                    stderr_text = _collected_output_text(stderr_chunks)
                progress = _progress_snippet(stdout_text + "\n" + stderr_text)
                elapsed = int(now - started_at)
                if progress:
                    logger.info(f"{progress_label} 进度: {progress}")
                else:
                    logger.info(f"{progress_label} 仍在进行中，已耗时 {elapsed} 秒")
                last_heartbeat = now
            if timeout_seconds > 0 and now - started_at > timeout_seconds:
                _terminate_subprocess(proc)
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
                with stdout_lock:
                    stdout_text = _collected_output_text(stdout_chunks)
                with stderr_lock:
                    stderr_text = _collected_output_text(stderr_chunks)
                raise subprocess.TimeoutExpired(
                    command,
                    timeout_seconds,
                    output=stdout_text,
                    stderr=stderr_text,
                )
            time.sleep(1)
    except RuntimeStopRequested:
        raise
    except Exception:
        try:
            _terminate_subprocess(proc)
        except Exception:
            pass
        raise


def _pip_install_asr_requirements(
    python_path: str,
    requirements_path: Path,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Install ASR dependencies into the target Python interpreter."""
    if not requirements_path.is_file():
        raise RuntimeError(f"找不到 ASR 依赖文件: {requirements_path}")
    label = "AI 总结 ASR 依赖自动安装"
    logger.info(
        f"{label} 开始: python={python_path}, requirements={requirements_path}, "
        f"timeout={ASR_INSTALL_TIMEOUT_SECONDS}s"
    )
    started_at = time.time()
    result = _run_subprocess_with_progress(
        [
            python_path,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ],
        timeout_seconds=ASR_INSTALL_TIMEOUT_SECONDS,
        progress_label=label,
        stop_event=stop_event,
    )
    if result.returncode != 0:
        detail = _last_output_lines(result.stderr or result.stdout)
        raise RuntimeError(
            "ASR 依赖自动安装失败: "
            + (detail or f"pip 退出码 {result.returncode}")
        )
    elapsed = max(0.0, time.time() - started_at)
    logger.info(f"{label} 完成: elapsed={elapsed:.1f}s")


def _process_exists(pid: Any) -> bool:
    """Return whether a process id still appears to be alive."""
    try:
        process_id = int(pid)
    except (TypeError, ValueError):
        return False
    if process_id <= 0:
        return False
    if process_id == os.getpid():
        return True

    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {process_id}", "/FO", "CSV", "/NH"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        except Exception:
            return False
        return str(process_id) in (result.stdout or "")

    try:
        os.kill(process_id, 0)
    except OSError:
        return False
    return True


class AsrPythonRuntimeManager:
    """Validate the current AstrBot Python environment for FunASR."""

    def __init__(
        self,
        config: Any,
        state_store: Optional[AsrRuntimeStateStore] = None,
    ):
        self.config = config
        self.state_store = state_store or AsrRuntimeStateStore.from_config(config)
        self.modules = list(DEFAULT_REQUIRED_MODULES)
        self.install_lock_path = self.state_store.runtime_dir / "dependency_install.lock"
        self._task: Optional[asyncio.Task] = None
        self._stop_event = threading.Event()
        self._status = AsrRuntimeStatus("UNKNOWN", "ASR 依赖尚未检查")
        self._persist_status()

    def get_status(self) -> AsrRuntimeStatus:
        """Return the latest dependency preparation status."""
        return self._status

    def get_python_path(self) -> str:
        """Return the Python interpreter used by the running AstrBot process."""
        return sys.executable

    def ensure_background_prepare_started(self) -> None:
        """Start dependency checks or installation in the current event loop."""
        if self._status.state in {"READY", "CHECKING", "INSTALLING"}:
            return
        if self._task and not self._task.done():
            self._status = AsrRuntimeStatus("CHECKING", "ASR 依赖正在后台检查")
            self._persist_status()
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._status = AsrRuntimeStatus(
                "UNKNOWN",
                "当前没有可用事件循环，尚未启动 ASR 依赖检查",
            )
            self._persist_status()
            logger.warning("AI 总结 ASR 依赖检查未启动: 当前没有可用事件循环")
            return
        self._stop_event.clear()
        self._status = AsrRuntimeStatus("CHECKING", "ASR 依赖正在后台检查")
        self._persist_status()
        logger.info(
            "AI 总结 ASR 依赖检查开始: "
            f"python={self.get_python_path()}, modules={', '.join(self.modules)}"
        )
        self._task = loop.create_task(self._check_modules())

    async def ensure_ready_now(self) -> None:
        """Block until dependencies are ready or raise the preparation failure."""
        if self._status.state != "READY":
            self.ensure_background_prepare_started()
        if self._task and not self._task.done():
            await self._task
        if self._stop_event.is_set() and self._status.state != "READY":
            raise RuntimeStopRequested("ASR 依赖安装已停止")
        if self._status.state != "READY":
            raise RuntimeError(self._status.message)

    async def shutdown(self) -> None:
        """Signal background dependency work to stop and wait briefly for cleanup."""
        self._stop_event.set()
        task = self._task
        if task and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10)
            except asyncio.TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            except Exception:
                await asyncio.gather(task, return_exceptions=True)
        self._task = None

    async def _check_modules(self) -> None:
        """Check required imports and install missing ASR dependencies if needed."""
        python_path = self.get_python_path()
        try:
            if await self._modules_ready(python_path):
                self._mark_ready(python_path)
            else:
                if self._dependency_install_lock_is_active():
                    self._status = AsrRuntimeStatus(
                        "INSTALLING",
                        "ASR 依赖正在后台自动安装",
                    )
                    self._persist_status()
                    logger.info("AI 总结 ASR 依赖正在后台自动安装，等待复查")
                    await self._wait_for_external_install(python_path)
                    if await self._modules_ready(python_path):
                        self._mark_ready(python_path)
                        self._persist_status()
                        return

                requirements_path = _asr_requirements_path()
                self._status = AsrRuntimeStatus(
                    "INSTALLING",
                    f"ASR 依赖正在后台自动安装: {requirements_path}",
                )
                self._persist_status()
                logger.info(
                    "AI 总结 ASR 依赖自动安装开始: "
                    f"python={python_path}, requirements={requirements_path}"
                )
                await asyncio.to_thread(
                    self._install_requirements_sync,
                    python_path,
                    requirements_path,
                )
                logger.info("AI 总结 ASR 依赖自动安装完成，开始复查")
                if await self._modules_ready(python_path):
                    self._mark_ready(python_path)
                else:
                    self._status = AsrRuntimeStatus(
                        "FAILED",
                        "ASR 依赖自动安装后仍无法导入: "
                        + ", ".join(self.modules),
                    )
                    logger.warning(
                        "AI 总结 ASR 依赖自动安装后仍缺失: "
                        f"python={python_path}, modules={', '.join(self.modules)}"
                    )
        except RuntimeStopRequested:
            self._status = AsrRuntimeStatus("UNKNOWN", "ASR 依赖安装已停止")
            self._persist_status()
            logger.info("AI 总结 ASR 依赖安装已停止")
        except asyncio.CancelledError:
            self._status = AsrRuntimeStatus("UNKNOWN", "ASR 依赖检查已取消")
            self._persist_status()
            logger.info("AI 总结 ASR 依赖检查已取消")
            raise
        except Exception as exc:
            self._status = AsrRuntimeStatus("FAILED", str(exc))
            logger.warning(f"AI 总结 ASR 依赖检查失败: {exc}")
        self._persist_status()

    async def _modules_ready(self, python_path: str) -> bool:
        """Check required modules without blocking the event loop."""
        return await asyncio.to_thread(
            _python_has_modules,
            python_path,
            self.modules,
            self._stop_event,
        )

    def _mark_ready(self, python_path: str) -> None:
        """Mark ASR dependencies as ready and emit the readiness log."""
        self._status = AsrRuntimeStatus(
            "READY",
            f"ASR 依赖已就绪: {python_path}",
        )
        logger.info(f"AI 总结 ASR 依赖已就绪: python={python_path}")

    def _install_requirements_sync(
        self,
        python_path: str,
        requirements_path: Path,
    ) -> None:
        """Install dependencies under a cross-instance lock."""
        if self._stop_event.is_set():
            raise RuntimeStopRequested("ASR 依赖安装已停止")
        self._write_lock(self.install_lock_path, "funasr-dependencies")
        try:
            _pip_install_asr_requirements(
                python_path,
                requirements_path,
                self._stop_event,
            )
        finally:
            try:
                if self.install_lock_path.exists():
                    self.install_lock_path.unlink()
            except Exception:
                pass

    async def _wait_for_external_install(self, python_path: str) -> None:
        """Wait for another plugin instance to finish dependency installation."""
        deadline = time.time() + ASR_INSTALL_TIMEOUT_SECONDS
        started_at = time.time()
        last_heartbeat = started_at
        logger.info("AI 总结 ASR 依赖正在等待已有安装任务完成")
        while self._dependency_install_lock_is_active() and time.time() < deadline:
            if self._stop_event.is_set():
                raise RuntimeStopRequested("ASR 依赖安装已停止")
            if await self._modules_ready(python_path):
                return
            now = time.time()
            if now - last_heartbeat >= ASR_PROGRESS_LOG_INTERVAL_SECONDS:
                logger.info(
                    "AI 总结 ASR 依赖仍在等待已有安装任务完成: "
                    f"elapsed={int(now - started_at)}s"
                )
                last_heartbeat = now
            await asyncio.sleep(1)

    def _dependency_install_lock_is_active(self) -> bool:
        return self._lock_is_active(self.install_lock_path, ASR_INSTALL_TIMEOUT_SECONDS)

    def _persist_status(self) -> None:
        try:
            self.state_store.update_section(
                "dependencies",
                {
                    "state": self._status.state,
                    "message": self._status.message,
                    "python_path": self.get_python_path(),
                    "checked_at": int(time.time())
                    if self._status.state in {"READY", "FAILED"}
                    else 0,
                },
            )
        except Exception as exc:
            logger.warning(f"AI 总结 ASR 依赖状态落盘失败: {exc}")

    @staticmethod
    def _write_lock(path: Path, task: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "started_at": int(time.time()),
            "updated_at": int(time.time()),
            "task": task,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _lock_is_active(path: Path, stale_after: int) -> bool:
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        updated_at = int(
            payload.get("updated_at")
            or payload.get("started_at")
            or 0
        )
        if updated_at and time.time() - updated_at > stale_after:
            return False
        return _process_exists(payload.get("pid"))


class AsrModelRuntimeManager:
    """Prepare FunASR ASR/VAD models with ModelScope when configured."""

    def __init__(
        self,
        config: Any,
        python_runtime: Optional[AsrPythonRuntimeManager] = None,
        state_store: Optional[AsrRuntimeStateStore] = None,
    ):
        self.config = config
        self.models_dir = Path(config.asr_model_dir).resolve()
        self.python_runtime = python_runtime
        self.state_store = state_store or AsrRuntimeStateStore.from_config(config)
        self.runtime_dir = self.state_store.runtime_dir
        self.lock_path = self.runtime_dir / "model_download.lock"
        self._task: Optional[asyncio.Task] = None
        self._stop_event = threading.Event()
        self._status = self._detect_status()
        self._persist_status()

    def get_status(self) -> AsrRuntimeStatus:
        """Detect and return the current local model preparation status."""
        detected = self._detect_status()
        if self._status.state == "FAILED" and detected.state != "READY":
            self._persist_status()
            return self._status
        self._status = detected
        self._persist_status()
        return self._status

    def is_ready(self) -> bool:
        """Return whether all required ASR model directories are present."""
        return self._detect_status().state == "READY"

    def ensure_background_download_started(self) -> None:
        """Start background ASR/VAD model preparation when models are missing."""
        detected = self._detect_status()
        if detected.state == "READY":
            self._status = AsrRuntimeStatus("READY", "本地 ASR 模型已就绪")
            self._persist_status()
            logger.info(
                "AI 总结本地 ASR 模型已就绪，无需下载: "
                f"models_dir={self.models_dir}"
            )
            return
        if detected.state == "PREPARING":
            self._status = detected
            self._persist_status()
            logger.info(f"AI 总结本地 ASR 模型正在准备: {detected.message}")
            return
        if self._task and not self._task.done():
            self._status = AsrRuntimeStatus("PREPARING", "模型正在后台准备")
            self._persist_status()
            logger.info("AI 总结本地 ASR 模型后台准备任务已在运行")
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._status = AsrRuntimeStatus(
                "MISSING",
                "当前没有可用事件循环，无法启动后台模型准备任务",
            )
            self._persist_status()
            logger.warning("AI 总结本地 ASR 模型准备未启动: 当前没有可用事件循环")
            return
        self._stop_event.clear()
        self._task = loop.create_task(self._download_models())
        self._status = AsrRuntimeStatus("PREPARING", "模型正在后台准备")
        self._persist_status()
        logger.info(
            "AI 总结本地 ASR 模型准备开始: "
            f"source=modelscope, models_dir={self.models_dir}"
        )

    async def shutdown(self) -> None:
        """Signal model preparation to stop and wait briefly for cleanup."""
        self._stop_event.set()
        task = self._task
        if task and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10)
            except asyncio.TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            except Exception:
                await asyncio.gather(task, return_exceptions=True)
        self._task = None

    async def ensure_ready_now(self) -> None:
        """Block until ASR/VAD models are ready or raise the preparation failure."""
        if self._stop_event.is_set():
            raise RuntimeStopRequested("模型下载任务已停止")
        if self._detect_status().state == "READY":
            self._status = AsrRuntimeStatus("READY", "本地 ASR 模型已就绪")
            self._write_manifest()
            self._persist_status()
            return
        if self._task and not self._task.done():
            await self._task
            if self._stop_event.is_set():
                raise RuntimeStopRequested("模型下载任务已停止")
            if self._status.state != "READY":
                raise RuntimeError(self._status.message)
            return
        if self._detect_status().state == "PREPARING":
            raise RuntimeError("模型正在后台准备，请稍后重试")
        await self._download_models()
        if self._status.state != "READY":
            raise RuntimeError(self._status.message)

    def get_model_paths(self) -> Dict[str, str]:
        """Return the expected local directories for ASR and VAD models."""
        return {
            "asr": str(self.models_dir / MODEL_ALIASES["asr"]),
            "vad": str(self.models_dir / MODEL_ALIASES["vad"]),
        }

    def _detect_status(self) -> AsrRuntimeStatus:
        """Inspect model directories and active locks to determine readiness."""
        paths = self.get_model_paths()
        missing = [
            name
            for name, path in paths.items()
            if not self._looks_like_model_dir(Path(path))
        ]
        if not missing:
            return AsrRuntimeStatus("READY", "本地 ASR 模型已就绪")
        if self._download_lock_is_active():
            return AsrRuntimeStatus(
                "PREPARING",
                "本地 ASR 模型正在后台准备，请稍后重试",
            )
        if self.lock_path.exists():
            logger.info(f"AI 总结清理过期模型下载锁: {self.lock_path}")
            self._cleanup_stale_download_artifacts()
            return AsrRuntimeStatus("MISSING", "缺少模型: " + ", ".join(missing))
        return AsrRuntimeStatus("MISSING", "缺少模型: " + ", ".join(missing))

    @staticmethod
    def _looks_like_model_dir(path: Path) -> bool:
        if not path.is_dir():
            return False
        expected = ("configuration.json", "config.yaml", "model.pt")
        return any((path / name).exists() for name in expected)

    async def _download_models(self) -> None:
        """Prepare model files in a background task and persist the final status."""
        timeout_seconds = max(60, int(self.config.download_timeout_minutes) * 60)
        started_at = time.time()
        try:
            self._status = AsrRuntimeStatus("PREPARING", "模型正在后台准备")
            self._persist_status()
            logger.info(
                "AI 总结本地 ASR 模型准备任务启动: "
                f"models_dir={self.models_dir}"
            )
            if self.python_runtime:
                await self.python_runtime.ensure_ready_now()
            await asyncio.wait_for(
                asyncio.to_thread(self._download_models_sync),
                timeout=timeout_seconds,
            )
            self._status = self._detect_status()
            self._persist_status()
            if self._status.state == "READY":
                self._write_manifest()
                self._persist_status()
                logger.info(
                    "AI 总结本地 ASR 模型准备完成: "
                    f"elapsed={time.time() - started_at:.1f}s"
                )
        except RuntimeStopRequested:
            self._status = AsrRuntimeStatus("MISSING", "模型下载任务已停止")
            self._persist_status()
            logger.info("AI 总结本地 ASR 模型准备已停止")
        except asyncio.CancelledError:
            self._status = AsrRuntimeStatus("MISSING", "模型下载任务已取消")
            self._persist_status()
            logger.info("AI 总结本地 ASR 模型准备已取消")
            raise
        except Exception as exc:
            self._status = AsrRuntimeStatus("FAILED", str(exc))
            self._persist_status()
            logger.warning(f"AI 总结本地 ASR 模型准备失败: {exc}")

    def _download_models_sync(self) -> None:
        """Download ASR/VAD models and promote verified files into place."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._write_lock(self.lock_path)

        downloads = [
            ("asr", self.config.asr_model),
            ("vad", self.config.vad_model),
        ]
        try:
            for kind, repo in downloads:
                if self._stop_event.is_set():
                    raise RuntimeStopRequested("模型下载任务已停止")
                alias = MODEL_ALIASES[kind]
                final_dir = self.models_dir / alias
                if self._looks_like_model_dir(final_dir):
                    logger.info(
                        "AI 总结模型文件已存在，跳过下载: "
                        f"kind={kind}, path={final_dir}"
                    )
                    continue

                tmp_root = self._download_tmp_dir() / alias
                if tmp_root.exists():
                    shutil.rmtree(tmp_root, ignore_errors=True)
                tmp_root.mkdir(parents=True, exist_ok=True)

                downloaded = self._snapshot_download(kind, repo, tmp_root)
                downloaded_dir = Path(downloaded)
                if self._stop_event.is_set():
                    raise RuntimeStopRequested("模型下载任务已停止")
                if not self._looks_like_model_dir(downloaded_dir):
                    raise RuntimeError(f"模型下载后校验失败: {repo}")

                if final_dir.exists():
                    shutil.rmtree(final_dir, ignore_errors=True)
                shutil.copytree(downloaded_dir, final_dir)
                logger.info(
                    "AI 总结模型落盘完成: "
                    f"kind={kind}, path={final_dir}"
                )

            self._write_manifest()
        finally:
            try:
                if self.lock_path.exists():
                    self.lock_path.unlink()
            except Exception:
                pass
            if self._stop_event.is_set():
                self._cleanup_stale_download_artifacts()

    def _download_tmp_dir(self) -> Path:
        return self.models_dir.parent / ".download.tmp"

    def _download_lock_is_active(self) -> bool:
        if self._task and not self._task.done():
            return True
        if not self.lock_path.exists():
            return False
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        updated_at = int(
            payload.get("updated_at")
            or payload.get("started_at")
            or 0
        )
        stale_after = max(
            3600,
            int(getattr(self.config, "download_timeout_minutes", 60) or 60) * 60 * 2,
        )
        if updated_at and time.time() - updated_at > stale_after:
            return False
        return _process_exists(payload.get("pid"))

    def _cleanup_stale_download_artifacts(self) -> None:
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass
        tmp_dir = self._download_tmp_dir()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _manifest_dir(self) -> Path:
        if self.models_dir.name == "models":
            return self.models_dir.parent
        return self.models_dir

    def _write_manifest(self) -> None:
        paths = self.get_model_paths()
        manifest = {
            "engine": "funasr",
            "version": 1,
            "models": {
                "asr": {
                    "repo": self.config.asr_model,
                    "path": os.path.relpath(paths["asr"], self._manifest_dir()),
                },
                "vad": {
                    "repo": self.config.vad_model,
                    "path": os.path.relpath(paths["vad"], self._manifest_dir()),
                },
            },
            "created_at": int(time.time()),
        }
        manifest_dir = self._manifest_dir()
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.json"
        tmp_path = manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(manifest_path)

    def _snapshot_download(self, kind: str, repo: str, cache_dir: Path) -> str:
        """Run ModelScope snapshot_download in the managed Python interpreter."""
        python_path = (
            self.python_runtime.get_python_path()
            if self.python_runtime else
            sys.executable
        )
        downloader = (
            "from modelscope import snapshot_download; "
            "import sys; "
            "print(snapshot_download(sys.argv[1], cache_dir=sys.argv[2]))"
        )
        label = f"AI 总结模型下载进行中: kind={kind}, repo={repo}"
        logger.info(
            f"AI 总结模型下载开始: kind={kind}, repo={repo}, cache_dir={cache_dir}"
        )
        started_at = time.time()
        result = _run_subprocess_with_progress(
            [python_path, "-c", downloader, repo, str(cache_dir)],
            timeout_seconds=max(60, int(self.config.download_timeout_minutes) * 60),
            progress_label=label,
            stop_event=self._stop_event,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            fallback = self._find_modelscope_download(cache_dir, repo)
            if fallback and self._looks_like_model_dir(fallback):
                logger.warning(
                    "ModelScope 返回失败但模型文件已完整落盘，继续使用: "
                    f"{fallback}"
                )
                return str(fallback)
            raise RuntimeError(f"模型拉取失败: {repo}: {detail}")
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"模型拉取失败: {repo}: 未返回下载目录")
        logger.info(
            "AI 总结模型下载完成: "
            f"kind={kind}, repo={repo}, elapsed={time.time() - started_at:.1f}s"
        )
        return lines[-1]

    @staticmethod
    def _find_modelscope_download(cache_dir: Path, repo: str) -> Optional[Path]:
        if "/" not in repo:
            return None
        namespace, model_name = repo.split("/", 1)
        candidates = [
            cache_dir / namespace / model_name,
            cache_dir / "models" / namespace / model_name,
            cache_dir / "._____temp" / namespace / model_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _write_lock(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "started_at": int(time.time()),
            "updated_at": int(time.time()),
            "task": "funasr-asr-vad",
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _persist_status(self) -> None:
        try:
            self.state_store.update_section(
                "model_files",
                {
                    "state": self._status.state,
                    "message": self._status.message,
                    "models_dir": str(self.models_dir),
                    "updated_at": int(time.time()),
                },
            )
        except Exception as exc:
            logger.warning(f"AI 总结 ASR 模型状态落盘失败: {exc}")

