"""Subprocess worker for FunASR transcription.

This file intentionally has no project-relative imports so it can be executed
directly by a separate Python interpreter.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ALIASES = {
    "paraformer-zh": (
        "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    ),
    "fsmn-vad": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
}


TIMESTAMPED_SEGMENT_TARGET_MS = 25_000
TIMESTAMPED_SEGMENT_MIN_TOKENS = 24


def parse_args() -> argparse.Namespace:
    """Parse the command-line contract used by the parent plugin process."""
    parser = argparse.ArgumentParser(description="Transcribe wav with FunASR.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vad-model", required=True)
    parser.add_argument("--models-dir", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size-s", type=int, default=300)
    return parser.parse_args()


def configure_model_cache(models_dir: str) -> None:
    """Point model libraries at the plugin-owned model cache directory."""
    if not models_dir:
        return
    root = Path(models_dir).resolve()
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "modelscope_cache"))
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "huggingface" / "hub"))


def resolve_model_ref(name: str, models_dir: str) -> str:
    """Resolve aliases and local model directories before falling back to remote refs."""
    model_ref = str(name or "").strip()
    if not model_ref:
        return model_ref

    direct = Path(model_ref)
    if direct.exists():
        return str(direct.resolve())

    if models_dir:
        root = Path(models_dir).resolve()
        candidates = [root / model_ref]
        alias_name = ALIASES.get(model_ref)
        if alias_name:
            candidates.append(root / alias_name)
        if "/" in model_ref:
            namespace, model_name = model_ref.split("/", 1)
            candidates.append(root / model_name)
            candidates.append(root / "modelscope" / namespace / model_name)
            candidates.append(root / "modelscope" / "models" / namespace / model_name)
            candidates.append(root / "modelscope_cache" / "models" / namespace / model_name)
            candidates.append(root / model_ref.split("/")[-1])
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

    return model_ref


def normalize_funasr_text(result: Any) -> str:
    """Flatten FunASR output into readable transcript text."""
    return _plain_funasr_text(result)


def build_funasr_transcript_payload(result: Any) -> dict[str, Any]:
    """Build the JSON payload consumed by the parent summary process."""
    plain_text = _plain_funasr_text(result)
    segments = _timestamped_funasr_segments(result)
    if segments:
        text = "\n".join(
            f"[{segment['start']}-{segment['end']}] {segment['text']}"
            for segment in segments
            if segment.get("text")
        ).strip()
    else:
        text = plain_text
    return {
        "text": text,
        "plain_text": plain_text,
        "segments": segments,
    }


def _plain_funasr_text(result: Any) -> str:
    """Flatten FunASR output into readable transcript text without timestamps."""
    items = result if isinstance(result, list) else [result]
    parts: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            parts.append(text)

    return _normalize_transcript_spacing("\n".join(parts))


def _timestamped_funasr_segments(result: Any) -> list[dict[str, Any]]:
    """Convert word-level FunASR timestamps into readable transcript chunks."""
    items = result if isinstance(result, list) else [result]
    segments: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_text = str(item.get("text", "") or "").strip()
        raw_timestamps = item.get("timestamp")
        if not raw_text or not isinstance(raw_timestamps, list):
            continue
        timestamps = [_timestamp_pair(value) for value in raw_timestamps]
        timestamps = [value for value in timestamps if value is not None]
        tokens = _tokens_for_timestamps(raw_text, len(timestamps))
        if not tokens or len(tokens) != len(timestamps):
            continue
        segments.extend(_chunk_timestamped_tokens(tokens, timestamps))
    return segments


def _tokens_for_timestamps(text: str, expected_count: int) -> list[str]:
    """Return text tokens aligned with timestamp entries when possible."""
    if expected_count <= 0:
        return []
    space_tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    if len(space_tokens) == expected_count:
        return space_tokens
    compact = re.sub(r"\s+", "", text)
    chars = [char for char in compact if char]
    if len(chars) == expected_count:
        return chars
    return []


def _timestamp_pair(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        start = max(0, int(float(value[0])))
        end = max(start, int(float(value[1])))
    except (TypeError, ValueError):
        return None
    return start, end


def _chunk_timestamped_tokens(
    tokens: list[str],
    timestamps: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current_tokens: list[str] = []
    start_ms = 0
    end_ms = 0
    for token, timestamp in zip(tokens, timestamps):
        if not current_tokens:
            start_ms = timestamp[0]
        current_tokens.append(token)
        end_ms = timestamp[1]
        if (
            end_ms - start_ms >= TIMESTAMPED_SEGMENT_TARGET_MS
            and len(current_tokens) >= TIMESTAMPED_SEGMENT_MIN_TOKENS
        ):
            segments.append(_make_timestamped_segment(start_ms, end_ms, current_tokens))
            current_tokens = []
    if current_tokens:
        segments.append(_make_timestamped_segment(start_ms, end_ms, current_tokens))
    return segments


def _make_timestamped_segment(
    start_ms: int,
    end_ms: int,
    tokens: list[str],
) -> dict[str, Any]:
    return {
        "start": _format_timestamp(start_ms),
        "end": _format_timestamp(end_ms),
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "text": _join_timestamped_tokens(tokens),
    }


def _join_timestamped_tokens(tokens: list[str]) -> str:
    if any(re.search(r"[A-Za-z0-9]", token) for token in tokens):
        return _normalize_transcript_spacing(" ".join(tokens))
    return _normalize_transcript_spacing("".join(tokens))


def _format_timestamp(milliseconds: int) -> str:
    total_seconds = max(0, int(round(milliseconds / 1000)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _normalize_transcript_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or ""))
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([,.;:!?，。！？；：、])", r"\1", text)
    text = re.sub(r"([(\[（【])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]）】])", r"\1", text)
    return text.strip()


def transcribe(args: argparse.Namespace) -> dict[str, Any]:
    """Load FunASR, transcribe the provided audio file, and reject empty output."""
    configure_model_cache(args.models_dir)
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 环境未安装 funasr，无法执行本地语音转写"
        ) from exc

    model = AutoModel(
        model=resolve_model_ref(args.model, args.models_dir),
        vad_model=resolve_model_ref(args.vad_model, args.models_dir),
        device=args.device,
        disable_update=True,
    )
    result = model.generate(
        input=str(Path(args.input).resolve()),
        batch_size_s=max(1, int(args.batch_size_s)),
    )
    payload = build_funasr_transcript_payload(result)
    if not payload.get("text"):
        raise RuntimeError("FunASR 返回了空转写结果")
    return payload


def main() -> int:
    """Run the worker and write the transcript JSON artifact."""
    args = parse_args()
    payload = transcribe(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
