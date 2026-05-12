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
    """Flatten FunASR output into compact transcript text."""
    items = result if isinstance(result, list) else [result]
    parts: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            parts.append(text)

    text = "\n".join(parts)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def transcribe(args: argparse.Namespace) -> str:
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
    text = normalize_funasr_text(result)
    if not text:
        raise RuntimeError("FunASR 返回了空转写结果")
    return text


def main() -> int:
    """Run the worker and write the transcript JSON artifact."""
    args = parse_args()
    text = transcribe(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"text": text}, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
