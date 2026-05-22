"""Pillow image rendering helpers for summary messages."""
from __future__ import annotations

import asyncio
import html
import re
from pathlib import Path
from typing import List

_FONT_DIR = Path(__file__).resolve().parent.parent / "resource" / "font"
_LOCAL_FONT_PATH = _FONT_DIR / "NotoSansCJKsc-Regular.otf"
_LOCAL_BOLD_FONT_PATH = _FONT_DIR / "NotoSansCJKsc-Bold.otf"
_DEFAULT_CANVAS_WIDTH = 960
_MIN_CANVAS_WIDTH = 760
_DEFAULT_IMAGE_FONT_SIZE = 25
_MIN_IMAGE_FONT_SIZE = 16
_MAX_IMAGE_FONT_SIZE = 48


async def render_summary_image_file(
    text: str,
    content_format: str,
    output_path: str,
    *,
    title: str = "AI 视频总结",
    width: int = _DEFAULT_CANVAS_WIDTH,
    font_size: int = _DEFAULT_IMAGE_FONT_SIZE,
    timeout_seconds: int = 60,
) -> str:
    """Render a summary to a PNG file using the bundled Pillow font path."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    await _render_text_image_file(
        text,
        content_format,
        output,
        title,
        width=max(_MIN_CANVAS_WIDTH, int(width or _DEFAULT_CANVAS_WIDTH)),
        font_size=_normalize_image_font_size(font_size),
        timeout_seconds=timeout_seconds,
    )
    return _validated_output_path(output)


def _validated_output_path(output: Path) -> str:
    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError("渲染器未生成有效图片")
    return str(output)


async def _render_text_image_file(
    text: str,
    content_format: str,
    output_path: Path,
    title: str,
    *,
    width: int,
    font_size: int,
    timeout_seconds: int,
) -> None:
    await asyncio.wait_for(
        asyncio.to_thread(
            _render_text_image_file_sync,
            text,
            content_format,
            output_path,
            title,
            width,
            font_size,
        ),
        timeout=max(10, int(timeout_seconds or 60)),
    )


def _render_text_image_file_sync(
    text: str,
    content_format: str,
    output_path: Path,
    title: str,
    width: int,
    font_size: int,
) -> None:
    """Draw a readable summary card image using Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("缺少 Pillow，无法生成总结图片") from exc

    canvas_width = max(_MIN_CANVAS_WIDTH, int(width or _DEFAULT_CANVAS_WIDTH))
    margin_x = max(34, canvas_width // 24)
    content_width = canvas_width - margin_x * 2
    body_font_size = _normalize_image_font_size(font_size)

    title_font = _load_pillow_font(
        ImageFont,
        round(body_font_size * 1.6),
        bold=True,
    )
    heading_font = _load_pillow_font(
        ImageFont,
        round(body_font_size * 1.2),
        bold=True,
    )
    subheading_font = _load_pillow_font(
        ImageFont,
        round(body_font_size * 1.04),
        bold=True,
    )
    body_font = _load_pillow_font(ImageFont, body_font_size)

    probe = Image.new("RGB", (canvas_width, 200), "#f4f7f2")
    draw = ImageDraw.Draw(probe)
    image_title, sections = _summary_to_text_sections(text, content_format, title)
    title_lines = _wrap_text(draw, image_title, title_font, content_width - 72)
    title_line_height = _line_height(draw, title_font, 1.35)
    heading_line_height = _line_height(draw, heading_font, 1.32)
    subheading_line_height = _line_height(draw, subheading_font, 1.28)
    body_line_height = _line_height(draw, body_font, 1.45)

    hero_height = 48 + len(title_lines) * title_line_height + 42
    section_layouts = []
    total_height = hero_height + 26
    for index, section in enumerate(sections):
        layout, height = _measure_text_image_section(
            draw,
            section,
            content_width,
            heading_font,
            subheading_font,
            body_font,
            heading_line_height,
            subheading_line_height,
            body_line_height,
        )
        section_layouts.append((index, layout, height))
        total_height += height + 16
    total_height += 18

    image = Image.new("RGB", (canvas_width, total_height), "#f4f7f2")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, canvas_width, hero_height), fill="#edf4f1")
    y = 36
    for line in title_lines:
        line_width = _text_width(draw, line, title_font)
        draw.text(
            ((canvas_width - line_width) / 2, y),
            line,
            font=title_font,
            fill="#22313d",
        )
        y += title_line_height
    rule_width = 94
    draw.rounded_rectangle(
        (
            (canvas_width - rule_width) / 2,
            y + 12,
            (canvas_width + rule_width) / 2,
            y + 16,
        ),
        radius=2,
        fill="#5f9388",
    )

    y = hero_height + 24
    accents = ["#6a8fb7", "#5f9388", "#b07aa1", "#d28b52", "#4f9aa3", "#c06f7c"]
    for index, layout, height in section_layouts:
        x0 = margin_x
        x1 = canvas_width - margin_x
        y1 = y + height
        draw.rounded_rectangle(
            (x0, y, x1, y1),
            radius=8,
            fill="#ffffff",
            outline="#dfe7e2",
            width=1,
        )
        accent = accents[index % len(accents)]
        draw.rounded_rectangle((x0, y, x0 + 6, y1), radius=3, fill=accent)
        content_x = x0 + 28
        content_y = y + 20
        content_y = _draw_text_image_section(
            draw,
            layout,
            content_x,
            content_y,
            x1 - 24,
            heading_font,
            subheading_font,
            body_font,
            heading_line_height,
            subheading_line_height,
            body_line_height,
            accent,
        )
        y = y1 + 16

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def _summary_to_text_sections(
    text: str,
    content_format: str,
    default_title: str = "AI 视频总结",
) -> tuple[str, List[dict[str, object]]]:
    is_markdown = str(content_format or "").casefold() == "markdown"
    title = str(default_title or "").strip() or "AI 视频总结"
    sections: List[dict[str, object]] = [{"heading": "", "items": []}]

    for raw_line in str(text or "").strip().splitlines():
        line = raw_line.strip()
        if not line:
            if _section_has_content(sections[-1]):
                _section_items(sections).append(("blank", ""))
            continue
        if is_markdown and line.startswith("# "):
            title = _clean_inline_markdown(line[2:].strip()) or title
            continue
        if is_markdown and line.startswith("## "):
            sections.append({
                "heading": _clean_inline_markdown(line[3:].strip()),
                "items": [],
            })
            continue
        if is_markdown and line.startswith("### "):
            cleaned = _clean_inline_markdown(line[4:].strip())
            if cleaned:
                _section_items(sections).append(("subheading", cleaned))
            continue

        bullet = re.match(r"^(?:[-*+]|[0-9]+[.)])\s+(.+)$", line)
        if is_markdown and bullet:
            cleaned = _clean_inline_markdown(bullet.group(1))
            if cleaned:
                _section_items(sections).append(("bullet", cleaned))
        else:
            cleaned = _clean_inline_markdown(line)
            if cleaned:
                _section_items(sections).append(("text", cleaned))

    sections = [
        section for section in sections
        if _section_has_content(section)
    ]
    return title, sections or [{"heading": "", "items": [("text", "（无内容）")]}]


def _section_items(sections: List[dict[str, object]]) -> List[tuple[str, str]]:
    return sections[-1].setdefault("items", [])  # type: ignore[return-value]


def _section_has_content(section: dict[str, object]) -> bool:
    if str(section.get("heading") or "").strip():
        return True
    for kind, value in section.get("items", []):  # type: ignore[assignment]
        if kind != "blank" and str(value or "").strip():
            return True
    return False


def _measure_text_image_section(
    draw: object,
    section: dict[str, object],
    width: int,
    heading_font: object,
    subheading_font: object,
    body_font: object,
    heading_line_height: int,
    subheading_line_height: int,
    body_line_height: int,
) -> tuple[List[tuple[str, List[str]]], int]:
    inner_width = width - 56
    layout: List[tuple[str, List[str]]] = []
    height = 42
    heading = str(section.get("heading") or "").strip()
    if heading:
        lines = _wrap_text(draw, heading, heading_font, inner_width)
        layout.append(("heading", lines))
        height += len(lines) * heading_line_height + 18

    for kind, value in section.get("items", []):  # type: ignore[assignment]
        text = str(value or "").strip()
        if kind == "blank":
            layout.append(("blank", [""]))
            height += 8
            continue
        font = subheading_font if kind == "subheading" else body_font
        max_width = inner_width - (24 if kind == "bullet" else 0)
        lines = _wrap_text(draw, text, font, max_width)
        layout.append((kind, lines))
        if kind == "subheading":
            height += len(lines) * subheading_line_height + 8
        else:
            height += len(lines) * body_line_height + 10
    return layout, max(86, height + 12)


def _draw_text_image_section(
    draw: object,
    layout: List[tuple[str, List[str]]],
    x: int,
    y: int,
    right: int,
    heading_font: object,
    subheading_font: object,
    body_font: object,
    heading_line_height: int,
    subheading_line_height: int,
    body_line_height: int,
    accent: str,
) -> int:
    for kind, lines in layout:
        if kind == "blank":
            y += 8
            continue
        if kind == "heading":
            font = heading_font
            fill = "#22313d"
            line_height = heading_line_height
            bottom_gap = 14
        elif kind == "subheading":
            font = subheading_font
            fill = "#356f78"
            line_height = subheading_line_height
            bottom_gap = 8
        else:
            font = body_font
            fill = "#3e4a57"
            line_height = body_line_height
            bottom_gap = 10

        text_x = x
        if kind == "bullet":
            draw.ellipse((x, y + 12, x + 8, y + 20), fill=accent)
            text_x = x + 24
        for line in lines:
            draw.text((text_x, y), line, font=font, fill=fill)
            y += line_height
        y += bottom_gap
    return y


def _clean_inline_markdown(value: str) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = text.replace("\\", "")
    if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", text):
        return ""
    if "|" in text:
        parts = [part.strip() for part in text.strip("|").split("|")]
        text = " / ".join(part for part in parts if part)
    return text.strip()


def _wrap_text(draw: object, text: str, font: object, max_width: int) -> List[str]:
    value = str(text or "").strip()
    if not value:
        return [""]
    lines: List[str] = []
    current = ""
    for char in value:
        candidate = current + char
        if not current or _text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue
        lines.append(current.rstrip())
        current = char.lstrip()
    if current:
        lines.append(current.rstrip())
    return lines or [value]


def _text_width(draw: object, text: str, font: object) -> int:
    bbox = draw.textbbox((0, 0), str(text or ""), font=font)  # type: ignore[attr-defined]
    return int(bbox[2] - bbox[0])


def _line_height(draw: object, font: object, factor: float) -> int:
    bbox = draw.textbbox((0, 0), "视频总结Ag", font=font)  # type: ignore[attr-defined]
    return max(1, int((bbox[3] - bbox[1]) * factor))


def _normalize_image_font_size(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = _DEFAULT_IMAGE_FONT_SIZE
    return max(_MIN_IMAGE_FONT_SIZE, min(_MAX_IMAGE_FONT_SIZE, parsed))


def _load_pillow_font(image_font: object, size: int, *, bold: bool = False) -> object:
    font_path = _LOCAL_BOLD_FONT_PATH if bold else _LOCAL_FONT_PATH
    if not font_path.is_file():
        raise RuntimeError(f"本地字体文件不存在: {font_path}")
    try:
        return image_font.truetype(str(font_path), size)  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(f"加载本地字体失败: {font_path}") from exc
