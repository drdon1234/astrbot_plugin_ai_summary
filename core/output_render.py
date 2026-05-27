"""Pillow image rendering helpers for summary messages."""
from __future__ import annotations

import asyncio
import html
import re
from pathlib import Path
from typing import List

_FONT_DIR = Path(__file__).resolve().parent.parent / "resource" / "font"
_FONT_FAMILIES = {
    "noto_sans": ("NotoSansCJKsc-Regular.otf", "NotoSansCJKsc-Bold.otf"),
    "noto_serif": ("NotoSerifCJKsc-Regular.otf", "NotoSerifCJKsc-Bold.otf"),
    "lxgw_wenkai": ("LXGWWenKai-Regular.ttf", "LXGWWenKai-Medium.ttf"),
    "zcool_xiaowei": ("ZCOOLXiaoWei-Regular.ttf", "ZCOOLXiaoWei-Regular.ttf"),
    "zcool_qingke": (
        "ZCOOLQingKeHuangYou-Regular.ttf",
        "ZCOOLQingKeHuangYou-Regular.ttf",
    ),
}
_DEFAULT_CANVAS_WIDTH = 960
_MIN_CANVAS_WIDTH = 760
_BILINOTE_CANVAS_WIDTH = 760
_MIN_BILINOTE_CANVAS_WIDTH = 640
_DEFAULT_IMAGE_FONT_SIZE = 25
_MIN_IMAGE_FONT_SIZE = 16
_MAX_IMAGE_FONT_SIZE = 48
_NO_LINE_START_CHARS = "，。！？；：、,.!?;:)]}）】》"


async def render_summary_image_file(
    text: str,
    content_format: str,
    output_path: str,
    *,
    title: str = "AI 视频总结",
    width: int = _DEFAULT_CANVAS_WIDTH,
    font_size: int = _DEFAULT_IMAGE_FONT_SIZE,
    style: str = "fresh",
    font_family: str = "noto_sans",
    timeout_seconds: int = 60,
) -> str:
    """Render a summary to a PNG file using a bundled Pillow font family."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized_style = _normalize_image_style(style)
    normalized_font_family = _normalize_image_font_family(font_family)
    canvas_width = max(_MIN_CANVAS_WIDTH, int(width or _DEFAULT_CANVAS_WIDTH))
    if normalized_style in {"fresh", "tech", "serious"} and canvas_width == _DEFAULT_CANVAS_WIDTH:
        canvas_width = _BILINOTE_CANVAS_WIDTH
    await _render_text_image_file(
        text,
        content_format,
        output,
        title,
        width=canvas_width,
        font_size=_normalize_image_font_size(font_size),
        style=normalized_style,
        font_family=normalized_font_family,
        timeout_seconds=timeout_seconds,
    )
    return _validated_output_path(output)


def summary_to_html_document(
    text: str,
    content_format: str,
    *,
    title: str = "AI 视频总结",
    style: str = "fresh",
) -> str:
    """Return a lightweight HTML preview for local visual inspection."""
    normalized_style = _normalize_image_style(style)
    if normalized_style not in {"fresh", "tech", "serious"}:
        image_title, sections = _summary_to_text_sections(text, content_format, title)
        section_html = []
        for section in sections:
            heading = html.escape(str(section.get("heading") or ""))
            items = []
            for kind, value in section.get("items", []):  # type: ignore[assignment]
                if kind == "blank":
                    items.append("<br>")
                elif kind == "bullet":
                    items.append(f"<li>{html.escape(str(value))}</li>")
                else:
                    items.append(f"<p>{html.escape(str(value))}</p>")
            body = "\n".join(items)
            section_html.append(f"<section><h2>{heading}</h2>{body}</section>")
        return (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<style>body{font-family:sans-serif;background:#f4f7f2;color:#22313d;}"
            "main{max-width:880px;margin:32px auto;padding:24px;}"
            "section{background:#fff;border:1px solid #dfe7e2;border-radius:8px;"
            "padding:18px 24px;margin:16px 0;}h1{text-align:center;}</style>"
            f"</head><body><main><h1>{html.escape(image_title)}</h1>"
            f"{''.join(section_html)}</main></body></html>"
        )

    image_title, sections = _summary_to_bilinote_sections(text, content_format, title)
    section_html = []
    for section in sections:
        heading = html.escape(str(section.get("heading") or ""))
        items = []
        sticky = _bilinote_section_uses_note(section)
        for kind, value in section.get("items", []):  # type: ignore[assignment]
            if kind == "blank":
                continue
            inline = _marked_text_to_html(str(value or ""))
            if sticky:
                items.append(f"<p>{inline}</p>")
            elif kind == "bullet":
                items.append(f"<p class=\"bullet\">{inline}</p>")
            else:
                items.append(f"<p>{inline}</p>")
        body = "\n".join(items)
        note_class = " note" if sticky else ""
        section_html.append(
            f"<section><h2>{heading}</h2><div class=\"body{note_class}\">{body}</div></section>"
        )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<style>body{margin:0;background:#fdeef4;background-image:radial-gradient(#f6bdcf "
        "1.4px,transparent 1.4px);background-size:14px 14px;color:#51413f;"
        "font-family:'Noto Sans CJK SC','Microsoft YaHei',sans-serif;}"
        "main{width:640px;margin:32px auto;padding:58px 34px 36px;background:#f9dfd8;"
        "border:2px solid #f4b9bd;border-radius:16px;box-shadow:0 6px 0 #edb8b5;}"
        "h1{text-align:center;color:#ff638a;font-size:28px;line-height:1.55;}"
        "h2{display:inline-block;background:#9de7ee;color:#196971;border:3px solid #fff;"
        "border-radius:12px;padding:8px 18px;font-size:19px;box-shadow:3px 3px 0 #69c6cf;}"
        ".body{font-size:21px;line-height:1.9;margin:18px 0 34px;}.note{background:#fff8b7;"
        "padding:22px 24px;box-shadow:5px 5px 0 rgba(210,183,123,.35);}.bullet:before{"
        "content:'✨';color:#f7d928;margin-right:2px;}strong{color:#167d34;}</style>"
        f"</head><body><main><h1>{html.escape(image_title)}</h1>"
        f"{''.join(section_html)}</main></body></html>"
    )


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
    style: str,
    font_family: str,
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
            style,
            font_family,
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
    style: str,
    font_family: str,
) -> None:
    """Draw a readable summary card image using Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("缺少 Pillow，无法生成总结图片") from exc

    normalized_style = _normalize_image_style(style)
    if normalized_style in {"fresh", "tech", "serious"}:
        _render_bilinote_image_file_sync(
            Image,
            ImageDraw,
            ImageFont,
            text,
            content_format,
            output_path,
            title,
            width,
            font_size,
            normalized_style,
            font_family,
        )
        return

    canvas_width = max(_MIN_CANVAS_WIDTH, int(width or _DEFAULT_CANVAS_WIDTH))
    margin_x = max(34, canvas_width // 24)
    content_width = canvas_width - margin_x * 2
    body_font_size = _normalize_image_font_size(font_size)

    title_font = _load_pillow_font(
        ImageFont,
        round(body_font_size * 1.6),
        bold=True,
        font_family=font_family,
    )
    heading_font = _load_pillow_font(
        ImageFont,
        round(body_font_size * 1.2),
        bold=True,
        font_family=font_family,
    )
    subheading_font = _load_pillow_font(
        ImageFont,
        round(body_font_size * 1.04),
        bold=True,
        font_family=font_family,
    )
    body_font = _load_pillow_font(
        ImageFont,
        body_font_size,
        font_family=font_family,
    )

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


def _render_bilinote_image_file_sync(
    image_module: object,
    image_draw: object,
    image_font: object,
    text: str,
    content_format: str,
    output_path: Path,
    title: str,
    width: int,
    font_size: int,
    theme: str,
    font_family: str,
) -> None:
    """Draw a BiliNote-inspired soft note image with Pillow primitives."""
    palette = _image_theme_palette(theme)
    canvas_width = max(_MIN_BILINOTE_CANVAS_WIDTH, int(width or _BILINOTE_CANVAS_WIDTH))
    canvas_width = min(canvas_width, 900)
    card_x0 = max(34, canvas_width // 12)
    card_x1 = canvas_width - card_x0
    content_x = card_x0 + max(30, canvas_width // 22)
    content_width = card_x1 - content_x - max(30, canvas_width // 22)
    body_font_size = _normalize_image_font_size(font_size)

    title_font = _load_pillow_font(
        image_font,
        max(24, round(body_font_size * 1.18)),
        bold=True,
        font_family=font_family,
    )
    label_font = _load_pillow_font(
        image_font,
        max(18, round(body_font_size * 0.86)),
        bold=True,
        font_family=font_family,
    )
    body_font = _load_pillow_font(
        image_font,
        body_font_size,
        font_family=font_family,
    )
    strong_font = _load_pillow_font(
        image_font,
        body_font_size,
        bold=True,
        font_family=font_family,
    )
    note_font = _load_pillow_font(
        image_font,
        max(16, round(body_font_size * 0.96)),
        font_family=font_family,
    )
    note_strong_font = _load_pillow_font(
        image_font,
        max(16, round(body_font_size * 0.96)),
        bold=True,
        font_family=font_family,
    )

    probe = image_module.new("RGB", (canvas_width, 200), palette["background"])
    draw = image_draw.Draw(probe)
    image_title, sections = _summary_to_bilinote_sections(text, content_format, title)
    title_lines = _wrap_text(draw, image_title, title_font, content_width)
    title_line_height = _line_height(draw, title_font, 1.45)
    label_height = _line_height(draw, label_font, 1.08) + 22
    body_line_height = _line_height(draw, body_font, 1.55)
    note_line_height = _line_height(draw, note_font, 1.6)

    card_top = 36
    title_y = card_top + 64
    y = title_y + len(title_lines) * title_line_height + 54
    section_layouts = []
    for section in sections:
        layout, height = _measure_bilinote_section(
            draw,
            section,
            content_width,
            label_font,
            body_font,
            strong_font,
            note_font,
            note_strong_font,
            label_height,
            body_line_height,
            note_line_height,
        )
        section_layouts.append((layout, height))
        y += height

    card_bottom = y + 28
    canvas_height = card_bottom + 38
    image = image_module.new("RGB", (canvas_width, canvas_height), palette["background"])
    draw = image_draw.Draw(image)
    _draw_bilinote_background(draw, canvas_width, canvas_height, palette)

    card_radius = 16
    shadow_offset = 5
    draw.rounded_rectangle(
        (card_x0 + shadow_offset, card_top + shadow_offset, card_x1 + shadow_offset, card_bottom + shadow_offset),
        radius=card_radius,
        fill=palette["card_shadow"],
    )
    draw.rounded_rectangle(
        (card_x0, card_top, card_x1, card_bottom),
        radius=card_radius,
        fill=palette["card_fill"],
        outline=palette["card_outline"],
        width=2,
    )
    tape_width = min(126, max(96, canvas_width // 6))
    tape_x0 = (canvas_width - tape_width) // 2
    draw.rectangle(
        (tape_x0, card_top - 14, tape_x0 + tape_width, card_top + 12),
        fill=palette["tape_fill"],
    )

    current_y = title_y
    for line in title_lines:
        line_width = _text_width(draw, line, title_font)
        draw.text(
            ((canvas_width - line_width) / 2, current_y),
            line,
            font=title_font,
            fill=palette["title"],
        )
        current_y += title_line_height

    current_y += 54
    for layout, height in section_layouts:
        _draw_bilinote_section(
            draw,
            layout,
            content_x,
            current_y,
            content_width,
            label_font,
            body_font,
            strong_font,
            note_font,
            note_strong_font,
            label_height,
            body_line_height,
            note_line_height,
            palette,
        )
        current_y += height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def _draw_bilinote_background(
    draw: object,
    width: int,
    height: int,
    palette: dict[str, str],
) -> None:
    dot_color = palette["background_dot"]
    spacing = 14
    radius = 2
    for y in range(7, height, spacing):
        for x in range(7, width, spacing):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=dot_color)


def _measure_bilinote_section(
    draw: object,
    section: dict[str, object],
    content_width: int,
    label_font: object,
    body_font: object,
    strong_font: object,
    note_font: object,
    note_strong_font: object,
    label_height: int,
    body_line_height: int,
    note_line_height: int,
) -> tuple[dict[str, object], int]:
    heading = str(section.get("heading") or "").strip()
    sticky = _bilinote_section_uses_note(section)
    layout: dict[str, object] = {
        "heading": heading,
        "sticky": sticky,
        "items": [],
    }
    label_block_height = 0
    content_height = 0
    height = 0
    if heading:
        label_width = _text_width(draw, heading, label_font) + 38
        layout["label_width"] = min(label_width, content_width)
        label_block_height = label_height + 28
        height += label_block_height

    items = []
    for kind, value in section.get("items", []):  # type: ignore[assignment]
        item_text = str(value or "").strip()
        if kind == "blank" or not item_text:
            continue
        segments = _split_marked_segments(item_text)
        if sticky:
            lines = _wrap_rich_text(
                draw,
                segments,
                content_width - 58,
                note_font,
                note_strong_font,
            )
            items.append((kind, lines))
            content_height += len(lines) * note_line_height + 10
        else:
            text_width = content_width - (30 if kind == "bullet" else 0)
            lines = _wrap_rich_text(
                draw,
                segments,
                text_width,
                body_font,
                strong_font,
            )
            items.append((kind, lines))
            height += len(lines) * body_line_height + 12

    if sticky:
        note_height = max(92, content_height + 38)
        layout["note_height"] = note_height
        height = label_block_height + note_height + 38
    else:
        height += 30
    layout["items"] = items
    return layout, max(92, height)


def _draw_bilinote_section(
    draw: object,
    layout: dict[str, object],
    x: int,
    y: int,
    content_width: int,
    label_font: object,
    body_font: object,
    strong_font: object,
    note_font: object,
    note_strong_font: object,
    label_height: int,
    body_line_height: int,
    note_line_height: int,
    palette: dict[str, str],
) -> None:
    heading = str(layout.get("heading") or "").strip()
    if heading:
        label_width = int(layout.get("label_width") or content_width)
        label_width = min(label_width, content_width)
        draw.rounded_rectangle(
            (x + 4, y + 4, x + label_width + 4, y + label_height + 4),
            radius=10,
            fill=palette["label_shadow"],
        )
        draw.rounded_rectangle(
            (x, y, x + label_width, y + label_height),
            radius=10,
            fill=palette["label_fill"],
            outline=palette["label_outline"],
            width=3,
        )
        bbox = draw.textbbox((0, 0), heading, font=label_font)  # type: ignore[attr-defined]
        text_y = y + (label_height - (bbox[3] - bbox[1])) / 2 - 2
        draw.text((x + 18, text_y), heading, font=label_font, fill=palette["label_text"])
        y += label_height + 28

    items = list(layout.get("items") or [])
    if bool(layout.get("sticky")):
        note_height = int(layout.get("note_height") or 92)
        note_x0 = x + 12
        note_x1 = x + content_width - 12
        draw.rectangle(
            (note_x0 + 5, y + 5, note_x1 + 5, y + note_height + 5),
            fill=palette["note_shadow"],
        )
        draw.rectangle(
            (note_x0, y, note_x1, y + note_height),
            fill=palette["note_fill"],
            outline=palette["note_outline"],
        )
        pin_x = (note_x0 + note_x1) // 2
        draw.line((pin_x + 1, y - 1, pin_x + 1, y + 15), fill=palette["pin_line"], width=2)
        draw.ellipse((pin_x - 6, y - 8, pin_x + 6, y + 4), fill=palette["pin"])
        text_y = y + 28
        for _, lines in items:
            for line in lines:
                _draw_rich_line(
                    draw,
                    line,
                    note_x0 + 24,
                    text_y,
                    note_font,
                    note_strong_font,
                    palette["body_text"],
                    palette["strong_text"],
                )
                text_y += note_line_height
            text_y += 10
        return

    text_y = y
    for kind, lines in items:
        line_x = x
        if kind == "bullet":
            _draw_sparkle(draw, x - 4, text_y + 15, 9, palette["sparkle"])
            _draw_sparkle(draw, x + 7, text_y + 7, 5, palette["sparkle_light"])
            line_x = x + 28
        for line in lines:
            _draw_rich_line(
                draw,
                line,
                line_x,
                text_y,
                body_font,
                strong_font,
                palette["body_text"],
                palette["strong_text"],
            )
            text_y += body_line_height
        text_y += 12


def _draw_sparkle(
    draw: object,
    cx: int,
    cy: int,
    size: int,
    fill: str,
) -> None:
    draw.polygon(
        [
            (cx, cy - size),
            (cx + max(2, size // 3), cy - max(2, size // 3)),
            (cx + size, cy),
            (cx + max(2, size // 3), cy + max(2, size // 3)),
            (cx, cy + size),
            (cx - max(2, size // 3), cy + max(2, size // 3)),
            (cx - size, cy),
            (cx - max(2, size // 3), cy - max(2, size // 3)),
        ],
        fill=fill,
    )


def _image_theme_palette(theme: str) -> dict[str, str]:
    """Return colors for the selected image theme without changing content layout."""
    normalized = _normalize_image_style(theme)
    palettes = {
        "tech": {
            "background": "#07111f",
            "background_dot": "#173a60",
            "card_fill": "#0d1828",
            "card_outline": "#2d9cff",
            "card_shadow": "#050b14",
            "tape_fill": "#38e7ff",
            "title": "#6ee7ff",
            "label_fill": "#10364e",
            "label_shadow": "#18bde8",
            "label_outline": "#8ef6ff",
            "label_text": "#d7fbff",
            "note_fill": "#10243d",
            "note_outline": "#2b78d6",
            "note_shadow": "#061327",
            "pin": "#5eeaff",
            "pin_line": "#1f8ec2",
            "body_text": "#d7e7f7",
            "strong_text": "#7cffc6",
            "sparkle": "#6ee7ff",
            "sparkle_light": "#b3fff3",
        },
        "serious": {
            "background": "#eef1f5",
            "background_dot": "#cfd6df",
            "card_fill": "#ffffff",
            "card_outline": "#a8b2bf",
            "card_shadow": "#c8ced6",
            "tape_fill": "#d8dee6",
            "title": "#1f2937",
            "label_fill": "#dbe7f5",
            "label_shadow": "#9fb4cc",
            "label_outline": "#ffffff",
            "label_text": "#233044",
            "note_fill": "#f8fafc",
            "note_outline": "#d9e2ec",
            "note_shadow": "#d5dae3",
            "pin": "#334155",
            "pin_line": "#64748b",
            "body_text": "#334155",
            "strong_text": "#0f4c81",
            "sparkle": "#64748b",
            "sparkle_light": "#94a3b8",
        },
        "fresh": {
            "background": "#fdeef4",
            "background_dot": "#f4c1d2",
            "card_fill": "#f9ded8",
            "card_outline": "#f4b7bd",
            "card_shadow": "#efb9b6",
            "tape_fill": "#fde6b6",
            "title": "#ff6389",
            "label_fill": "#a5eef2",
            "label_shadow": "#69c8d1",
            "label_outline": "#ffffff",
            "label_text": "#176c72",
            "note_fill": "#fff9b8",
            "note_outline": "#f4e6a4",
            "note_shadow": "#e8d59a",
            "pin": "#ee4b4d",
            "pin_line": "#ba544d",
            "body_text": "#51413f",
            "strong_text": "#197b35",
            "sparkle": "#ffdf2e",
            "sparkle_light": "#fff16a",
        },
    }
    return palettes.get(normalized, palettes["fresh"])


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


def _summary_to_bilinote_sections(
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
            cleaned = _clean_segment_markdown(line[4:].strip())
            if cleaned:
                _section_items(sections).append(("subheading", cleaned))
            continue

        bullet = re.match(r"^(?:[-*+]|[0-9]+[.)])\s+(.+)$", line)
        if is_markdown and bullet:
            cleaned = _clean_segment_markdown(bullet.group(1))
            if cleaned:
                _section_items(sections).append(("bullet", cleaned))
        else:
            cleaned = _clean_segment_markdown(line)
            if cleaned:
                _section_items(sections).append(("text", cleaned))

    sections = [
        section for section in sections
        if _section_has_content(section)
    ]
    return title, sections or [{"heading": "", "items": [("text", "（无内容）")]}]


def _bilinote_section_uses_note(section: dict[str, object]) -> bool:
    heading = str(section.get("heading") or "").replace(" ", "")
    if not heading:
        return False
    note_keywords = (
        "关键总结",
        "AI总结",
        "事件背景",
        "背景概述",
        "当前处置",
        "前置进展",
    )
    excluded_keywords = (
        "事件脉络",
        "视频脉络",
        "核心事件",
        "主体关系",
        "身份信息",
        "经验启示",
        "注释",
    )
    return any(word in heading for word in note_keywords) and not any(
        word in heading for word in excluded_keywords
    )


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


def _clean_segment_markdown(value: str) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("\\", "")
    if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", text):
        return ""
    if "|" in text:
        parts = [part.strip() for part in text.strip("|").split("|")]
        text = " / ".join(part for part in parts if part)
    return text.strip()


def _split_marked_segments(value: str) -> List[tuple[str, str]]:
    text = _clean_segment_markdown(value)
    if not text:
        return [("（无内容）", "normal")]
    segments: List[tuple[str, str]] = []
    marker = re.compile(r"(\*\*|__)(.*?)\1")
    position = 0
    for match in marker.finditer(text):
        if match.start() > position:
            _append_rich_segment(
                segments,
                _strip_light_markdown(text[position:match.start()]),
                "normal",
            )
        _append_rich_segment(
            segments,
            _strip_light_markdown(match.group(2)),
            "strong",
        )
        position = match.end()
    if position < len(text):
        _append_rich_segment(
            segments,
            _strip_light_markdown(text[position:]),
            "normal",
        )
    return segments or [(_strip_light_markdown(text), "normal")]


def _strip_light_markdown(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    return text


def _append_rich_segment(
    segments: List[tuple[str, str]],
    text: str,
    style: str,
) -> None:
    if not text:
        return
    if segments and segments[-1][1] == style:
        segments[-1] = (segments[-1][0] + text, style)
    else:
        segments.append((text, style))


def _wrap_rich_text(
    draw: object,
    segments: List[tuple[str, str]],
    max_width: int,
    normal_font: object,
    strong_font: object,
) -> List[List[tuple[str, str]]]:
    lines: List[List[tuple[str, str]]] = []
    current: List[tuple[str, str]] = []
    for text, style in segments:
        for char in str(text or ""):
            if char in {"\r", "\n"}:
                if current:
                    lines.append(current)
                    current = []
                continue
            candidate = list(current)
            _append_rich_segment(candidate, char, style)
            if (
                current
                and char in _NO_LINE_START_CHARS
                and _rich_text_width(draw, candidate, normal_font, strong_font) > max_width
            ):
                _append_rich_segment(current, char, style)
                continue
            if current and _rich_text_width(draw, candidate, normal_font, strong_font) > max_width:
                lines.append(current)
                current = []
                if char.isspace():
                    continue
            _append_rich_segment(current, char, style)
    if current:
        lines.append(current)
    return lines or [[("（无内容）", "normal")]]


def _rich_text_width(
    draw: object,
    segments: List[tuple[str, str]],
    normal_font: object,
    strong_font: object,
) -> int:
    total = 0
    for text, style in segments:
        font = strong_font if style == "strong" else normal_font
        total += _text_width(draw, text, font)
    return total


def _draw_rich_line(
    draw: object,
    line: List[tuple[str, str]],
    x: int,
    y: int,
    normal_font: object,
    strong_font: object,
    normal_fill: str,
    strong_fill: str,
) -> None:
    current_x = x
    for text, style in line:
        font = strong_font if style == "strong" else normal_font
        fill = strong_fill if style == "strong" else normal_fill
        draw.text((current_x, y), text, font=font, fill=fill)
        current_x += _text_width(draw, text, font)


def _marked_text_to_html(value: str) -> str:
    parts = []
    for text, style in _split_marked_segments(value):
        escaped = html.escape(text)
        parts.append(f"<strong>{escaped}</strong>" if style == "strong" else escaped)
    return "".join(parts)


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
        if char in _NO_LINE_START_CHARS:
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


def _normalize_image_style(value: object) -> str:
    text = str(value or "").strip().casefold()
    mapping = {
        "fresh": "fresh",
        "bilinote": "fresh",
        "note": "fresh",
        "fresh_note": "fresh",
        "清新": "fresh",
        "清新便签": "fresh",
        "便签": "fresh",
        "粉色便签": "fresh",
        "tech": "tech",
        "technology": "tech",
        "sci_fi": "tech",
        "科技感": "tech",
        "科技": "tech",
        "serious": "serious",
        "professional": "serious",
        "formal": "serious",
        "专业严肃": "serious",
        "严肃": "serious",
        "专业": "serious",
        "card": "card",
        "default": "card",
        "soft_card": "card",
        "温和卡片": "card",
        "卡片": "card",
    }
    return mapping.get(text, "fresh")


def _normalize_image_font_family(value: object) -> str:
    text = str(value or "").strip()
    lowered = text.casefold()
    mapping = {
        "noto_sans": "noto_sans",
        "noto sans": "noto_sans",
        "default": "noto_sans",
        "默认黑体": "noto_sans",
        "黑体": "noto_sans",
        "思源黑体": "noto_sans",
        "noto serif": "noto_serif",
        "noto_serif": "noto_serif",
        "serif": "noto_serif",
        "专业宋体": "noto_serif",
        "宋体": "noto_serif",
        "思源宋体": "noto_serif",
        "lxgw_wenkai": "lxgw_wenkai",
        "lxgw wenkai": "lxgw_wenkai",
        "wenkai": "lxgw_wenkai",
        "清新文楷": "lxgw_wenkai",
        "文楷": "lxgw_wenkai",
        "霞鹜文楷": "lxgw_wenkai",
        "zcool_xiaowei": "zcool_xiaowei",
        "zcool xiaowei": "zcool_xiaowei",
        "xiaowei": "zcool_xiaowei",
        "标题手札": "zcool_xiaowei",
        "站酷小薇": "zcool_xiaowei",
        "zcool_qingke": "zcool_qingke",
        "zcool qingke": "zcool_qingke",
        "qingke": "zcool_qingke",
        "科技窄体": "zcool_qingke",
        "站酷庆科黄油体": "zcool_qingke",
    }
    return mapping.get(text, mapping.get(lowered, "noto_sans"))


def _load_pillow_font(
    image_font: object,
    size: int,
    *,
    bold: bool = False,
    font_family: str = "noto_sans",
) -> object:
    resolved_font_path = _default_font_path(bold=bold, font_family=font_family)
    if not resolved_font_path.is_file():
        raise RuntimeError(f"本地字体文件不存在: {resolved_font_path}")
    try:
        return image_font.truetype(str(resolved_font_path), size)  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError(f"加载本地字体失败: {resolved_font_path}") from exc


def _default_font_path(*, bold: bool, font_family: str = "noto_sans") -> Path:
    """Return one of the bundled font paths; arbitrary system paths are not allowed."""
    family = _normalize_image_font_family(font_family)
    regular_name, bold_name = _FONT_FAMILIES.get(
        family,
        _FONT_FAMILIES["noto_sans"],
    )
    return _FONT_DIR / (bold_name if bold else regular_name)
