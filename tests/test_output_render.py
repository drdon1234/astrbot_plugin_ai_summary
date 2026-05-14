from pathlib import Path
import asyncio
import tempfile
import unittest

from PIL import Image

from core import output_render


class RecordingImageFont:
    calls = []

    @classmethod
    def truetype(cls, path, size):
        cls.calls.append((Path(path).name, size))
        return object()


class OutputRenderFontTests(unittest.TestCase):
    def setUp(self):
        RecordingImageFont.calls = []

    def test_load_pillow_font_uses_bundled_noto_regular_and_bold(self):
        self.assertTrue(output_render._LOCAL_FONT_PATH.is_file())
        self.assertTrue(output_render._LOCAL_BOLD_FONT_PATH.is_file())

        output_render._load_pillow_font(RecordingImageFont, 23)
        output_render._load_pillow_font(RecordingImageFont, 38, bold=True)

        self.assertEqual(
            RecordingImageFont.calls,
            [
                ("NotoSansCJKsc-Regular.otf", 23),
                ("NotoSansCJKsc-Bold.otf", 38),
            ],
        )

    def test_brief_summary_renders_without_large_bottom_gap(self):
        output = Path(tempfile.gettempdir()) / "brief-summary-layout.png"

        output_render._render_text_image_file_sync(
            (
                "# 二次元女性角色的AIGC插画展示\n\n"
                "视频主要是多张二次元/AIGC风格女性角色插画的连续展示，场景以卧室、"
                "窗边和华丽室内空间为主，角色服装从浅色上衣切换到黑色职业装或亮面服饰。\n\n"
                "音频更像零散歌词或背景音乐，没有形成清晰叙事。"
            ),
            "markdown",
            output,
            960,
        )

        with Image.open(output) as image:
            self.assertEqual(image.width, 960)
            self.assertLess(image.height, 560)

    def test_default_render_width_is_chat_friendly(self):
        output = Path(tempfile.gettempdir()) / "default-summary-width.png"

        asyncio.run(
            output_render.render_summary_image_file(
                "# AI 视频总结\n\n简略总结内容。",
                "markdown",
                str(output),
            )
        )

        with Image.open(output) as image:
            self.assertEqual(image.width, 960)

    def test_professional_summary_uses_vertical_single_column_layout(self):
        output = Path(tempfile.gettempdir()) / "professional-summary-layout.png"
        items = "\n".join(
            f"- [{index:02d}:00] 这是一条较长的视频脉络说明，用来验证专业模式会自动换行，"
            "靠纵向延展保持手机和电脑端可读。"
            for index in range(12)
        )

        output_render._render_text_image_file_sync(
            f"# 专业视频总结\n\n## 视频脉络\n{items}",
            "markdown",
            output,
            960,
        )

        with Image.open(output) as image:
            self.assertEqual(image.width, 960)
            self.assertGreater(image.height, image.width)


if __name__ == "__main__":
    unittest.main()
