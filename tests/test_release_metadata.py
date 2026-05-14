from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parent.parent


class ReleaseMetadataTests(unittest.TestCase):
    def test_release_version_is_0_1_4_everywhere(self):
        metadata = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertRegex(metadata, r"(?m)^version:\s+v0\.1\.4$")
        self.assertRegex(main, r'(?m)^\s+"0\.1\.4",\s*$')
        self.assertIn("Version-v0.1.4-green", readme)

    def test_removed_renderer_names_do_not_remain(self):
        checked_paths = [
            ROOT / "README.md",
            ROOT / "docs" / "ARCHITECTURE.md",
            ROOT / "_conf_schema.json",
            ROOT / "core" / "output_render.py",
        ]
        forbidden = re.compile(r"arialuni|imgkit|wkhtml", re.IGNORECASE)

        leftovers = []
        for path in checked_paths:
            text = path.read_text(encoding="utf-8")
            for match in forbidden.finditer(text):
                leftovers.append(f"{path.relative_to(ROOT)}:{match.group(0)}")

        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
