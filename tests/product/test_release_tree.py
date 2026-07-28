import tempfile
import unittest
from pathlib import Path

from scripts.check_release_tree import audit


class ReleaseTreeTests(unittest.TestCase):
    def test_clean_allowlist_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            for relative in ("README.md", "LICENSE", "config/source-catalog.json", "docs/data-sources.md", "Dockerfile", "compose.yaml"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("safe\n", encoding="utf-8")
                files.append(path)

            findings = audit(root, files)

        self.assertEqual(findings, [])

    def test_controlled_path_and_absolute_user_path_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controlled = root / "reference" / "source.pdf"
            controlled.parent.mkdir(parents=True)
            controlled.write_bytes(b"data")
            text = root / "README.md"
            text.write_text("C:\\Users\\private\\model", encoding="utf-8")

            findings = audit(root, [controlled, text])

        self.assertTrue(any("forbidden tracked path" in item for item in findings))
        self.assertTrue(any("sensitive text pattern" in item for item in findings))


if __name__ == "__main__":
    unittest.main()
