import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.export_release_tree import export, read_allowlist, selected_paths


ROOT = Path(__file__).resolve().parents[2]


class ExportReleaseTreeTests(unittest.TestCase):
    def test_public_gitignore_protects_local_sources_artifacts_and_caches(self) -> None:
        text = (ROOT / "release" / "public.gitignore").read_text(encoding="utf-8")

        self.assertIn("reference/", text)
        self.assertIn(".env", text)
        self.assertIn("kb_corpus_build/indexes/", text)
        self.assertIn("kb_corpus_build/corpus/", text)
        self.assertIn("**/__pycache__/", text)

    def test_allowlist_selects_exact_files_and_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "allowlist.txt"
            path.write_text("README.md\nsrc/em_rag/api/\n", encoding="utf-8")
            exact, prefixes = read_allowlist(path)

        selected = selected_paths(
            ["README.md", "src/em_rag/api/app.py", "src/em_rag/evaluation/cli.py"],
            exact,
            prefixes,
        )
        self.assertEqual(selected, ["README.md", "src/em_rag/api/app.py"])

    def test_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "allowlist.txt"
            path.write_text("../secret\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_allowlist(path)

    def test_export_reads_head_bytes_not_dirty_working_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            output = Path(tmp) / "release"
            root.mkdir()
            (root / "README.md").write_text("dirty working tree\n", encoding="utf-8")
            allowlist = root / "allowlist.txt"
            allowlist.write_text("README.md\n", encoding="utf-8")
            with patch("scripts.export_release_tree.tracked_paths", return_value=["README.md"]), patch(
                "scripts.export_release_tree.committed_bytes", return_value=b"committed release\n"
            ):
                selected = export(root, output, allowlist)

            self.assertEqual(selected, ["README.md"])
            self.assertEqual((output / "README.md").read_bytes(), b"committed release\n")

    def test_export_maps_public_gitignore_to_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            output = Path(tmp) / "release"
            root.mkdir()
            allowlist = root / "allowlist.txt"
            allowlist.write_text("release/public.gitignore\n", encoding="utf-8")
            with patch(
                "scripts.export_release_tree.tracked_paths",
                return_value=["release/public.gitignore"],
            ), patch(
                "scripts.export_release_tree.committed_bytes",
                return_value=b"reference/\nkb_corpus_build/indexes/\n",
            ):
                export(root, output, allowlist)

            self.assertTrue((output / ".gitignore").is_file())
            self.assertEqual(
                (output / ".gitignore").read_bytes(),
                (output / "release" / "public.gitignore").read_bytes(),
            )
