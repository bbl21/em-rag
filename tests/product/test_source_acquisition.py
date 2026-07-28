import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from em_rag.bootstrap.__main__ import main
from em_rag.bootstrap.sources import SourceAcquisitionError, SourceSpec, acquire_source, load_source_catalog


class FakeResponse(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def fake_opener(data):
    return lambda _request, timeout: FakeResponse(data)


def spec(data, **changes):
    defaults = dict(source_id="public-pdf", url="https://example.org/book.pdf", filename="book.pdf",
                    sha256=hashlib.sha256(data).hexdigest(), max_bytes=1024 * 1024, file_type="pdf")
    defaults.update(changes)
    return SourceSpec(**defaults)


class SourceAcquisitionTests(unittest.TestCase):
    def test_downloads_verifies_pdf_and_writes_receipt(self):
        data = b"%PDF-1.7\npublic data"
        with tempfile.TemporaryDirectory() as tmp:
            result = acquire_source(spec(data), Path(tmp), {"example.org"}, opener=fake_opener(data))
            self.assertEqual(result.status, "acquired")
            self.assertEqual(result.path.read_bytes(), data)
            self.assertEqual(json.loads(result.receipt_path.read_text(encoding="utf-8"))["bytes"], len(data))

    def test_rejects_hash_mismatch_and_removes_partial(self):
        data = b"%PDF-1.7\nwrong"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(SourceAcquisitionError, "SHA-256"):
                acquire_source(spec(data, sha256="0" * 64), root, {"example.org"}, opener=fake_opener(data))
            self.assertFalse((root / "book.pdf.partial").exists())

    def test_rejects_unallowlisted_host(self):
        data = b"%PDF-1.7\n"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SourceAcquisitionError, "allowlisted"):
                acquire_source(spec(data), Path(tmp), {"other.example"}, opener=fake_opener(data))

    def test_manual_source_never_calls_opener(self):
        source = spec(b"x", acquisition_mode="manual_download", sha256="")
        result = acquire_source(source, Path(tempfile.gettempdir()), {"example.org"}, opener=lambda *_a, **_k: self.fail("called"))
        self.assertEqual(result.status, "manual_required")
        self.assertIn("Provide book.pdf manually", result.message)

    def test_rejects_unsafe_source_id_before_writing(self):
        data = b"%PDF-1.7\\npublic data"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "reference"
            with self.assertRaisesRegex(SourceAcquisitionError, "source_id"):
                acquire_source(spec(data, source_id="../escaped"), root, {"example.org"}, opener=fake_opener(data))
            self.assertFalse((Path(tmp) / "escaped.receipt.json").exists())

    def test_rejects_root_directory_source_id_for_archives(self):
        data = b"PK\\x03\\x04"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SourceAcquisitionError, "source_id"):
                acquire_source(spec(data, source_id=".", filename="source.zip", file_type="zip"), Path(tmp), {"example.org"}, opener=fake_opener(data))

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as bundle: bundle.writestr("../escape.txt", "bad")
            data = archive.getvalue()
            with self.assertRaisesRegex(SourceAcquisitionError, "unsafe"):
                acquire_source(spec(data, filename="bad.zip", file_type="zip"), Path(tmp), {"example.org"}, opener=fake_opener(data))

    def test_catalog_loader_validates_https_and_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sources.json"
            path.write_text(json.dumps({"sources": [spec(b"x").__dict__]}), encoding="utf-8")
            self.assertEqual(load_source_catalog(path, {"example.org"})[0].source_id, "public-pdf")
            path.write_text(json.dumps([dict(spec(b"x").__dict__, url="http://example.org/x")]), encoding="utf-8")
            with self.assertRaisesRegex(SourceAcquisitionError, "HTTPS"):
                load_source_catalog(path, {"example.org"})
            path.write_text(json.dumps([dict(spec(b"x").__dict__, acquisition_mode="manual_download", url="http://example.org/x")]), encoding="utf-8")
            with self.assertRaisesRegex(SourceAcquisitionError, "HTTPS"):
                load_source_catalog(path, {"example.org"})

    def test_cli_rejects_unreadable_catalogue_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "bad.json"
            catalog.write_text("{", encoding="utf-8")
            self.assertEqual(main(["acquire-sources", "--project-root", str(root), "--catalog", str(catalog)]), 2)


if __name__ == "__main__":
    unittest.main()
