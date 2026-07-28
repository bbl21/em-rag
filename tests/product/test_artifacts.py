import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from em_rag.adapters.artifacts import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def test_missing_artifact_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = ArtifactStore(Path(tmp)).snapshot()

        self.assertFalse(snapshot.ready)
        self.assertEqual(snapshot.artifact_id, "unavailable")
        self.assertEqual(
            set(snapshot.missing),
            {"bm25_index", "bm25_docstore", "vector_index", "vector_metadata", "vector_docstore"},
        )

    def test_ready_artifact_has_stable_checksum_identity_and_quality_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "kb_corpus_build" / "indexes" / "bm25" / "bm25_index.pkl"
            docstore = index.with_name("bm25_docstore.jsonl")
            index.parent.mkdir(parents=True)
            index.write_bytes(b"index")
            docstore.write_text(json.dumps({"chunk_id": "c1"}) + "\n", encoding="utf-8")
            vector = root / "kb_corpus_build" / "indexes" / "vector"
            vector.mkdir(parents=True)
            (vector / "faiss.index").write_bytes(b"vector-index")
            (vector / "index_metadata.json").write_text(json.dumps({"model": "test"}), encoding="utf-8")
            (vector / "docstore.jsonl").write_text(json.dumps({"chunk_id": "c1"}) + "\n", encoding="utf-8")
            report = root / "kb_corpus_build" / "eval" / "retrieval_quality_v2" / "reports" / "judgment_release_status.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"status": "NEEDS_CALIBRATION"}), encoding="utf-8")

            snapshot = ArtifactStore(root).snapshot()
            checksums = {
                "bm25_docstore": hashlib.sha256(docstore.read_bytes()).hexdigest(),
                "bm25_index": hashlib.sha256(index.read_bytes()).hexdigest(),
                "vector_docstore": hashlib.sha256((vector / "docstore.jsonl").read_bytes()).hexdigest(),
                "vector_index": hashlib.sha256((vector / "faiss.index").read_bytes()).hexdigest(),
                "vector_metadata": hashlib.sha256((vector / "index_metadata.json").read_bytes()).hexdigest(),
            }
            identity = hashlib.sha256(
                "\n".join(f"{name}:{checksums[name]}" for name in sorted(checksums)).encode()
            ).hexdigest()[:16]

        self.assertTrue(snapshot.ready)
        self.assertEqual(snapshot.artifact_id, f"em-rag-{identity}")
        self.assertEqual(snapshot.document_count, 1)
        self.assertEqual(snapshot.quality_status, "needs_calibration")

    def test_bm25_capability_is_independent_of_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root)
            bm25_index = root / "kb_corpus_build" / "indexes" / "bm25" / "bm25_index.pkl"
            bm25_docstore = bm25_index.with_name("bm25_docstore.jsonl")
            bm25_index.parent.mkdir(parents=True)
            bm25_index.write_bytes(b"index")
            bm25_docstore.write_text(json.dumps({"chunk_id": "bm25"}) + "\n", encoding="utf-8")

            snapshot = store.snapshot()

        self.assertTrue(snapshot.capabilities["bm25"])
        self.assertFalse(snapshot.capabilities["vector"])

    def test_vector_capability_is_independent_of_bm25(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root)
            vector = root / "kb_corpus_build" / "indexes" / "vector"
            vector.mkdir(parents=True)
            (vector / "faiss.index").write_bytes(b"index")
            (vector / "index_metadata.json").write_text(json.dumps({"model": "test"}), encoding="utf-8")
            (vector / "docstore.jsonl").write_text(json.dumps({"chunk_id": "v1"}) + "\n", encoding="utf-8")

            snapshot = store.snapshot()

        self.assertFalse(snapshot.ready)
        self.assertFalse(snapshot.capabilities["bm25"])
        self.assertTrue(snapshot.capabilities["vector"])

    def test_structured_capability_requires_all_expected_databases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root)
            structured = root / "kb_corpus_build" / "indexes" / "structured"
            structured.mkdir(parents=True)
            (structured / "formula.sqlite").write_bytes(b"partial")

            partial = store.snapshot()
            (structured / "terms.sqlite").write_bytes(b"terms")
            (structured / "propagation_models.sqlite").write_bytes(b"models")
            complete = store.snapshot()

        self.assertFalse(partial.capabilities["structured"])
        self.assertTrue(complete.capabilities["structured"])

    def test_quality_status_cache_is_invalidated_when_report_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root)
            for path in store.required_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact\n")
            report = (
                root
                / "kb_corpus_build"
                / "eval"
                / "retrieval_quality_v2"
                / "reports"
                / "judgment_release_status.json"
            )
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"status": "NEEDS_CALIBRATION"}), encoding="utf-8")

            first = store.snapshot()
            report.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            second = store.snapshot()

        self.assertEqual(first.quality_status, "needs_calibration")
        self.assertEqual(second.quality_status, "pass")


if __name__ == "__main__":
    unittest.main()
