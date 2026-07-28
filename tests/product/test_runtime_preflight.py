import json
import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from em_rag.adapters.legacy_retrieval import LegacyRetrievalAdapter


class RuntimePreflightTests(unittest.TestCase):
    @staticmethod
    def _write_bm25(root: Path, *, docstore_chunk_id: str = "chunk-1") -> None:
        bm25 = root / "kb_corpus_build" / "indexes" / "bm25"
        bm25.mkdir(parents=True, exist_ok=True)
        index = {
            "doc_count": 1,
            "documents": [{"chunk_id": "chunk-1", "tokens": ["antenna"]}],
        }
        with (bm25 / "bm25_index.pkl").open("wb") as handle:
            pickle.dump(index, handle)
        (bm25 / "bm25_docstore.jsonl").write_text(
            json.dumps({"chunk_id": docstore_chunk_id, "content_md": "Antenna evidence"}) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _bm25_module(root: Path):
        def load_bm25(build_root: Path):
            with (build_root / "indexes" / "bm25" / "bm25_index.pkl").open("rb") as handle:
                index = pickle.load(handle)
            rows = [
                json.loads(line)
                for line in (build_root / "indexes" / "bm25" / "bm25_docstore.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            return index, {row["chunk_id"]: row for row in rows}

        return SimpleNamespace(load_bm25=load_bm25, score_bm25=lambda index, query: [])

    @staticmethod
    def _write_structured(root: Path, *, related_chunk_id: str = "chunk-1") -> None:
        structured = root / "kb_corpus_build" / "indexes" / "structured"
        structured.mkdir(parents=True, exist_ok=True)
        for filename, table in (
            ("formula.sqlite", "formulas"),
            ("terms.sqlite", "terms"),
            ("propagation_models.sqlite", "propagation_models"),
        ):
            connection = sqlite3.connect(structured / filename)
            try:
                connection.execute(
                    f"create table {table} (related_chunk_ids_json text, search_text text)"
                )
                connection.execute(
                    f"insert into {table} values (?, ?)",
                    (json.dumps([related_chunk_id]), "antenna"),
                )
                connection.commit()
            finally:
                connection.close()

    def test_bm25_preflight_loads_scores_and_validates_chunk_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_bm25(root)
            adapter = LegacyRetrievalAdapter(root)
            adapter._module = self._bm25_module(root)

            ok_status, _ = adapter.bm25_preflight(max_age_seconds=0)
            self._write_bm25(root, docstore_chunk_id="wrong-chunk")
            bad_status, bad_error = adapter.bm25_preflight(max_age_seconds=0)

        self.assertEqual(ok_status, "ok")
        self.assertEqual(bad_status, "error")
        self.assertIn("chunk IDs are inconsistent", bad_error)

    def test_structured_preflight_checks_sqlite_and_chunk_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_bm25(root)
            self._write_structured(root)
            adapter = LegacyRetrievalAdapter(root)

            ok_status, _ = adapter.structured_preflight(max_age_seconds=0)
            terms = root / "kb_corpus_build" / "indexes" / "structured" / "terms.sqlite"
            connection = sqlite3.connect(terms)
            try:
                connection.execute(
                    "update terms set related_chunk_ids_json = ?", (json.dumps(["missing"]),)
                )
                connection.commit()
            finally:
                connection.close()
            bad_status, bad_error = adapter.structured_preflight(max_age_seconds=0)

        self.assertEqual(ok_status, "ok")
        self.assertEqual(bad_status, "error")
        self.assertIn("invalid chunk reference", bad_error)

    def test_structured_preflight_reports_absent_indexes_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = LegacyRetrievalAdapter(Path(tmp))
            status, error = adapter.structured_preflight()

        self.assertEqual(status, "unavailable")
        self.assertEqual(error, "")


if __name__ == "__main__":
    unittest.main()
