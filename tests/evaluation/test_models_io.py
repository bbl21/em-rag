import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from unittest.mock import patch

from em_rag.evaluation.io import read_jsonl, write_jsonl
from em_rag.evaluation.models import EvalCase, Qrel, RankedEvidence, RetrievalRun


class ModelTests(unittest.TestCase):
    def test_answerable_case_from_dict_is_immutable(self):
        case = EvalCase.from_dict(
            {
                "query_id": "q-answerable",
                "query": "What limits the validity of the two-ray model?",
                "category": "propagation",
                "expected_facets": ["assumptions", "limitations"],
                "is_hard_negative": False,
                "requires_multiple_evidence": True,
                "split": "development",
                "language": "en",
            }
        )

        self.assertEqual(case.expected_facets, ("assumptions", "limitations"))
        with self.assertRaises(FrozenInstanceError):
            case.query = "changed"

    def test_hard_negative_case_accepts_empty_expected_facets(self):
        case = EvalCase.from_dict(
            {
                "query_id": "q-negative",
                "query": "What is the lunar soil permittivity in this corpus?",
                "category": "out_of_scope",
                "expected_facets": [],
                "is_hard_negative": True,
                "requires_multiple_evidence": False,
                "split": "adversarial",
            }
        )

        self.assertTrue(case.is_hard_negative)
        self.assertEqual(case.language, "en")

    def test_eval_case_rejects_empty_query(self):
        with self.assertRaises(ValueError):
            EvalCase.from_dict(
                {
                    "query_id": "q-empty",
                    "query": "  ",
                    "category": "propagation",
                    "expected_facets": [],
                    "is_hard_negative": False,
                    "requires_multiple_evidence": False,
                    "split": "regression",
                }
            )

    def test_eval_case_rejects_non_english_language(self):
        with self.assertRaises(ValueError):
            EvalCase.from_dict(
                {
                    "query_id": "q-language",
                    "query": "Explain free-space path loss.",
                    "category": "propagation",
                    "expected_facets": [],
                    "is_hard_negative": False,
                    "requires_multiple_evidence": False,
                    "split": "holdout",
                    "language": "fr",
                }
            )

    def test_eval_case_rejects_non_string_split_with_value_error(self):
        with self.assertRaises(ValueError):
            EvalCase.from_dict(
                {
                    "query_id": "q-split-type",
                    "query": "Explain free-space path loss.",
                    "category": "propagation",
                    "expected_facets": [],
                    "is_hard_negative": False,
                    "requires_multiple_evidence": False,
                    "split": ["development"],
                }
            )

    def test_qrel_accepts_graded_relevance(self):
        qrel = Qrel.from_dict(
            {
                "query_id": "q-answerable",
                "chunk_id": "chunk-7",
                "relevance": 3,
                "supported_facets": ["assumptions"],
                "confidence": 0.95,
                "judgment_source": "agent_adjudication",
            }
        )

        self.assertEqual(qrel.supported_facets, ("assumptions",))
        self.assertEqual(qrel.relevance, 3)

    def test_qrel_rejects_relevance_outside_scale(self):
        for relevance in (-1, 4):
            with self.subTest(relevance=relevance), self.assertRaises(ValueError):
                Qrel.from_dict(
                    {
                        "query_id": "q-answerable",
                        "chunk_id": "chunk-7",
                        "relevance": relevance,
                        "supported_facets": [],
                        "confidence": 0.8,
                        "judgment_source": "agent_pass_1",
                    }
                )

    def test_qrel_rejects_non_string_judgment_source_with_value_error(self):
        with self.assertRaises(ValueError):
            Qrel.from_dict(
                {
                    "query_id": "q-answerable",
                    "chunk_id": "chunk-7",
                    "relevance": 2,
                    "supported_facets": [],
                    "confidence": 0.8,
                    "judgment_source": ["agent_pass_1"],
                }
            )

    def test_retrieval_run_rejects_duplicate_evidence_rank(self):
        row = self._run_row()
        row["results"].append({**row["results"][0], "chunk_id": "chunk-8"})

        with self.assertRaises(ValueError):
            RetrievalRun.from_dict(row)

    def test_retrieval_run_rejects_missing_citation(self):
        row = self._run_row()
        del row["results"][0]["citation"]

        with self.assertRaises(ValueError):
            RetrievalRun.from_dict(row)

    def test_retrieval_run_builds_ranked_evidence(self):
        run = RetrievalRun.from_dict(self._run_row())

        self.assertEqual(run.results[0].rank, 1)
        self.assertIsInstance(run.results[0], RankedEvidence)
        self.assertEqual(run.confidence_threshold, 10.0)

    def test_retrieval_run_rejects_non_finite_confidence_threshold(self):
        for threshold in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(threshold=threshold):
                row = self._run_row()
                row["confidence_threshold"] = threshold

                with self.assertRaises(ValueError):
                    RetrievalRun.from_dict(row)

    @staticmethod
    def _run_row():
        return {
            "run_id": "run-1",
            "query_id": "q-answerable",
            "artifact_id": "artifact-1",
            "results": [
                {
                    "rank": 1,
                    "chunk_id": "chunk-7",
                    "score": 12.5,
                    "citation": "source.tex:120",
                    "text": "The model assumes a flat reflecting surface.",
                    "source_id": "source-1",
                }
            ],
            "degraded": False,
            "confidence_threshold": 10.0,
        }


class JsonlIoTests(unittest.TestCase):
    def test_write_then_read_is_byte_stable_utf8_lf(self):
        rows = [
            asdict(
                EvalCase.from_dict(
                    {
                        "query_id": "q-unicode",
                        "query": "When is the Fraunhofer approximation valid?",
                        "category": "field_regions",
                        "expected_facets": ["distance ≥ 2D²/λ"],
                        "is_hard_negative": False,
                        "requires_multiple_evidence": False,
                        "split": "regression",
                    }
                )
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"

            write_jsonl(first, rows)
            decoded = read_jsonl(first)
            write_jsonl(second, decoded)

            first_bytes = first.read_bytes()
            self.assertEqual(first_bytes, second.read_bytes())
            self.assertNotIn(b"\r\n", first_bytes)
            self.assertIn("≥".encode("utf-8"), first_bytes)
            self.assertEqual(decoded, [json.loads(first_bytes.decode("utf-8").strip())])

    def test_read_jsonl_rejects_blank_lines_and_garbage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            for content in ('{"query_id":"q1"}\n\n', '{"query_id":"q1"}\nnot-json\n'):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8", newline="\n")
                    with self.assertRaises(ValueError):
                        read_jsonl(path)

    def test_read_jsonl_rejects_non_object_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.jsonl"
            path.write_text('["not", "an", "object"]\n', encoding="utf-8", newline="\n")

            with self.assertRaises(ValueError):
                read_jsonl(path)

    def test_read_jsonl_rejects_non_finite_json_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "non-finite.jsonl"
            for constant in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(constant=constant):
                    path.write_text(
                        f'{{"query_id":"q1","score":{constant}}}\n',
                        encoding="utf-8",
                        newline="\n",
                    )
                    with self.assertRaises(ValueError):
                        read_jsonl(path)

    def test_run_identity_allows_one_run_id_across_queries(self):
        rows = [
            {"run_id": "shared-run", "query_id": "q-1"},
            {"run_id": "shared-run", "query_id": "q-2"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"

            write_jsonl(path, rows)

            self.assertEqual(read_jsonl(path), rows)

    def test_run_identity_rejects_duplicate_run_query_pair(self):
        rows = [
            {"run_id": "shared-run", "query_id": "q-1"},
            {"run_id": "shared-run", "query_id": "q-1"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-runs.jsonl"

            with self.assertRaises(ValueError):
                write_jsonl(path, rows)

    def test_read_jsonl_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            path.write_text(
                '{"query_id":"q1","query":"first"}\n'
                '{"query_id":"q1","query":"second"}\n',
                encoding="utf-8",
                newline="\n",
            )

            with self.assertRaises(ValueError):
                read_jsonl(path)

    def test_write_jsonl_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            rows = [
                {"query_id": "q1", "query": "first"},
                {"query_id": "q1", "query": "second"},
            ]

            with self.assertRaises(ValueError):
                write_jsonl(path, rows)

    def test_write_jsonl_rejects_disk_byte_mismatch_after_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mismatch.jsonl"
            with patch.object(Path, "read_bytes", return_value=b"corrupt") as read_bytes:
                with self.assertRaises(ValueError):
                    write_jsonl(path, [{"query_id": "q1", "query": "first"}])

            read_bytes.assert_called_once_with()

    def test_read_jsonl_rejects_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_bytes(b'{"query_id":"q1","query":"\xff"}\n')

            with self.assertRaises(ValueError):
                read_jsonl(path)


if __name__ == "__main__":
    unittest.main()
