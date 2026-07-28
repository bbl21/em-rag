import math
import unittest

from em_rag.evaluation.metrics import (
    evaluate_run,
    hard_negative_false_positive,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from em_rag.evaluation.models import EvalCase, Qrel, RetrievalRun


class RetrievalMetricTests(unittest.TestCase):
    def setUp(self):
        self.case = self._case("q-graded")
        self.qrels = [
            self._qrel("q-graded", "doc-3", 3),
            self._qrel("q-graded", "doc-2", 2),
            self._qrel("q-graded", "doc-0", 0),
        ]
        self.run = self._run(
            "q-graded",
            [(1, "doc-0", 0.95), (2, "doc-2", 0.80), (3, "doc-3", 0.70)],
            threshold=0.90,
        )

    def test_hand_calculated_metrics_for_reordered_graded_results(self):
        expected_ndcg = (3 / math.log2(3)) / (7 + 3 / math.log2(3))

        self.assertEqual(recall_at_k(self.qrels, self.run, 2), 0.5)
        self.assertEqual(precision_at_k(self.qrels, self.run, 2), 0.5)
        self.assertEqual(mrr_at_k(self.qrels, self.run, 2), 0.5)
        self.assertAlmostEqual(ndcg_at_k(self.qrels, self.run, 2), expected_ndcg)

    def test_sparse_ranks_use_explicit_rank_for_cutoff_and_discount(self):
        qrels = [self._qrel("q-sparse", "doc-relevant", 3)]
        run = self._run(
            "q-sparse",
            [(2, "doc-relevant", 0.8)],
            threshold=0.5,
        )

        self.assertEqual(recall_at_k(qrels, run, 1), 0.0)
        self.assertEqual(precision_at_k(qrels, run, 1), 0.0)
        self.assertEqual(mrr_at_k(qrels, run, 1), 0.0)
        self.assertEqual(ndcg_at_k(qrels, run, 1), 0.0)
        self.assertEqual(recall_at_k(qrels, run, 2), 1.0)
        self.assertEqual(precision_at_k(qrels, run, 2), 0.5)
        self.assertEqual(mrr_at_k(qrels, run, 2), 0.5)
        self.assertAlmostEqual(ndcg_at_k(qrels, run, 2), 1 / math.log2(3))

    def test_public_at_k_metrics_reject_invalid_k(self):
        metrics = (recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k)
        for metric in metrics:
            for invalid_k in (0, -1, True, 1.5):
                with self.subTest(metric=metric.__name__, k=invalid_k):
                    with self.assertRaises(ValueError):
                        metric(self.qrels, self.run, invalid_k)

    def test_evaluate_run_rejects_query_id_mismatch(self):
        mismatched_case = self._case("q-other")

        with self.assertRaises(ValueError):
            evaluate_run(mismatched_case, self.qrels, self.run)

    def test_unjudged_documents_count_as_zero_relevance(self):
        run = self._run(
            "q-graded",
            [(1, "unjudged", 0.99), (2, "doc-3", 0.80)],
            threshold=0.90,
        )

        self.assertEqual(precision_at_k(self.qrels, run, 1), 0.0)
        self.assertEqual(mrr_at_k(self.qrels, run, 2), 0.5)

    def test_zero_relevant_qrels_return_zero_without_division_error(self):
        qrels = [self._qrel("q-zero", "doc-0", 0)]
        run = self._run("q-zero", [(1, "doc-0", 0.1)], threshold=0.5)

        self.assertEqual(recall_at_k(qrels, run, 10), 0.0)
        self.assertEqual(precision_at_k(qrels, run, 5), 0.0)
        self.assertEqual(mrr_at_k(qrels, run, 10), 0.0)
        self.assertEqual(ndcg_at_k(qrels, run, 10), 0.0)

    def test_hard_negative_uses_run_confidence_threshold_inclusively(self):
        below = self._run("q-negative", [(1, "doc-a", 4.99)], threshold=5.0)
        equal = self._run("q-negative", [(1, "doc-a", 5.0)], threshold=5.0)

        self.assertFalse(hard_negative_false_positive(below))
        self.assertTrue(hard_negative_false_positive(equal))

    def test_evaluate_run_returns_standard_metric_record(self):
        result = evaluate_run(self.case, self.qrels, self.run)

        self.assertEqual(
            result,
            {
                "query_id": "q-graded",
                "recall_at_10": 1.0,
                "recall_at_50": 1.0,
                "mrr_at_10": 0.5,
                "ndcg_at_10": ndcg_at_k(self.qrels, self.run, 10),
                "precision_at_5": 0.4,
                "hard_negative_false_positive": False,
            },
        )

    def test_evaluate_run_reports_hard_negative_false_positive(self):
        case = self._case("q-negative", hard_negative=True)
        run = self._run("q-negative", [(1, "unjudged", 2.0)], threshold=2.0)

        result = evaluate_run(case, [], run)

        self.assertTrue(result["hard_negative_false_positive"])
        self.assertEqual(result["recall_at_10"], 0.0)

    @staticmethod
    def _case(query_id, hard_negative=False):
        return EvalCase.from_dict(
            {
                "query_id": query_id,
                "query": "A metric fixture query",
                "category": "evaluation",
                "expected_facets": [] if hard_negative else ["answer"],
                "is_hard_negative": hard_negative,
                "requires_multiple_evidence": False,
                "split": "adversarial" if hard_negative else "development",
            }
        )

    @staticmethod
    def _qrel(query_id, chunk_id, relevance):
        return Qrel.from_dict(
            {
                "query_id": query_id,
                "chunk_id": chunk_id,
                "relevance": relevance,
                "supported_facets": [],
                "confidence": 1.0,
                "judgment_source": "human_calibration",
            }
        )

    @staticmethod
    def _run(query_id, ranked, threshold):
        return RetrievalRun.from_dict(
            {
                "run_id": f"run-{query_id}",
                "query_id": query_id,
                "artifact_id": "artifact-1",
                "results": [
                    {
                        "rank": rank,
                        "chunk_id": chunk_id,
                        "score": score,
                        "citation": f"source.tex:{rank}",
                        "text": f"Evidence {chunk_id}",
                        "source_id": "source-1",
                    }
                    for rank, chunk_id, score in ranked
                ],
                "degraded": False,
                "confidence_threshold": threshold,
            }
        )


if __name__ == "__main__":
    unittest.main()
