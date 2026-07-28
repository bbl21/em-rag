import unittest
from dataclasses import asdict

from em_rag.evaluation.models import EvalCase, RetrievalRun
from em_rag.evaluation.pooling import (
    JudgeRow,
    Judgment,
    build_blind_pool,
    merge_judgments,
)


class BlindPoolTests(unittest.TestCase):
    def test_pool_deduplicates_and_exposes_only_blinded_evidence(self):
        case = self._case("q001", "When is the two-ray model valid?")
        first = self._run("bm25", case.query_id, "shared", 1, 12.0)
        second = self._run("hybrid", case.query_id, "shared", 3, 0.82)

        rows = build_blind_pool([(case, (first, second))], seed=17)

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            asdict(rows[0]),
            {
                "judgment_id": rows[0].judgment_id,
                "query_id": "q001",
                "query": "When is the two-ray model valid?",
                "expected_facets": ("assumptions", "limitations"),
                "chunk_id": "shared",
                "source_id": "source-shared",
                "citation": "source.tex:10-14",
                "text": "Full text for shared.",
            },
        )
        self.assertTrue(rows[0].judgment_id.startswith("j_"))
        self.assertNotIn("run_id", asdict(rows[0]))
        self.assertNotIn("score", asdict(rows[0]))
        self.assertNotIn("rank", asdict(rows[0]))

    def test_pool_order_is_deterministic_for_a_fixed_seed(self):
        case = self._case("q001", "When is the two-ray model valid?")
        runs = tuple(
            self._run("run-a", case.query_id, chunk_id, rank, float(rank))
            for rank, chunk_id in enumerate(("a", "b", "c", "d"), start=1)
        )

        first = build_blind_pool([(case, runs)], seed=9)
        second = build_blind_pool([(case, tuple(reversed(runs)))], seed=9)

        self.assertEqual(first, second)

    @staticmethod
    def _case(query_id, query):
        return EvalCase.from_dict(
            {
                "query_id": query_id,
                "query": query,
                "category": "propagation",
                "expected_facets": ["assumptions", "limitations"],
                "is_hard_negative": False,
                "requires_multiple_evidence": True,
                "split": "development",
            }
        )

    @staticmethod
    def _run(run_id, query_id, chunk_id, rank, score):
        return RetrievalRun.from_dict(
            {
                "run_id": run_id,
                "query_id": query_id,
                "artifact_id": f"artifact-{run_id}",
                "results": [
                    {
                        "rank": rank,
                        "chunk_id": chunk_id,
                        "score": score,
                        "citation": "source.tex:10-14",
                        "text": f"Full text for {chunk_id}.",
                        "source_id": f"source-{chunk_id}",
                    }
                ],
                "degraded": False,
                "confidence_threshold": 0.5,
            }
        )


class JudgmentTests(unittest.TestCase):
    def test_direct_judge_row_rejects_blank_query_and_chunk_identity(self):
        valid = {
            "judgment_id": "j_a",
            "query_id": "q001",
            "query": "What is the valid range?",
            "expected_facets": ("range",),
            "chunk_id": "chunk-1",
            "source_id": "source-1",
            "citation": "source.tex:10-14",
            "text": "The range is stated here.",
        }
        for field in ("query_id", "query", "chunk_id"):
            with self.subTest(field=field):
                values = {**valid, field: "  "}
                with self.assertRaisesRegex(ValueError, field):
                    JudgeRow(**values)

    def test_direct_judgment_rejects_blank_id(self):
        with self.assertRaisesRegex(ValueError, "judgment_id"):
            self._direct_judgment(judgment_id="  ")

    def test_direct_judgment_rejects_relevance_outside_scale(self):
        with self.assertRaisesRegex(ValueError, "relevance"):
            self._direct_judgment(relevance=99)

    def test_direct_judgment_rejects_non_finite_confidence(self):
        with self.assertRaisesRegex(ValueError, "confidence"):
            self._direct_judgment(confidence=float("nan"))

    def test_direct_judgment_rejects_confidence_outside_range(self):
        for value in (-0.01, 1.01):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    self._direct_judgment(confidence=value)

    def test_direct_judgment_rejects_non_numeric_confidence(self):
        for value in (True, "0.9"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    self._direct_judgment(confidence=value)

    def test_direct_judgment_normalizes_supported_facets_and_confidence(self):
        judgment = self._direct_judgment(
            supported_facets=["assumptions"],
            confidence=1,
        )

        self.assertEqual(judgment.supported_facets, ("assumptions",))
        self.assertIs(type(judgment.supported_facets), tuple)
        self.assertEqual(judgment.confidence, 1.0)
        self.assertIs(type(judgment.confidence), float)

    def test_direct_judgment_rejects_blank_source_quote(self):
        with self.assertRaisesRegex(ValueError, "source_quote"):
            self._direct_judgment(source_quote="  ")

    def test_direct_judgment_rejects_blank_reason(self):
        with self.assertRaisesRegex(ValueError, "reason"):
            self._direct_judgment(reason="")

    def test_direct_judgment_rejects_non_boolean_flags(self):
        for field in ("scope_correct", "citation_supported", "pollution"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    self._direct_judgment(**{field: 1})

    def test_direct_judgment_rejects_invalid_supported_facets_shape(self):
        invalid_values = (
            "assumptions",
            ("assumptions", "assumptions"),
            ("assumptions", "  "),
            ("assumptions", 1),
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "supported_facets"):
                    self._direct_judgment(supported_facets=value)

    def test_judgment_rejects_missing_source_quote(self):
        row = self._judgment("j_a", relevance=3)
        row["source_quote"] = "  "

        with self.assertRaisesRegex(ValueError, "source_quote"):
            Judgment.from_dict(row)

    def test_merge_surfaces_disagreement_and_requires_adjudication(self):
        pass1 = [
            Judgment.from_dict(self._judgment("j_a", relevance=3, confidence=0.9)),
            Judgment.from_dict(self._judgment("j_b", relevance=0, confidence=0.9)),
        ]
        pass2 = [
            Judgment.from_dict(self._judgment("j_a", relevance=0, confidence=0.9)),
            Judgment.from_dict(self._judgment("j_b", relevance=3, confidence=0.9)),
        ]

        merged = merge_judgments(pass1, pass2, adjudication=[])

        self.assertEqual(merged.agreement.exact_agreement, 0.0)
        self.assertLess(merged.agreement.weighted_cohen_kappa, 0.8)
        self.assertEqual(merged.agreement.status, "not_release_eligible")
        self.assertEqual(merged.agreement.adjudication_ids, ("j_a", "j_b"))
        self.assertEqual(merged.agreement.unresolved_ids, ("j_a", "j_b"))
        self.assertIsNone(merged.rows[0].final)
        self.assertEqual(merged.rows[0].pass1.relevance, 3)
        self.assertEqual(merged.rows[0].pass2.relevance, 0)

    def test_one_grade_disagreement_blocks_release_even_with_high_kappa(self):
        grades = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        pass1 = [
            Judgment.from_dict(self._judgment(f"j_{index}", relevance=grade))
            for index, grade in enumerate(grades)
        ]
        pass2_grades = list(grades)
        pass2_grades[6] = 1
        pass2 = [
            Judgment.from_dict(self._judgment(f"j_{index}", relevance=grade))
            for index, grade in enumerate(pass2_grades)
        ]

        merged = merge_judgments(pass1, pass2, adjudication=[])

        self.assertEqual(merged.agreement.exact_agreement, 0.9)
        self.assertGreater(merged.agreement.weighted_cohen_kappa, 0.8)
        self.assertEqual(merged.agreement.adjudication_ids, ())
        self.assertEqual(merged.agreement.unresolved_ids, ("j_6",))
        self.assertIsNone(merged.rows[6].final)
        self.assertFalse(merged.agreement.release_eligible)
        self.assertEqual(merged.agreement.status, "not_release_eligible")

    def test_one_grade_unresolved_accepts_human_without_declared_sample(self):
        pass1 = [Judgment.from_dict(self._judgment("j_minor", relevance=2))]
        pass2 = [Judgment.from_dict(self._judgment("j_minor", relevance=1))]
        human = [Judgment.from_dict(self._judgment("j_minor", relevance=2))]

        try:
            merged = merge_judgments(
                pass1,
                pass2,
                adjudication=[],
                human_calibration=human,
            )
        except ValueError as exc:
            self.fail(f"non-mandatory unresolved row must allow calibration: {exc}")

        row = merged.rows[0]
        self.assertEqual(merged.agreement.adjudication_ids, ())
        self.assertEqual(merged.agreement.unresolved_ids, ())
        self.assertEqual(row.pass1.relevance, 2)
        self.assertEqual(row.pass2.relevance, 1)
        self.assertIsNone(row.adjudication)
        self.assertEqual(row.human_calibration.relevance, 2)
        self.assertEqual(row.final.relevance, 2)

    def test_facet_unresolved_accepts_human_without_declared_sample(self):
        left = self._judgment("j_facets_calibration", relevance=2)
        right = self._judgment("j_facets_calibration", relevance=2)
        right["supported_facets"] = ["limitations"]
        human = self._judgment("j_facets_calibration", relevance=2)

        try:
            merged = merge_judgments(
                [Judgment.from_dict(left)],
                [Judgment.from_dict(right)],
                adjudication=[],
                human_calibration=[Judgment.from_dict(human)],
            )
        except ValueError as exc:
            self.fail(f"facet-unresolved row must allow calibration: {exc}")

        row = merged.rows[0]
        self.assertEqual(merged.agreement.adjudication_ids, ())
        self.assertEqual(merged.agreement.unresolved_ids, ())
        self.assertEqual(row.pass1.supported_facets, ("assumptions",))
        self.assertEqual(row.pass2.supported_facets, ("limitations",))
        self.assertIsNone(row.adjudication)
        self.assertEqual(row.human_calibration.supported_facets, ("assumptions",))
        self.assertEqual(row.final, row.human_calibration)

    def test_supported_facet_disagreement_is_unresolved_but_not_mandatory(self):
        left = self._judgment("j_facets", relevance=2)
        right = self._judgment("j_facets", relevance=2)
        right["supported_facets"] = ["limitations"]

        merged = merge_judgments(
            [Judgment.from_dict(left)],
            [Judgment.from_dict(right)],
            adjudication=[],
        )

        self.assertEqual(merged.agreement.weighted_cohen_kappa, 1.0)
        self.assertEqual(merged.agreement.adjudication_ids, ())
        self.assertEqual(merged.agreement.unresolved_ids, ("j_facets",))
        self.assertIsNone(merged.rows[0].final)
        self.assertFalse(merged.agreement.release_eligible)

    def test_scope_or_citation_disagreement_and_low_confidence_are_adjudicated(self):
        pass1 = [
            Judgment.from_dict(
                self._judgment("j_scope", 2, scope_correct=True, confidence=0.9)
            ),
            Judgment.from_dict(
                self._judgment("j_citation", 2, citation_supported=True, confidence=0.9)
            ),
            Judgment.from_dict(self._judgment("j_low", 2, confidence=0.74)),
        ]
        pass2 = [
            Judgment.from_dict(
                self._judgment("j_scope", 2, scope_correct=False, confidence=0.9)
            ),
            Judgment.from_dict(
                self._judgment(
                    "j_citation",
                    2,
                    citation_supported=False,
                    confidence=0.9,
                )
            ),
            Judgment.from_dict(self._judgment("j_low", 2, confidence=0.9)),
        ]

        merged = merge_judgments(pass1, pass2, adjudication=[])

        self.assertEqual(
            merged.agreement.adjudication_ids,
            ("j_citation", "j_low", "j_scope"),
        )
        self.assertFalse(merged.agreement.release_eligible)

    def test_adjudication_resolves_flagged_row_without_erasing_passes(self):
        pass1 = [Judgment.from_dict(self._judgment("j_a", relevance=3))]
        pass2 = [Judgment.from_dict(self._judgment("j_a", relevance=1))]
        adjudication = [Judgment.from_dict(self._judgment("j_a", relevance=2))]

        merged = merge_judgments(pass1, pass2, adjudication)

        self.assertEqual(merged.rows[0].pass1.relevance, 3)
        self.assertEqual(merged.rows[0].pass2.relevance, 1)
        self.assertEqual(merged.rows[0].adjudication.relevance, 2)
        self.assertEqual(merged.rows[0].final.relevance, 2)
        self.assertEqual(merged.agreement.unresolved_ids, ())

    def test_human_calibration_resolves_an_unresolved_row_and_is_audited(self):
        pass1 = [Judgment.from_dict(self._judgment("j_a", relevance=3))]
        pass2 = [Judgment.from_dict(self._judgment("j_a", relevance=1))]
        human = [Judgment.from_dict(self._judgment("j_a", relevance=2))]

        try:
            merged = merge_judgments(
                pass1,
                pass2,
                adjudication=[],
                human_calibration=human,
            )
        except TypeError as exc:
            self.fail(f"human calibration input must be supported: {exc}")

        self.assertEqual(merged.rows[0].pass1.relevance, 3)
        self.assertEqual(merged.rows[0].pass2.relevance, 1)
        self.assertEqual(merged.rows[0].human_calibration.relevance, 2)
        self.assertEqual(merged.rows[0].final.relevance, 2)
        self.assertEqual(merged.agreement.unresolved_ids, ())

    def test_human_calibration_rejects_resolved_unsampled_rows(self):
        pass1 = [Judgment.from_dict(self._judgment("j_a", relevance=2))]
        pass2 = [Judgment.from_dict(self._judgment("j_a", relevance=2))]
        human = [Judgment.from_dict(self._judgment("j_a", relevance=1))]

        try:
            with self.assertRaisesRegex(ValueError, "unresolved or sampled"):
                merge_judgments(
                    pass1,
                    pass2,
                    adjudication=[],
                    human_calibration=human,
                )
        except TypeError as exc:
            self.fail(f"human calibration input must be supported: {exc}")

    def test_human_calibration_accepts_a_declared_sample_and_keeps_passes(self):
        pass1 = [Judgment.from_dict(self._judgment("j_a", relevance=2))]
        pass2 = [Judgment.from_dict(self._judgment("j_a", relevance=2))]
        human = [Judgment.from_dict(self._judgment("j_a", relevance=2))]

        try:
            merged = merge_judgments(
                pass1,
                pass2,
                adjudication=[],
                human_calibration=human,
                sampled_ids=("j_a",),
            )
        except TypeError as exc:
            self.fail(f"declared calibration samples must be supported: {exc}")

        self.assertEqual(merged.rows[0].pass1.relevance, 2)
        self.assertEqual(merged.rows[0].pass2.relevance, 2)
        self.assertEqual(merged.rows[0].human_calibration.relevance, 2)
        self.assertEqual(merged.rows[0].final.relevance, 2)

    @classmethod
    def _direct_judgment(cls, **overrides):
        values = cls._judgment("j_a", relevance=2)
        values["supported_facets"] = tuple(values["supported_facets"])
        values.update(overrides)
        return Judgment(**values)

    @staticmethod
    def _judgment(
        judgment_id,
        relevance,
        confidence=0.9,
        scope_correct=True,
        citation_supported=True,
    ):
        return {
            "judgment_id": judgment_id,
            "relevance": relevance,
            "supported_facets": ["assumptions"],
            "scope_correct": scope_correct,
            "citation_supported": citation_supported,
            "pollution": False,
            "confidence": confidence,
            "source_quote": "The source states this directly.",
            "reason": "The quoted span supports the assigned grade.",
        }


if __name__ == "__main__":
    unittest.main()
