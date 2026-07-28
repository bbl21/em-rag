import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import em_rag.evaluation.gates as gates
from em_rag.evaluation.gates import aggregate_quality_gate
from em_rag.evaluation.io import read_jsonl, write_jsonl
from em_rag.evaluation.models import EvalCase, Qrel, RetrievalRun


class AggregateQualityGateTests(unittest.TestCase):
    def test_passing_run_reports_machine_stable_pass(self):
        case = self._case("q-pass")
        qrel = self._qrel("q-pass", "chunk-pass")
        run = self._run("q-pass", "chunk-pass")

        result = aggregate_quality_gate(
            cases=[case],
            qrels=[qrel],
            runs=[run],
            findings=[],
            agreement=self._agreement(),
            thresholds=self._thresholds(),
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["metrics"]["ndcg_at_10"], 1.0)
        self.assertEqual(result["artifact_ids"], ["artifact-q-pass"])
        self.assertEqual(result["run_ids"], ["run-q-pass"])
        self.assertEqual(result["mechanical_blocking_counts"], {})
        self.assertEqual(result["judgment_agreement"]["weighted_cohen_kappa"], 0.9)
        self.assertTrue(result["metrics_available"])
        self.assertEqual(result["metrics"]["citation_validity"], 1.0)
        self.assertEqual(result["metrics"]["precision_at_5"], 0.2)
        self.assertNotIn("precision_at_5", result["thresholds"])
        self.assertNotIn("precision_at_5", result["metric_checks"])

    def test_retrieval_metric_below_threshold_fails_release(self):
        case = self._case("q-threshold")
        qrel = self._qrel("q-threshold", "relevant")
        run = self._run("q-threshold", "irrelevant")

        result = aggregate_quality_gate(
            [case], [qrel], [run], [], self._agreement(),
            self._thresholds(),
        )

        self.assertEqual(result["status"], "FAIL")
        self.assertIn("metric_below_threshold:recall_at_10", result["gate_failures"])

    def test_missing_retrieval_thresholds_are_rejected(self):
        case = self._case("q-no-thresholds")
        qrel = self._qrel("q-no-thresholds", "chunk-no-thresholds")
        run = self._run("q-no-thresholds", "chunk-no-thresholds")

        with self.assertRaisesRegex(ValueError, "missing retrieval thresholds"):
            aggregate_quality_gate(
                [case], [qrel], [run], [], self._agreement(), {}
            )

    def test_fixed_policy_thresholds_cannot_be_weakened(self):
        case = self._case("q-fixed-policy")
        qrel = self._qrel("q-fixed-policy", "chunk-fixed-policy")
        run = self._run("q-fixed-policy", "chunk-fixed-policy")
        weakened = {
            "recall_at_10": 0.8999999999995,
            "recall_at_50": 0.9499999999995,
            "mrr_at_10": 0.7999999999995,
            "ndcg_at_10": 0.7999999999995,
            "citation_validity": 0.9999999999995,
            "hard_negative_false_positive_rate": 0.0500000000005,
            "weighted_cohen_kappa": 0.7999999999995,
            "max_category_regression": 0.0500000000005,
        }

        for threshold, value in weakened.items():
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "fixed policy threshold"):
                    aggregate_quality_gate(
                        [case],
                        [qrel],
                        [run],
                        [],
                        self._agreement(),
                        self._thresholds(**{threshold: value}),
                    )

    def test_category_ndcg_regression_greater_than_five_percent_fails(self):
        case = self._case("q-category", category="propagation")
        qrel = self._qrel("q-category", "relevant")
        run = self._run("q-category", "irrelevant")

        result = aggregate_quality_gate(
            [case], [qrel], [run], [], self._agreement(), self._thresholds(),
            baseline_category_metrics={"propagation": 0.06},
        )

        self.assertEqual(result["category_deltas"], {"propagation": -0.06})
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("category_regression:propagation", result["gate_failures"])

    def test_citation_validity_below_one_fails_release(self):
        case = self._case("q-citation")
        qrel = self._qrel("q-citation", "chunk-citation")
        run = self._run("q-citation", "chunk-citation")
        finding = {
            "code": "citation_mismatch",
            "severity": "blocking",
            "query_id": "q-citation",
            "chunk_id": "chunk-citation",
            "detail": "Citation differs from the canonical document.",
        }

        result = aggregate_quality_gate(
            [case], [qrel], [run], [finding], self._agreement(), self._thresholds(),
        )

        self.assertEqual(result["metrics"]["citation_validity"], 0.0)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("citation_validity_below_threshold", result["gate_failures"])

    def test_mechanical_severity_cannot_downgrade_a_blocking_code(self):
        case = self._case("q-severity")
        qrel = self._qrel("q-severity", "chunk-severity")
        run = self._run("q-severity", "chunk-severity")
        forged = {
            "code": "missing_required_facet",
            "severity": "warning",
            "query_id": "q-severity",
            "chunk_id": None,
            "detail": "The required facet is absent.",
        }

        with self.assertRaisesRegex(ValueError, "canonical severity"):
            aggregate_quality_gate(
                [case],
                [qrel],
                [run],
                [forged],
                self._agreement(),
                self._thresholds(),
            )

    def test_low_kappa_requires_calibration_and_suppresses_retrieval_claims(self):
        case = self._case("q-kappa")
        qrel = self._qrel("q-kappa", "chunk-kappa")
        run = self._run("q-kappa", "chunk-kappa")

        result = aggregate_quality_gate(
            [case], [qrel], [run], [],
            self._agreement(kappa=0.79, eligible=False), self._thresholds(),
        )

        self.assertEqual(result["status"], "NEEDS_CALIBRATION")
        self.assertFalse(result["metrics_available"])
        for metric in ("recall_at_10", "mrr_at_10", "ndcg_at_10"):
            self.assertIsNone(result["metrics"][metric])
        self.assertIn("judgment_agreement_not_release_eligible", result["gate_failures"])

    def test_missing_answerable_qrels_requires_calibration(self):
        case = self._case("q-no-qrels")
        run = self._run("q-no-qrels", "chunk-no-qrels")

        result = aggregate_quality_gate(
            [case], [], [run], [], self._agreement(), self._thresholds(),
        )

        self.assertEqual(result["status"], "NEEDS_CALIBRATION")
        self.assertFalse(result["metrics_available"])
        self.assertEqual(result["uncalibrated_query_ids"], ["q-no-qrels"])

    def test_hard_negative_false_positive_fails_release(self):
        case = self._case("q-negative", hard_negative=True)
        run = self._run("q-negative", "unsupported", score=0.8, threshold=0.8)
        finding = {
            "code": "hard_negative_false_positive",
            "severity": "blocking",
            "query_id": "q-negative",
            "chunk_id": "unsupported",
            "detail": "Evidence met the hard-negative confidence threshold.",
        }

        result = aggregate_quality_gate(
            [case], [], [run], [finding], self._agreement(), self._thresholds(),
        )

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["hard_negative_false_positives"], 1)
        self.assertEqual(result["hard_negative_false_positive_rate"], 1.0)
        self.assertIn("hard_negative_false_positive", result["gate_failures"])

    def test_hard_negative_rate_at_five_percent_passes(self):
        answerable = self._case("q-answerable")
        cases = [answerable]
        qrels = [self._qrel("q-answerable", "chunk-answerable")]
        runs = [self._run("q-answerable", "chunk-answerable")]
        for index in range(20):
            query_id = f"q-negative-{index:02d}"
            cases.append(self._case(query_id, hard_negative=True))
            runs.append(
                self._run(
                    query_id,
                    f"unsupported-{index:02d}",
                    score=0.5 if index == 0 else 0.49,
                    threshold=0.5,
                )
            )

        finding = {
            "code": "hard_negative_false_positive",
            "severity": "blocking",
            "query_id": "q-negative-00",
            "chunk_id": "unsupported-00",
            "detail": "Evidence met the hard-negative confidence threshold.",
        }
        result = aggregate_quality_gate(
            cases,
            qrels,
            runs,
            [finding],
            self._agreement(),
            self._thresholds(),
        )

        self.assertEqual(result["hard_negative_false_positive_rate"], 0.05)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["mechanical_blocking_counts"]["hard_negative_false_positive"],
            1,
        )
        self.assertNotIn("hard_negative_false_positive", result["gate_failures"])

    def test_hard_negative_rate_above_five_percent_fails_with_findings(self):
        answerable = self._case("q-answerable")
        cases = [answerable]
        qrels = [self._qrel("q-answerable", "chunk-answerable")]
        runs = [self._run("q-answerable", "chunk-answerable")]
        findings = []
        for index in range(20):
            query_id = f"q-negative-{index:02d}"
            chunk_id = f"unsupported-{index:02d}"
            cases.append(self._case(query_id, hard_negative=True))
            runs.append(
                self._run(
                    query_id,
                    chunk_id,
                    score=0.5 if index < 2 else 0.49,
                    threshold=0.5,
                )
            )
            if index < 2:
                findings.append(
                    {
                        "code": "hard_negative_false_positive",
                        "severity": "blocking",
                        "query_id": query_id,
                        "chunk_id": chunk_id,
                        "detail": "Evidence met the hard-negative confidence threshold.",
                    }
                )

        result = aggregate_quality_gate(
            cases,
            qrels,
            runs,
            findings,
            self._agreement(),
            self._thresholds(),
        )

        self.assertEqual(result["hard_negative_false_positive_rate"], 0.10)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["mechanical_blocking_counts"]["hard_negative_false_positive"],
            2,
        )
        self.assertIn("hard_negative_false_positive", result["gate_failures"])

    def test_paired_bootstrap_is_fixed_and_requires_positive_lower_bound(self):
        confirmed = gates.paired_bootstrap_ndcg([0.1, 0.2, 0.3], [0.3, 0.4, 0.5])
        repeated = gates.paired_bootstrap_ndcg([0.1, 0.2, 0.3], [0.3, 0.4, 0.5])
        tied = gates.paired_bootstrap_ndcg([0.2, 0.4], [0.2, 0.4])

        self.assertEqual(confirmed, repeated)
        self.assertEqual(confirmed["resamples"], 10_000)
        self.assertGreater(confirmed["lower_bound"], 0.0)
        self.assertEqual(confirmed["improvement"], "confirmed_improvement")
        self.assertEqual(tied["improvement"], "no_confirmed_improvement")

    def test_paired_comparison_reports_category_deltas_and_claim_boundary(self):
        baseline = [
            {"query_id": "q-a", "category": "propagation", "ndcg_at_10": 0.2},
            {"query_id": "q-b", "category": "propagation", "ndcg_at_10": 0.3},
        ]
        improved = [
            {"query_id": "q-a", "category": "propagation", "ndcg_at_10": 0.4},
            {"query_id": "q-b", "category": "propagation", "ndcg_at_10": 0.5},
        ]
        regressed = [
            {"query_id": "q-a", "category": "propagation", "ndcg_at_10": 0.14},
            {"query_id": "q-b", "category": "propagation", "ndcg_at_10": 0.24},
        ]

        comparison = gates.compare_quality_runs(baseline, improved)
        failed = gates.compare_quality_runs(baseline, regressed)

        self.assertEqual(comparison["status"], "PASS")
        self.assertEqual(comparison["paired_query_ids"], ["q-a", "q-b"])
        self.assertAlmostEqual(comparison["category_deltas"]["propagation"], 0.2)
        self.assertEqual(
            comparison["paired_ndcg"]["improvement"], "confirmed_improvement"
        )
        self.assertEqual(failed["status"], "FAIL")
        self.assertEqual(
            failed["paired_ndcg"]["improvement"], "no_confirmed_improvement"
        )
        self.assertIn("category_regression:propagation", failed["gate_failures"])

    def test_comparison_category_limit_cannot_be_weakened(self):
        baseline = [
            {"query_id": "q-policy", "category": "propagation", "ndcg_at_10": 0.9}
        ]
        candidate = [
            {"query_id": "q-policy", "category": "propagation", "ndcg_at_10": 0.0}
        ]

        with self.assertRaisesRegex(ValueError, "fixed policy threshold"):
            gates.compare_quality_runs(
                baseline, candidate, max_category_regression=0.0500000000005
            )

    def test_any_category_drop_beyond_five_percent_fails(self):
        baseline = [
            {
                "query_id": "q-strict",
                "category": "propagation",
                "ndcg_at_10": 0.5500000000005,
            }
        ]
        candidate = [
            {"query_id": "q-strict", "category": "propagation", "ndcg_at_10": 0.50}
        ]

        comparison = gates.compare_quality_runs(baseline, candidate)

        self.assertEqual(comparison["status"], "FAIL")
        self.assertIn("category_regression:propagation", comparison["gate_failures"])

    def test_multiple_exact_five_percent_drops_stay_on_boundary(self):
        baseline = [
            {
                "query_id": f"q-boundary-{index}",
                "category": "propagation",
                "ndcg_at_10": value,
            }
            for index, value in enumerate((0.55, 0.65, 0.75))
        ]
        candidate = [
            {
                "query_id": f"q-boundary-{index}",
                "category": "propagation",
                "ndcg_at_10": value,
            }
            for index, value in enumerate((0.50, 0.60, 0.70))
        ]

        comparison = gates.compare_quality_runs(baseline, candidate)

        self.assertEqual(comparison["status"], "PASS")
        self.assertEqual(comparison["category_deltas"]["propagation"], -0.05)

    def test_exact_five_percent_category_drop_is_not_greater_than_limit(self):
        baseline = [
            {"query_id": "q-boundary", "category": "propagation", "ndcg_at_10": 0.55}
        ]
        candidate = [
            {"query_id": "q-boundary", "category": "propagation", "ndcg_at_10": 0.50}
        ]

        comparison = gates.compare_quality_runs(baseline, candidate)

        self.assertEqual(comparison["status"], "PASS")
        self.assertEqual(comparison["gate_failures"], [])

    @staticmethod
    def _case(query_id, *, category="propagation", hard_negative=False, split=None):
        return EvalCase.from_dict(
            {
                "query_id": query_id,
                "query": "Explain the validated propagation condition.",
                "category": category,
                "expected_facets": [] if hard_negative else ["condition"],
                "is_hard_negative": hard_negative,
                "requires_multiple_evidence": False,
                "split": split or ("adversarial" if hard_negative else "development"),
            }
        )

    @staticmethod
    def _qrel(query_id, chunk_id, relevance=3):
        return Qrel.from_dict(
            {
                "query_id": query_id,
                "chunk_id": chunk_id,
                "relevance": relevance,
                "supported_facets": ["condition"] if relevance else [],
                "confidence": 1.0,
                "judgment_source": "human_calibration",
            }
        )

    @staticmethod
    def _run(query_id, chunk_id, *, score=1.0, threshold=0.5):
        return RetrievalRun.from_dict(
            {
                "run_id": f"run-{query_id}",
                "query_id": query_id,
                "artifact_id": f"artifact-{query_id}",
                "results": [
                    {
                        "rank": 1,
                        "chunk_id": chunk_id,
                        "score": score,
                        "citation": "source.tex:10",
                        "text": "The stated condition is validated.",
                        "source_id": "source-1",
                    }
                ],
                "degraded": False,
                "confidence_threshold": threshold,
            }
        )

    @staticmethod
    def _agreement(*, kappa=0.9, eligible=True):
        return {
            "total": 4,
            "exact_agreement": 1.0,
            "weighted_cohen_kappa": kappa,
            "release_eligible": eligible,
            "status": "release_eligible" if eligible else "not_release_eligible",
            "adjudication_ids": [],
            "unresolved_ids": [],
        }

    @staticmethod
    def _thresholds(**overrides):
        values = {
            "recall_at_10": 0.90,
            "recall_at_50": 0.95,
            "mrr_at_10": 0.80,
            "ndcg_at_10": 0.80,
        }
        values.update(overrides)
        return values


class QualityGateCliTests(unittest.TestCase):
    def test_cli_exposes_the_five_required_commands(self):
        import em_rag.evaluation.cli as evaluation_cli

        self.assertEqual(
            evaluation_cli.command_names(),
            (
                "validate-dataset",
                "build-pool",
                "validate-judgments",
                "evaluate-run",
                "compare-runs",
            ),
        )

    def test_build_pool_accepts_active_dataset_case_fields(self):
        import em_rag.evaluation.cli as evaluation_cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            runs_path = root / "runs.jsonl"
            output_path = root / "pool.jsonl"
            write_jsonl(
                cases_path,
                [
                    {
                        "query_id": "q-active",
                        "query": "Explain the validated propagation condition.",
                        "category": "condition_limitation",
                        "expected_evidence_facets": ["condition"],
                        "expected_intent": "explanation",
                        "expected_sources": ["source-a"],
                        "is_hard_negative": False,
                        "language": "en",
                        "notes": "active dataset metadata",
                        "provenance": "curated_candidate_v2",
                        "requires_multi_citation": True,
                        "requires_web_check": False,
                        "split": "development",
                    }
                ],
            )
            write_jsonl(
                runs_path,
                [asdict(AggregateQualityGateTests._run("q-active", "chunk-active"))],
            )

            return_code = evaluation_cli.main(
                [
                    "build-pool",
                    "--cases",
                    str(cases_path),
                    "--runs",
                    str(runs_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "7",
                ]
            )

            self.assertEqual(return_code, 0)
            rows = read_jsonl(output_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["expected_facets"], ["condition"])

    def test_build_pool_rejects_active_dataset_without_language(self):
        import em_rag.evaluation.cli as evaluation_cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            runs_path = root / "runs.jsonl"
            output_path = root / "pool.jsonl"
            write_jsonl(
                cases_path,
                [
                    {
                        "query_id": "q-active",
                        "query": "Explain the validated propagation condition.",
                        "category": "condition_limitation",
                        "expected_evidence_facets": ["condition"],
                        "is_hard_negative": False,
                        "requires_multi_citation": False,
                        "split": "development",
                    }
                ],
            )
            write_jsonl(
                runs_path,
                [asdict(AggregateQualityGateTests._run("q-active", "chunk-active"))],
            )

            return_code = evaluation_cli.main(
                [
                    "build-pool",
                    "--cases",
                    str(cases_path),
                    "--runs",
                    str(runs_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "7",
                ]
            )

            self.assertEqual(return_code, 2)
            self.assertFalse(output_path.exists())

    def test_build_pool_rejects_unknown_active_dataset_field(self):
        import em_rag.evaluation.cli as evaluation_cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            runs_path = root / "runs.jsonl"
            output_path = root / "pool.jsonl"
            write_jsonl(
                cases_path,
                [
                    {
                        "query_id": "q-active",
                        "query": "Explain the validated propagation condition.",
                        "language": "en",
                        "category": "condition_limitation",
                        "expected_intent": "explanation",
                        "expected_sources": ["source-a"],
                        "expected_evidence_facets": ["condition"],
                        "is_hard_negative": False,
                        "requires_web_check": False,
                        "requires_multi_citation": False,
                        "notes": "active dataset metadata",
                        "provenance": "curated_candidate_v2",
                        "split": "development",
                        "unexpected_metdata": "typo",
                    }
                ],
            )
            write_jsonl(
                runs_path,
                [asdict(AggregateQualityGateTests._run("q-active", "chunk-active"))],
            )

            return_code = evaluation_cli.main(
                [
                    "build-pool",
                    "--cases",
                    str(cases_path),
                    "--runs",
                    str(runs_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "7",
                ]
            )

            self.assertEqual(return_code, 2)
            self.assertFalse(output_path.exists())

    def test_evaluate_run_writes_utf8_lf_outputs_and_hides_holdout_details(self):
        import em_rag.evaluation.cli as evaluation_cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._write_evaluation_inputs(root)
            output = root / "quality"

            return_code = evaluation_cli.main(
                self._evaluate_args(inputs, output)
            )

            self.assertEqual(return_code, 0)
            for name in ("metrics.json", "quality_gate.json", "quality_report.md"):
                payload = (output / name).read_bytes()
                self.assertTrue(payload)
                self.assertNotIn(b"\r\n", payload)
                payload.decode("utf-8")
            gate = json.loads((output / "quality_gate.json").read_text(encoding="utf-8"))
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["status"], "PASS")
            self.assertNotIn("per_query", gate)
            self.assertEqual(
                [record["query_id"] for record in metrics["per_query"]],
                ["q-development"],
            )
            self.assertNotIn("holdout_details", metrics)
            self.assertIn("category_breakdown", metrics)

    def test_evaluate_run_accepts_one_run_id_across_multiple_queries(self):
        import em_rag.evaluation.cli as evaluation_cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._write_evaluation_inputs(root)
            run_rows = read_jsonl(inputs["runs"])
            for row in run_rows:
                row["run_id"] = "shared-experiment"
            inputs["runs"].write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in run_rows
                ),
                encoding="utf-8",
                newline="\n",
            )
            output = root / "shared-run"

            return_code = evaluation_cli.main(self._evaluate_args(inputs, output))

            self.assertEqual(return_code, 0)
            gate = json.loads((output / "quality_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["run_ids"], ["shared-experiment"])

    def test_holdout_detail_requires_include_and_rotation_flags_together(self):
        import em_rag.evaluation.cli as evaluation_cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._write_evaluation_inputs(root)
            denied = root / "denied"

            denied_code = evaluation_cli.main(
                self._evaluate_args(inputs, denied) + ["--include-holdout-details"]
            )

            self.assertEqual(denied_code, 2)
            self.assertFalse((denied / "metrics.json").exists())

            rotation_only = root / "rotation-only"
            rotation_only_code = evaluation_cli.main(
                self._evaluate_args(inputs, rotation_only) + ["--rotation-flag"]
            )
            self.assertEqual(rotation_only_code, 0)
            rotation_metrics = json.loads(
                (rotation_only / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("holdout_details", rotation_metrics)

            allowed = root / "allowed"
            allowed_code = evaluation_cli.main(
                self._evaluate_args(inputs, allowed)
                + ["--include-holdout-details", "--rotation-flag"]
            )
            self.assertEqual(allowed_code, 0)
            allowed_metrics = json.loads(
                (allowed / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [record["query_id"] for record in allowed_metrics["holdout_details"]],
                ["q-holdout"],
            )

    def test_default_outputs_redact_holdout_specific_identifiers(self):
        import em_rag.evaluation.cli as evaluation_cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._write_evaluation_inputs(root)
            qrels = [
                row for row in read_jsonl(inputs["qrels"])
                if row["query_id"] != "q-holdout"
            ]
            write_jsonl(inputs["qrels"], qrels)
            output = root / "redacted"

            return_code = evaluation_cli.main(self._evaluate_args(inputs, output))

            self.assertEqual(return_code, 1)
            gate_text = (output / "quality_gate.json").read_text(encoding="utf-8")
            report_text = (output / "quality_report.md").read_text(encoding="utf-8")
            for private_value in (
                "q-holdout",
                "artifact-q-holdout",
                "run-q-holdout",
            ):
                self.assertNotIn(private_value, gate_text)
                self.assertNotIn(private_value, report_text)

    def test_failed_release_gate_returns_one_after_writing_reports(self):
        import em_rag.evaluation.cli as evaluation_cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._write_evaluation_inputs(root)
            qrels = read_jsonl(inputs["qrels"])
            qrels[0]["chunk_id"] = "unretrieved-development-chunk"
            write_jsonl(inputs["qrels"], qrels)
            output = root / "failed"

            return_code = evaluation_cli.main(self._evaluate_args(inputs, output))

            self.assertEqual(return_code, 1)
            gate = json.loads((output / "quality_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["status"], "FAIL")

    def test_auxiliary_commands_run_on_valid_inputs(self):
        import em_rag.evaluation.cli as evaluation_cli

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._write_evaluation_inputs(root)

            dataset_output = root / "dataset.json"
            self.assertEqual(
                evaluation_cli.main(
                    [
                        "validate-dataset",
                        "--cases",
                        str(inputs["cases"]),
                        "--qrels",
                        str(inputs["qrels"]),
                        "--output",
                        str(dataset_output),
                    ]
                ),
                0,
            )
            self.assertEqual(
                json.loads(dataset_output.read_text(encoding="utf-8"))["status"],
                "PASS",
            )

            pool_output = root / "pool.jsonl"
            self.assertEqual(
                evaluation_cli.main(
                    [
                        "build-pool",
                        "--cases",
                        str(inputs["cases"]),
                        "--runs",
                        str(inputs["runs"]),
                        "--output",
                        str(pool_output),
                        "--seed",
                        "7",
                    ]
                ),
                0,
            )
            self.assertEqual(len(read_jsonl(pool_output)), 2)

            judgment = {
                "judgment_id": "j-a",
                "relevance": 2,
                "supported_facets": ["condition"],
                "scope_correct": True,
                "citation_supported": True,
                "pollution": False,
                "confidence": 0.9,
                "source_quote": "The source states the condition.",
                "reason": "The quoted span directly supports the grade.",
            }
            pass1 = root / "pass1.jsonl"
            pass2 = root / "pass2.jsonl"
            write_jsonl(pass1, [judgment])
            write_jsonl(pass2, [judgment])
            judgments_output = root / "judgments.json"
            self.assertEqual(
                evaluation_cli.main(
                    [
                        "validate-judgments",
                        "--pass1",
                        str(pass1),
                        "--pass2",
                        str(pass2),
                        "--output",
                        str(judgments_output),
                    ]
                ),
                0,
            )
            self.assertTrue(
                json.loads(judgments_output.read_text(encoding="utf-8"))[
                    "agreement"
                ]["release_eligible"]
            )

            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            comparison_output = root / "comparison.json"
            baseline.write_text(
                json.dumps(
                    {
                        "per_query": [
                            {
                                "query_id": "q-a",
                                "category": "propagation",
                                "ndcg_at_10": 0.2,
                            },
                            {
                                "query_id": "q-b",
                                "category": "propagation",
                                "ndcg_at_10": 0.3,
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            candidate.write_text(
                json.dumps(
                    {
                        "per_query": [
                            {
                                "query_id": "q-a",
                                "category": "propagation",
                                "ndcg_at_10": 0.4,
                            },
                            {
                                "query_id": "q-b",
                                "category": "propagation",
                                "ndcg_at_10": 0.5,
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            self.assertEqual(
                evaluation_cli.main(
                    [
                        "compare-runs",
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--output",
                        str(comparison_output),
                    ]
                ),
                0,
            )
            comparison = json.loads(comparison_output.read_text(encoding="utf-8"))
            self.assertEqual(comparison["paired_ndcg"]["resamples"], 10_000)

    @staticmethod
    def _evaluate_args(inputs, output):
        return [
            "evaluate-run",
            "--cases",
            str(inputs["cases"]),
            "--qrels",
            str(inputs["qrels"]),
            "--runs",
            str(inputs["runs"]),
            "--findings",
            str(inputs["findings"]),
            "--agreement",
            str(inputs["agreement"]),
            "--thresholds",
            str(inputs["thresholds"]),
            "--output-dir",
            str(output),
        ]

    @staticmethod
    def _write_evaluation_inputs(root):
        cases = [
            AggregateQualityGateTests._case("q-development"),
            AggregateQualityGateTests._case("q-holdout", split="holdout"),
        ]
        qrels = [
            AggregateQualityGateTests._qrel("q-development", "chunk-development"),
            AggregateQualityGateTests._qrel("q-holdout", "chunk-holdout"),
        ]
        runs = [
            AggregateQualityGateTests._run(
                "q-development", "chunk-development"
            ),
            AggregateQualityGateTests._run("q-holdout", "chunk-holdout"),
        ]
        paths = {
            "cases": root / "cases.jsonl",
            "qrels": root / "qrels.jsonl",
            "runs": root / "runs.jsonl",
            "findings": root / "findings.jsonl",
            "agreement": root / "agreement.json",
            "thresholds": root / "thresholds.json",
        }
        write_jsonl(paths["cases"], [asdict(case) for case in cases])
        write_jsonl(paths["qrels"], [asdict(qrel) for qrel in qrels])
        write_jsonl(paths["runs"], [asdict(run) for run in runs])
        write_jsonl(paths["findings"], [])
        paths["agreement"].write_text(
            json.dumps(AggregateQualityGateTests._agreement(), sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        paths["thresholds"].write_text(
            json.dumps(AggregateQualityGateTests._thresholds(), sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return paths


if __name__ == "__main__":
    unittest.main()
