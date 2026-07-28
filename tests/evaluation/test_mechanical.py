import unittest

from em_rag.evaluation.mechanical import inspect_run
from em_rag.evaluation.models import EvalCase, RankedEvidence, RetrievalRun


class MechanicalInspectionTests(unittest.TestCase):
    def test_clean_run_has_no_findings(self):
        case = self._case(expected_facets=("frequency range",))
        run = self._run(
            self._evidence(
                "chunk-1",
                "source.tex:10",
                "The frequency range is 2 GHz to 6 GHz under the stated condition.",
            )
        )
        docstore = {
            "chunk-1": self._document(
                "source.tex:10",
                "The frequency range is 2 GHz to 6 GHz under the stated condition.",
                group_id="section-1",
            )
        }

        self.assertEqual(inspect_run(case, run, docstore), ())

    def test_missing_chunk_is_blocking(self):
        findings = inspect_run(
            self._case(),
            self._run(self._evidence("missing", "source.tex:1", "Evidence text.")),
            {},
        )

        self.assertFinding(findings, "missing_chunk", "blocking", "missing")

    def test_citation_mismatch_is_blocking(self):
        findings = inspect_run(
            self._case(),
            self._run(self._evidence("chunk-1", "source.tex:99", "Evidence text.")),
            {"chunk-1": self._document("source.tex:10", "Evidence text.")},
        )

        self.assertFinding(findings, "citation_mismatch", "blocking", "chunk-1")

    def test_duplicate_chunk_or_group_is_a_warning(self):
        cases = (
            (
                self._run(
                    self._evidence("chunk-1", "source.tex:1", "First evidence.", rank=1),
                    self._evidence("chunk-1", "source.tex:2", "Second evidence.", rank=2),
                    allow_duplicate_chunks=True,
                ),
                {
                    "chunk-1": self._document(
                        "source.tex:1", "First evidence.", group_id="group-1"
                    )
                },
            ),
            (
                self._run(
                    self._evidence("chunk-1", "source.tex:1", "First evidence.", rank=1),
                    self._evidence("chunk-2", "source.tex:2", "Second evidence.", rank=2),
                ),
                {
                    "chunk-1": self._document(
                        "source.tex:1", "First evidence.", group_id="group-1"
                    ),
                    "chunk-2": self._document(
                        "source.tex:2", "Second evidence.", group_id="group-1"
                    ),
                },
            ),
        )
        for run, docstore in cases:
            with self.subTest(run=run):
                findings = inspect_run(self._case(), run, docstore)
                self.assertFinding(findings, "duplicate_evidence", "warning")

    def test_exact_same_citation_repeated_is_a_warning(self):
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "First evidence.", rank=1),
            self._evidence("chunk-2", "source.tex:1", "Second evidence.", rank=2),
        )
        docstore = {
            "chunk-1": self._document("source.tex:1", "First evidence."),
            "chunk-2": self._document("source.tex:1", "Second evidence."),
        }

        findings = inspect_run(self._case(), run, docstore)

        self.assertFinding(findings, "duplicate_evidence", "warning")

    def test_missing_expected_facet_is_blocking(self):
        case = self._case(expected_facets=("line of sight", "frequency range"))
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "This model assumes line-of-sight propagation.")
        )
        docstore = {
            "chunk-1": self._document(
                "source.tex:1", "This model assumes line-of-sight propagation."
            )
        }

        findings = inspect_run(case, run, docstore)

        missing = [finding for finding in findings if finding.code == "missing_required_facet"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].severity, "blocking")
        self.assertIn("frequency range", missing[0].detail)

    def test_expected_facet_requires_normalized_token_sequence_boundary(self):
        case = self._case(expected_facets=("range",))
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "The antenna arrangement is fixed.")
        )
        docstore = {
            "chunk-1": self._document(
                "source.tex:1", "The antenna arrangement is fixed."
            )
        }

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "missing_required_facet", "blocking")

    def test_formula_without_variable_or_condition_context_is_a_warning(self):
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "L = 20 log10(d).")
        )
        docstore = {"chunk-1": self._document("source.tex:1", "L = 20 log10(d).")}

        findings = inspect_run(self._case(), run, docstore)

        self.assertFinding(findings, "formula_context_gap", "warning", "chunk-1")

    def test_distant_condition_word_does_not_supply_formula_context(self):
        text = (
            "When the introductory discussion applies, "
            + ("background " * 20)
            + "L = 20 log10(d)."
        )
        run = self._run(self._evidence("chunk-1", "source.tex:1", text))
        docstore = {"chunk-1": self._document("source.tex:1", text)}

        findings = inspect_run(self._case(), run, docstore)

        self.assertFinding(findings, "formula_context_gap", "warning", "chunk-1")

    def test_nearby_variable_context_supports_formula(self):
        text = "L = 20 log10(d), where d denotes distance."
        run = self._run(self._evidence("chunk-1", "source.tex:1", text))
        docstore = {"chunk-1": self._document("source.tex:1", text)}

        findings = inspect_run(self._case(), run, docstore)

        self.assertFalse(
            any(finding.code == "formula_context_gap" for finding in findings)
        )

    def test_overlapping_formula_markers_emit_one_context_gap(self):
        text = "x = sqrt(y)."
        run = self._run(self._evidence("chunk-1", "source.tex:1", text))
        docstore = {"chunk-1": self._document("source.tex:1", text)}

        findings = inspect_run(self._case(), run, docstore)

        formula_findings = [
            finding for finding in findings if finding.code == "formula_context_gap"
        ]
        self.assertEqual(len(formula_findings), 1)

    def test_distinct_overlapping_formula_windows_keep_local_context(self):
        text = (
            "x = 1."
            + (" background" * 14)
            + " y = 2, where y denotes the output."
        )
        run = self._run(self._evidence("chunk-1", "source.tex:1", text))
        docstore = {"chunk-1": self._document("source.tex:1", text)}

        findings = inspect_run(self._case(), run, docstore)

        formula_findings = [
            finding for finding in findings if finding.code == "formula_context_gap"
        ]
        self.assertEqual(len(formula_findings), 1)

    def test_number_or_unit_from_query_absent_from_cited_text_is_a_warning(self):
        case = self._case(query="What condition applies at 28 GHz?")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "The condition applies at millimeter wavelengths.")
        )
        docstore = {
            "chunk-1": self._document(
                "source.tex:1", "The condition applies at millimeter wavelengths."
            )
        }

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "numeric_support_gap", "warning")

    def test_sentence_final_integer_without_support_is_a_warning(self):
        case = self._case(query="Use 5.")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "Use no numeric value.")
        )
        docstore = {
            "chunk-1": self._document("source.tex:1", "Use no numeric value.")
        }

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "numeric_support_gap", "warning")

    def test_percent_unit_requires_exact_support(self):
        case = self._case(query="Use 5%.")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "Use 5 meters.")
        )
        docstore = {
            "chunk-1": self._document("source.tex:1", "Use 5 meters.")
        }

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "numeric_support_gap", "warning")

    def test_sentence_boundary_blocks_label_chaining(self):
        case = self._case(query="Use 5. Cases differ.")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "The canonical value is 5.")
        )
        docstore = {
            "chunk-1": self._document(
                "source.tex:1", "The canonical value is 5."
            )
        }

        findings = inspect_run(case, run, docstore)

        self.assertFalse(any(finding.code == "numeric_support_gap" for finding in findings))

    def test_evidence_sentence_boundary_blocks_numeric_candidate(self):
        case = self._case(query="Use 5 cases.")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "Use 5. Cases differ.")
        )
        docstore = {
            "chunk-1": self._document("source.tex:1", "Use 5. Cases differ.")
        }

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "numeric_support_gap", "warning")

    def test_evidence_punctuation_blocks_compound_numeric_candidate(self):
        degree = chr(0xB0)
        case = self._case(query=f"Use 5{degree}C.")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", f"Use 5{degree}. C.")
        )
        docstore = {
            "chunk-1": self._document("source.tex:1", f"Use 5{degree}. C.")
        }

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "numeric_support_gap", "warning")

    def test_compound_symbol_and_alpha_unit_requires_exact_match(self):
        degree = chr(0xB0)
        case = self._case(query=f"Use 5{degree}C.")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", f"Use 5{degree}F.")
        )
        docstore = {
            "chunk-1": self._document("source.tex:1", f"Use 5{degree}F.")
        }

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "numeric_support_gap", "warning")

    def test_compound_symbol_and_alpha_unit_exact_match_is_supported(self):
        degree = chr(0xB0)
        case = self._case(query=f"Use 5{degree}C.")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", f"Use 5{degree}C.")
        )
        docstore = {
            "chunk-1": self._document("source.tex:1", f"Use 5{degree}C.")
        }

        findings = inspect_run(case, run, docstore)

        self.assertFalse(any(finding.code == "numeric_support_gap" for finding in findings))

    def test_decimal_number_and_unit_exact_match_is_supported(self):
        case = self._case(query="What occurs at 2.5 GHz?")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "The event occurs at 2.5 GHz.")
        )
        docstore = {
            "chunk-1": self._document(
                "source.tex:1", "The event occurs at 2.5 GHz."
            )
        }

        findings = inspect_run(case, run, docstore)

        self.assertFalse(any(finding.code == "numeric_support_gap" for finding in findings))

    def test_signed_number_requires_exact_sign_match(self):
        case = self._case(query="What is the level at -5 dB?")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "The level is +5 dB.")
        )
        docstore = {
            "chunk-1": self._document(
                "source.tex:1", "The level is +5 dB."
            )
        }

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "numeric_support_gap", "warning")

    def test_numeric_support_requires_number_boundary(self):
        case = self._case(query="What condition applies at 2 GHz?")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "The condition applies at 12 GHz.")
        )
        docstore = {
            "chunk-1": self._document(
                "source.tex:1", "The condition applies at 12 GHz."
            )
        }

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "numeric_support_gap", "warning")

    def test_numeric_support_requires_matching_adjacent_unit(self):
        case = self._case(query="What occurs after 5 seconds?")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "The event occurs after 5 meters.")
        )
        docstore = {
            "chunk-1": self._document(
                "source.tex:1", "The event occurs after 5 meters."
            )
        }

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "numeric_support_gap", "warning")

    def test_unknown_unit_in_bounded_measurement_context_is_matched(self):
        case = self._case(query="What occurs after 5 quuxes?")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "The event occurs after 5 quuxes.")
        )
        docstore = {
            "chunk-1": self._document(
                "source.tex:1", "The event occurs after 5 quuxes."
            )
        }

        findings = inspect_run(case, run, docstore)

        self.assertFalse(any(finding.code == "numeric_support_gap" for finding in findings))

    def test_long_unit_in_ordinary_phrasing_requires_exact_match(self):
        case = self._case(query="Compare 5 seconds.")
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "Compare 5 meters.")
        )
        docstore = {
            "chunk-1": self._document("source.tex:1", "Compare 5 meters.")
        }

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "numeric_support_gap", "warning")

    def test_hard_negative_high_confidence_evidence_is_blocking(self):
        case = self._case(hard_negative=True)
        run = self._run(
            self._evidence("chunk-1", "source.tex:1", "Unsupported answer.", score=0.8),
            threshold=0.8,
        )
        docstore = {"chunk-1": self._document("source.tex:1", "Unsupported answer.")}

        findings = inspect_run(case, run, docstore)

        self.assertFinding(findings, "hard_negative_false_positive", "blocking", "chunk-1")

    def assertFinding(self, findings, code, severity, chunk_id=None):
        matches = [finding for finding in findings if finding.code == code]
        self.assertTrue(matches, f"missing finding {code}: {findings}")
        self.assertTrue(all(finding.severity == severity for finding in matches))
        self.assertTrue(all(finding.query_id == "q-1" for finding in matches))
        if chunk_id is not None:
            self.assertTrue(any(finding.chunk_id == chunk_id for finding in matches))

    @staticmethod
    def _case(query="Explain the cited result.", expected_facets=(), hard_negative=False):
        return EvalCase(
            query_id="q-1",
            query=query,
            category="evaluation",
            expected_facets=tuple(expected_facets),
            is_hard_negative=hard_negative,
            requires_multiple_evidence=False,
            split="adversarial" if hard_negative else "development",
        )

    @staticmethod
    def _evidence(chunk_id, citation, text, rank=1, score=0.5):
        return RankedEvidence(
            rank=rank,
            chunk_id=chunk_id,
            score=score,
            citation=citation,
            text=text,
            source_id="source-1",
        )

    @staticmethod
    def _run(*results, threshold=0.9, allow_duplicate_chunks=False):
        if allow_duplicate_chunks:
            return RetrievalRun(
                run_id="run-1",
                query_id="q-1",
                artifact_id="artifact-1",
                results=tuple(results),
                degraded=False,
                confidence_threshold=threshold,
            )
        return RetrievalRun.from_dict(
            {
                "run_id": "run-1",
                "query_id": "q-1",
                "artifact_id": "artifact-1",
                "results": [
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk_id,
                        "score": item.score,
                        "citation": item.citation,
                        "text": item.text,
                        "source_id": item.source_id,
                    }
                    for item in results
                ],
                "degraded": False,
                "confidence_threshold": threshold,
            }
        )

    @staticmethod
    def _document(citation, text, group_id=None):
        document = {"citation": citation, "text": text, "source_id": "source-1"}
        if group_id is not None:
            document["group_id"] = group_id
        return document


if __name__ == "__main__":
    unittest.main()
