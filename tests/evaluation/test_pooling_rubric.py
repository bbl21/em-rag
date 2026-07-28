import unittest
from pathlib import Path


class RubricTests(unittest.TestCase):
    def test_rubric_defines_all_grades_and_blind_source_grounding_rules(self):
        rubric = Path(
            "kb_corpus_build/eval/retrieval_quality_v2/judgment_rubric.md"
        ).read_text(encoding="utf-8")

        expected_definitions = {
            3: "Directly relevant and sufficient",
            2: "Relevant but incomplete",
            1: "Marginally relevant",
            0: "Not relevant",
        }
        for grade, definition in expected_definitions.items():
            self.assertIn(f"### Grade {grade}", rubric)
            self.assertIn(definition, rubric)
        self.assertIn("Do not use outside knowledge", rubric)
        self.assertIn("full text", rubric)
        self.assertIn("Do not infer retriever identity", rubric)
        self.assertIn("source quote", rubric)
        self.assertIn("Never average conflicting grades", rubric)
        self.assertIn("kappa below 0.8", rubric)
        self.assertIn("unresolved rows or a declared sample", rubric)


if __name__ == "__main__":
    unittest.main()
