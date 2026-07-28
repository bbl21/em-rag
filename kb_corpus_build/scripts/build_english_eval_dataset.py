#!/usr/bin/env python3
"""Build the deterministic English-only retrieval evaluation dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from kb_corpus_build.scripts.unicode_han import contains_han
except ModuleNotFoundError:
    from unicode_han import contains_han


ACTIVE_DATASET = Path("kb_corpus_build/eval/datasets/retrieval_quality_eval_cases.jsonl")
MANIFEST_PATH = Path("kb_corpus_build/eval/retrieval_quality_v2/dataset_manifest.json")
CANONICAL_CORPUS = Path("kb_corpus_build/corpus/chunks.canonical.jsonl")
SELECTION_SEED = "retrieval-quality-v2-english-2026-07-13"
CATEGORY_ORDER = (
    "definition",
    "formula_variable",
    "numeric_range",
    "condition_limitation",
    "exact_symbol",
    "multiple_evidence",
    "cross_source",
    "hard_negative",
    "long_noisy",
)
SPLIT_ORDER = ("development", "regression", "holdout")
CANDIDATE_COUNTS = {
    "definition": 40,
    "formula_variable": 40,
    "numeric_range": 27,
    "condition_limitation": 27,
    "exact_symbol": 27,
    "multiple_evidence": 27,
    "cross_source": 20,
    "hard_negative": 20,
    "long_noisy": 12,
}
SPLIT_MATRIX = {
    "definition": {"development": 14, "regression": 8, "holdout": 8},
    "formula_variable": {"development": 14, "regression": 8, "holdout": 8},
    "numeric_range": {"development": 9, "regression": 6, "holdout": 5},
    "condition_limitation": {"development": 9, "regression": 5, "holdout": 6},
    "exact_symbol": {"development": 9, "regression": 6, "holdout": 5},
    "multiple_evidence": {"development": 9, "regression": 5, "holdout": 6},
    "cross_source": {"development": 7, "regression": 4, "holdout": 4},
    "hard_negative": {"development": 6, "regression": 4, "holdout": 5},
    "long_noisy": {"development": 3, "regression": 4, "holdout": 3},
}
ACCEPTED_COUNTS = {
    category: sum(split_counts.values())
    for category, split_counts in SPLIT_MATRIX.items()
}

ELLINGSON = "ellingson_em_vol1"
MIT = "mit_em_applications"
MODERN = "modern_antennas_microwave_circuits"
P1411 = "itu_r_p1411_13"

LEGACY_FACET_FIXES = {
    "q024": ["scattering parameters", "scattering"],
    "q030": ["microstrip", "patch antennas"],
    "q045": ["relevant parameters", "LoS"],
    "q046": ["valid for", "ITU-R P.1411"],
    "q051": ["urban", "micro-cellular"],
}
SOURCE_CHECK_FIXED_IDS = frozenset(
    {
        *LEGACY_FACET_FIXES,
        "en_formula_variable_017",
        "en_formula_variable_018",
        "en_numeric_range_010",
        "en_numeric_range_011",
        "en_numeric_range_012",
        "en_numeric_range_013",
        "en_numeric_range_014",
        "en_numeric_range_015",
        "en_numeric_range_016",
        "en_numeric_range_017",
        "en_condition_limitation_003",
        "en_condition_limitation_004",
        "en_exact_symbol_012",
        "en_exact_symbol_016",
        "en_multiple_evidence_019",
        "en_multiple_evidence_020",
        "en_long_noisy_009",
        "en_long_noisy_012",
    }
)
SOURCE_CHECK_REPLACED_IDS = frozenset({"en_definition_012", "en_definition_018"})
SOURCE_TEXT_FIELDS = ("content_md",)

LEGACY_CASES = json.loads(r"""[
  {
    "query_id": "q001",
    "query": "Maxwell equations",
    "category": "formula_variable",
    "expected_intent": "formula",
    "expected_sources": [
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "Maxwell",
      "equations"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q002",
    "query": "differential form of Maxwell equations",
    "category": "formula_variable",
    "expected_intent": "formula",
    "expected_sources": [
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "curl",
      "divergence",
      "Maxwell"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q003",
    "query": "wave equation in electromagnetics",
    "category": "formula_variable",
    "expected_intent": "formula",
    "expected_sources": [
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "wave equation"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q004",
    "query": "electric field definition",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "electric field"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q005",
    "query": "magnetic field definition",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "magnetic field"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q006",
    "query": "electromagnetic boundary conditions",
    "category": "condition_limitation",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "boundary conditions"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q007",
    "query": "Poynting vector",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "Poynting vector"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q008",
    "query": "displacement current",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "displacement current"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q009",
    "query": "uniform plane wave",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "uniform plane wave"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q010",
    "query": "power density of electromagnetic waves",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "power density"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q011",
    "query": "transmission line",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "transmission line"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q012",
    "query": "characteristic impedance",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "characteristic impedance"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q013",
    "query": "reflection coefficient",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "reflection coefficient"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q014",
    "query": "VSWR",
    "category": "exact_symbol",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "VSWR",
      "standing wave"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q015",
    "query": "S11",
    "category": "exact_symbol",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "S11",
      "reflection coefficient"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q016",
    "query": "S21",
    "category": "exact_symbol",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "S21",
      "transmission"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q017",
    "query": "impedance matching",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "impedance matching"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q018",
    "query": "Smith chart",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "Smith chart"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q019",
    "query": "epsilon_r",
    "category": "exact_symbol",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "epsilon_r",
      "permittivity"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q020",
    "query": "Z0",
    "category": "exact_symbol",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "Z0",
      "characteristic impedance"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q021",
    "query": "quarter-wave transformer",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "quarter-wave transformer"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q022",
    "query": "TEM transmission line",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "TEM",
      "transmission line"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q023",
    "query": "microstrip line",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "microstrip"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q024",
    "query": "scattering parameters",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "S-parameters",
      "scattering"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q025",
    "query": "antenna gain",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "antenna gain"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q026",
    "query": "directivity",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "directivity"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q027",
    "query": "radiation efficiency",
    "category": "cross_source",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "efficiency"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q028",
    "query": "radiation pattern",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "radiation pattern"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q029",
    "query": "antenna polarization",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "polarization"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q030",
    "query": "microstrip patch antenna",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "microstrip",
      "patch antenna"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q031",
    "query": "phased array",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "phased array"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q032",
    "query": "array factor",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "array factor"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q033",
    "query": "antenna bandwidth",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "bandwidth"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q034",
    "query": "effective aperture",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "effective aperture"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q035",
    "query": "short dipole antenna",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "short dipole"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q036",
    "query": "radar equation",
    "category": "formula_variable",
    "expected_intent": "formula",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "radar equation"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q037",
    "query": "beamforming",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "beamforming"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q038",
    "query": "link budget antenna parameters",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "link budget",
      "antenna"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q039",
    "query": "ITU-R P.1411 frequency range",
    "category": "numeric_range",
    "expected_intent": "frequency_range",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "ITU-R P.1411",
      "frequency range",
      "300 MHz"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q040",
    "query": "LoS path loss",
    "category": "multiple_evidence",
    "expected_intent": "definition",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "LoS",
      "path loss"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q041",
    "query": "NLoS path loss",
    "category": "multiple_evidence",
    "expected_intent": "definition",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "NLoS",
      "path loss"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q042",
    "query": "building entry loss",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "building entry loss"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q043",
    "query": "short-range outdoor propagation",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "short-range",
      "outdoor"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q044",
    "query": "propagation scenario in ITU-R P.1411",
    "category": "definition",
    "expected_intent": "scenario",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "scenario",
      "propagation"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q045",
    "query": "input parameters for ITU-R P.1411 LoS model",
    "category": "condition_limitation",
    "expected_intent": "input_parameters",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "input parameters",
      "LoS"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q046",
    "query": "limitations of ITU-R P.1411 models",
    "category": "condition_limitation",
    "expected_intent": "limitations",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "limitations",
      "ITU-R P.1411"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q047",
    "query": "street canyon NLoS model",
    "category": "condition_limitation",
    "expected_intent": "scenario",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "street canyon",
      "NLoS"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q048",
    "query": "above rooftop propagation",
    "category": "condition_limitation",
    "expected_intent": "scenario",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "rooftop",
      "propagation"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q049",
    "query": "below rooftop propagation",
    "category": "condition_limitation",
    "expected_intent": "scenario",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "below rooftop",
      "propagation"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q050",
    "query": "millimetre-wave propagation in ITU-R P.1411",
    "category": "condition_limitation",
    "expected_intent": "definition",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "millimetre-wave",
      "LoS",
      "NLoS"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q051",
    "query": "urban microcell propagation",
    "category": "condition_limitation",
    "expected_intent": "definition",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "urban",
      "microcell"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q052",
    "query": "basic transmission loss models in P.1411",
    "category": "definition",
    "expected_intent": "definition",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "basic transmission loss"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q061",
    "query": "How to use PyAEDT to create HFSS project?",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q062",
    "query": "CST VBA API example for patch antenna",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q063",
    "query": "OpenEMS Python script for antenna simulation",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q064",
    "query": "satellite orbital mechanics",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q065",
    "query": "quantum computing qubit decoherence",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q066",
    "query": "medical MRI safety limit",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q067",
    "query": "ANSYS license server troubleshooting",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "q068",
    "query": "finite element meshing Python API",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s001",
    "query": "reflection coefficient Gamma S11",
    "category": "exact_symbol",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1",
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "reflection coefficient",
      "Gamma",
      "S11"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s002",
    "query": "characteristic impedance Z0 Zo",
    "category": "exact_symbol",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1"
    ],
    "expected_evidence_facets": [
      "characteristic impedance",
      "Z0"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s003",
    "query": "S-parameters S11 S21 scattering matrix",
    "category": "exact_symbol",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "S11",
      "S21",
      "scattering"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": true,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s004",
    "query": "directivity antenna gain effective aperture",
    "category": "multiple_evidence",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "directivity",
      "antenna gain",
      "effective aperture"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": true,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s005",
    "query": "microstrip patch antenna printed antenna feed",
    "category": "multiple_evidence",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "ellingson_em_vol1"
    ],
    "expected_evidence_facets": [
      "microstrip",
      "patch antenna",
      "antenna"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s006",
    "query": "Maxwell equations curl divergence differential form",
    "category": "formula_variable",
    "expected_intent": "formula",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "ellingson_em_vol1",
      "mit_em_applications"
    ],
    "expected_evidence_facets": [
      "Maxwell",
      "curl",
      "divergence"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s007",
    "query": "LoS NLoS path loss street canyon ITU-R P.1411",
    "category": "multiple_evidence",
    "expected_intent": "scenario",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "LoS",
      "NLoS",
      "path loss",
      "ITU-R P.1411"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": true,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s008",
    "query": "ITU-R P.1411 300 MHz 300 GHz frequency range",
    "category": "numeric_range",
    "expected_intent": "frequency_range",
    "expected_sources": [
      "itu_r_p1411_13"
    ],
    "expected_evidence_facets": [
      "ITU-R P.1411",
      "300 MHz",
      "300 GHz"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s009",
    "query": "VSWR reflection coefficient standing wave ratio",
    "category": "exact_symbol",
    "expected_intent": "definition",
    "expected_sources": [
      "ellingson_em_vol1"
    ],
    "expected_evidence_facets": [
      "VSWR",
      "reflection coefficient",
      "standing wave ratio"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s010",
    "query": "S21 scattering parameter transmission coefficient",
    "category": "exact_symbol",
    "expected_intent": "definition",
    "expected_sources": [
      "modern_antennas_microwave_circuits"
    ],
    "expected_evidence_facets": [
      "S21",
      "scattering",
      "transmission"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s011",
    "query": "antenna gain efficiency directivity formula",
    "category": "formula_variable",
    "expected_intent": "formula",
    "expected_sources": [
      "modern_antennas_microwave_circuits",
      "ellingson_em_vol1"
    ],
    "expected_evidence_facets": [
      "antenna gain",
      "efficiency",
      "directivity"
    ],
    "is_hard_negative": false,
    "requires_web_check": false,
    "requires_multi_citation": true,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s012",
    "query": "orbital transfer Hohmann maneuver",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s013",
    "query": "solid mechanics finite element stress mesh",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s014",
    "query": "Python package API for mesh generation",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s015",
    "query": "MATLAB antenna toolbox license error",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  },
  {
    "query_id": "s016",
    "query": "CUDA memory optimization transformer inference",
    "category": "hard_negative",
    "expected_intent": "out_of_scope",
    "expected_sources": [],
    "expected_evidence_facets": [],
    "is_hard_negative": true,
    "requires_web_check": false,
    "requires_multi_citation": false,
    "notes": "Migrated from the valid English subset of the original 84-case dataset.",
    "language": "en",
    "provenance": "migrated_legacy_84"
  }
]""")


DEFINITION_SPECS = [
    ("What is electric flux density?", [ELLINGSON, MIT], ["electric flux density"]),
    ("What is magnetic flux density?", [ELLINGSON, MIT], ["magnetic flux density"]),
    ("What is electric potential?", [ELLINGSON, MIT], ["electric potential"]),
    ("What is permittivity?", [ELLINGSON, MIT], ["permittivity"]),
    ("What is permeability?", [ELLINGSON, MIT], ["permeability"]),
    ("What is electrical conductivity?", [ELLINGSON, MIT], ["conductivity"]),
    ("What is intrinsic impedance in an electromagnetic medium?", [ELLINGSON, MIT], ["intrinsic impedance"]),
    ("What is skin depth?", [ELLINGSON, MIT], ["skin depth"]),
    ("What is phase velocity?", [ELLINGSON, MIT], ["phase velocity"]),
    ("What is group velocity?", [MIT], ["group velocity"]),
    ("What is wavelength in a material medium?", [ELLINGSON, MIT], ["wavelength", "medium"]),
    ("What is conduction current density?", [ELLINGSON, MIT], ["conduction current density"]),
    ("What is surface current density?", [ELLINGSON, MIT], ["surface current density"]),
    ("What is line charge density?", [ELLINGSON, MIT], ["line charge density"]),
    ("What is the magnetic vector potential?", [ELLINGSON, MIT], ["magnetic vector potential"]),
    ("What is the effective length of an antenna?", [MODERN, MIT], ["effective length", "antenna"]),
    ("What is a half-wave dipole antenna?", [MODERN, MIT], ["half-wave dipole"]),
    ("What is an electrically short dipole?", [MIT], ["short dipole"]),
    ("What is an aperture antenna?", [MODERN, MIT], ["aperture antenna"]),
    ("What is a horn antenna?", [MODERN, MIT], ["horn antenna"]),
    ("What is a loop antenna?", [MODERN, MIT], ["loop antenna"]),
    ("What is a microwave network?", [MODERN], ["microwave network"]),
    ("What is line-of-sight propagation?", [P1411], ["line-of-sight", "propagation"]),
]

FORMULA_TOPICS = [
    ("Gauss's law for electric fields", [ELLINGSON, MIT], ["divergence", "electric flux density", "charge density"]),
    ("Gauss's law for magnetic fields", [ELLINGSON, MIT], ["divergence", "magnetic flux density", "zero"]),
    ("Faraday's law", [ELLINGSON, MIT], ["curl", "electric field", "magnetic flux density"]),
    ("the Ampere-Maxwell law", [ELLINGSON, MIT], ["curl", "magnetic field", "current density", "displacement current"]),
    ("the charge continuity equation", [ELLINGSON, MIT], ["divergence", "current density", "charge density"]),
    ("the Poynting vector", [ELLINGSON, MIT], ["electric field", "magnetic field", "power density"]),
    ("plane-wave intrinsic impedance", [ELLINGSON, MIT], ["electric field", "magnetic field", "intrinsic impedance"]),
    ("the wavelength-frequency relation", [ELLINGSON, MIT], ["wavelength", "frequency", "phase velocity"]),
    ("the propagation constant", [ELLINGSON, MIT], ["attenuation constant", "phase propagation constant", "propagation constant"]),
    ("the voltage reflection coefficient", [ELLINGSON, MODERN, MIT], ["reflection coefficient", "load impedance", "characteristic impedance"]),
    ("the VSWR and reflection-coefficient relationship", [ELLINGSON, MODERN, MIT], ["VSWR", "reflection coefficient"]),
    ("the input impedance of a terminated transmission line", [ELLINGSON, MODERN, MIT], ["input impedance", "characteristic impedance", "electrical length"]),
    ("quarter-wave transformer impedance", [MODERN, MIT], ["quarter-wave transformer", "impedance"]),
    ("the Friis transmission equation", [MODERN, MIT], ["received power", "antenna gain", "wavelength", "distance"]),
    ("the antenna gain-directivity-efficiency relation", [MODERN, MIT], ["antenna gain", "directivity", "efficiency"]),
    ("the effective-aperture and antenna-gain relation", [MODERN, MIT], ["effective aperture", "antenna gain", "wavelength"]),
    ("the LoS basic transmission loss model", [P1411], ["basic transmission loss", "distance", "frequency"]),
]

NUMERIC_RANGE_TOPICS = [
    ("the overall scope of Recommendation ITU-R P.1411", [P1411], ["ITU-R P.1411", "300 MHz", "300 GHz"], 9),
    ("the P.1411 NLoS2 dense-urban microcell submodel", [P1411], ["NLoS2", "800", "2 000 MHz"], 8),
    ("the P.1411 street-canyon frequency-dependent submodel", [P1411], ["street canyon", "2 GHz", "38 GHz"], 8),
]
NUMERIC_RANGE_TEMPLATES = [
    "What frequency range is specified for {label}?",
    "Give the lower and upper frequency limits for {label}.",
    "Which frequency endpoints are stated for {label}?",
    "Report the applicable numeric frequency interval for {label}.",
    "What are the documented frequency bounds for {label}?",
    "From what minimum to what maximum frequency does {label} apply?",
    "State the start and end frequencies for {label}.",
    "Identify the full frequency span assigned to {label}.",
    "What exact lower and upper frequencies delimit {label}?",
]

CONDITION_TOPICS = [
    ("electromagnetic boundary conditions at a material interface", [ELLINGSON, MIT], ["boundary conditions", "interface"]),
    ("the perfect-electric-conductor boundary condition", [ELLINGSON, MIT], ["perfect conductor", "tangential component"]),
    ("TEM propagation on a transmission line", [ELLINGSON, MODERN, MIT], ["TEM", "transverse"]),
    ("a quarter-wave transformer impedance match", [MODERN, MIT], ["quarter-wave transformer", "matched"]),
    ("interpreting S11 as an input reflection coefficient", [MODERN], ["S11", "matched"]),
    ("interpreting S21 as a forward transmission coefficient", [MODERN], ["S21", "matched"]),
    ("using the Friis transmission equation", [MODERN, MIT], ["far field", "line of sight"]),
    ("the scope of Recommendation ITU-R P.1411", [P1411], ["outdoor", "short-range"]),
    ("the P.1411 LoS basic transmission loss model", [P1411], ["LoS", "distance", "frequency"]),
    ("the P.1411 NLoS street-canyon model", [P1411], ["NLoS", "street canyon"]),
]

EXACT_SYMBOL_SPECS = [
    ("E", "electromagnetics", "electric field", [ELLINGSON, MIT]),
    ("D", "electromagnetics", "electric flux density", [ELLINGSON, MIT]),
    ("H", "electromagnetics", "magnetic field intensity", [ELLINGSON, MIT]),
    ("B", "electromagnetics", "magnetic flux density", [ELLINGSON, MIT]),
    ("J", "Maxwell equations", "current density", [ELLINGSON, MIT]),
    ("rho_v", "Maxwell equations", "volume charge density", [ELLINGSON, MIT]),
    ("epsilon_r", "material parameters", "relative permittivity", [ELLINGSON, MIT]),
    ("mu_r", "material parameters", "relative permeability", [ELLINGSON, MIT]),
    ("eta", "plane-wave propagation", "intrinsic impedance", [ELLINGSON, MIT]),
    ("Gamma", "transmission lines", "reflection coefficient", [ELLINGSON, MODERN, MIT]),
    ("alpha", "wave propagation", "attenuation constant", [ELLINGSON, MIT]),
    ("beta", "wave propagation", "phase propagation constant", [ELLINGSON, MIT]),
    ("gamma", "transmission-line propagation", "propagation constant", [ELLINGSON, MODERN, MIT]),
    ("k", "wave propagation", "wavenumber", [ELLINGSON, MIT]),
    ("lambda", "wave propagation", "wavelength", [ELLINGSON, MIT]),
    ("S12", "a two-port microwave network", "scattering parameters", [MODERN]),
    ("S22", "a two-port microwave network", "output reflection coefficient", [MODERN]),
]

MULTIPLE_EVIDENCE_TOPICS = [
    ("electric field E and electric flux density D", [ELLINGSON, MIT], ["electric field", "electric flux density"]),
    ("magnetic field H and magnetic flux density B", [ELLINGSON, MIT], ["magnetic field", "magnetic flux density"]),
    ("the differential forms of Faraday's law and the Ampere-Maxwell law", [ELLINGSON, MIT], ["Faraday", "Ampere", "curl"]),
    ("reflection coefficient and VSWR", [ELLINGSON, MODERN, MIT], ["reflection coefficient", "VSWR"]),
    ("S11 reflection and S21 transmission", [MODERN], ["S11", "S21", "scattering"]),
    ("antenna gain, directivity, and radiation efficiency", [MODERN, MIT], ["antenna gain", "directivity", "efficiency"]),
    ("antenna gain and effective aperture", [MODERN, MIT], ["antenna gain", "effective aperture"]),
    ("array factor and the total radiation pattern", [MODERN, MIT], ["array factor", "radiation pattern"]),
    ("LoS and NLoS basic transmission loss", [P1411], ["LoS", "NLoS", "basic transmission loss"]),
    ("P.1411 frequency range, scenarios, and limitations", [P1411], ["frequency range", "scenario", "valid for"]),
    ("street-canyon and over-rooftop NLoS propagation", [P1411], ["street canyon", "rooftop", "NLoS"]),
]

CROSS_SOURCE_SPECS = [
    ("Compare how the core Maxwell equations are presented across the EM references.", [ELLINGSON, MIT], ["Maxwell", "equations"]),
    ("Compare the definition of reflection coefficient across transmission-line and microwave-network references.", [ELLINGSON, MODERN, MIT], ["reflection coefficient"]),
    ("Compare characteristic impedance evidence across the available transmission-line sources.", [ELLINGSON, MODERN, MIT], ["characteristic impedance"]),
    ("Compare antenna gain definitions across the antenna references.", [MODERN, MIT], ["antenna gain"]),
    ("Compare Poynting-vector definitions across the foundational EM sources.", [ELLINGSON, MIT], ["Poynting vector"]),
]

HARD_NEGATIVE_QUERIES = [
    "Qubit gate fidelity calibration procedure",
    "Hospital medical MRI pulse sequence optimization",
    "ANSYS license server port configuration",
    "CUDA transformer kernel profiling",
    "PyAEDT scripting API documentation",
    "CST VBA macro debugging",
    "OpenEMS Python geometry API",
]

LONG_NOISY_SPECS = [
    ("I am trying to understand a two-port measurement and need the evidence that explains what S11 actually means at port 1 rather than a general discussion of antennas.", [MODERN], ["S11", "reflection coefficient"], "definition"),
    ("Please find the transmission-line relationship that connects voltage standing wave ratio with the magnitude of the reflection coefficient and explain the quantities in that expression.", [ELLINGSON, MODERN, MIT], ["VSWR", "reflection coefficient"], "relation"),
    ("For an antenna comparison, I need a source-grounded explanation of how gain differs from directivity and where radiation efficiency enters the relationship.", [MODERN, MIT], ["antenna gain", "directivity", "efficiency"], "relation"),
    ("Without using a narrow submodel range, identify the complete frequency span covered by Recommendation ITU-R P.1411 for outdoor short-range propagation.", [P1411], ["ITU-R P.1411", "300 MHz", "300 GHz"], "frequency_range"),
    ("I need the evidence for both line-of-sight and non-line-of-sight basic transmission loss in P.1411, including the terms that distinguish the two model families.", [P1411], ["LoS", "NLoS", "basic transmission loss"], "formula"),
    ("Locate the differential Maxwell equations and include enough evidence to distinguish the curl equations from the divergence equations rather than returning a generic chapter overview.", [ELLINGSON, MIT], ["Maxwell", "curl", "divergence"], "formula"),
    ("For matching two unequal transmission-line impedances, find the quarter-wave transformer condition and the formula that determines the required transformer impedance.", [MODERN, MIT], ["quarter-wave transformer", "impedance", "matched"], "formula"),
    ("Explain antenna radiation efficiency using the direct ratio between radiated power and accepted or input power, avoiding unrelated aperture-efficiency material.", [MODERN, MIT], ["radiation efficiency", "radiated power", "input power"], "formula"),
    ("Find a definition of a microstrip patch antenna that identifies the printed radiating patch and substrate rather than only mentioning a generic microstrip transmission line.", [MODERN], ["microstrip", "patch antennas", "substrate"], "definition"),
    ("I need the transmission-line tool that maps normalized impedance and reflection coefficient on a circular chart, together with a clear identification of that chart.", [MODERN, MIT], ["Smith chart", "impedance", "reflection coefficient"], "definition"),
    ("For a phased array, retrieve evidence that separates the array factor from the element pattern and shows how they contribute to the overall radiation pattern.", [MODERN, MIT], ["array factor", "element pattern", "radiation pattern"], "relation"),
    ("Summarize the assumptions and limitations attached to the ITU-R P.1411 short-range propagation models, including where scenario and frequency applicability constrain their use.", [P1411], ["valid for", "scenario", "frequency range"], "limitations"),
]


def normalize_source_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    for character in ("\\", "_", "{", "}"):
        text = text.replace(character, "")
    text = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", text)
    text = "".join(character if character.isalnum() else " " for character in text)
    return " ".join(text.split())


def _normalized_text_supports(normalized_text: str, normalized_phrase: str) -> bool:
    return bool(normalized_phrase) and f" {normalized_phrase} " in f" {normalized_text} "


def source_text_supports(source_text: Any, facet: Any) -> bool:
    return _normalized_text_supports(normalize_source_text(source_text), normalize_source_text(facet))


def source_check_action(row: dict[str, Any]) -> str:
    query_id = str(row.get("query_id") or "")
    if query_id in SOURCE_CHECK_REPLACED_IDS:
        return "replaced"
    if query_id in SOURCE_CHECK_FIXED_IDS:
        return "fixed"
    return "pass"


def source_check_candidates(rows: list[dict[str, Any]], canonical_path: Path) -> dict[str, Any]:
    canonical_path = Path(canonical_path)
    if not canonical_path.is_file():
        raise FileNotFoundError(f"canonical corpus not found: {canonical_path.as_posix()}")

    required_by_source: dict[str, set[str]] = {}
    for row in rows:
        if row.get("is_hard_negative"):
            continue
        facets = [normalize_source_text(facet) for facet in row.get("expected_evidence_facets") or []]
        for source_id in row.get("expected_sources") or []:
            required_by_source.setdefault(str(source_id), set()).update(facets)

    found_by_source = {source_id: set() for source_id in required_by_source}
    with canonical_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid canonical JSONL at line {line_number}") from exc
            source_id = str(chunk.get("source_id") or "")
            if source_id not in required_by_source:
                continue
            pending = required_by_source[source_id] - found_by_source[source_id]
            if not pending:
                continue
            normalized_fields = [
                normalize_source_text(chunk.get(field))
                for field in SOURCE_TEXT_FIELDS
                if str(chunk.get(field) or "").strip()
            ]
            for facet in pending:
                if any(_normalized_text_supports(field_text, facet) for field_text in normalized_fields):
                    found_by_source[source_id].add(facet)

    action_counts: Counter[str] = Counter()
    unresolved_count = 0
    for row in rows:
        if row.get("is_hard_negative"):
            resolved = True
        else:
            sources = [str(source_id) for source_id in row.get("expected_sources") or []]
            facets = [normalize_source_text(facet) for facet in row.get("expected_evidence_facets") or []]
            if row.get("category") == "cross_source":
                resolved = bool(sources and facets) and all(
                    facet in found_by_source.get(source_id, set())
                    for source_id in sources
                    for facet in facets
                )
            else:
                resolved = bool(sources and facets) and all(
                    any(facet in found_by_source.get(source_id, set()) for source_id in sources)
                    for facet in facets
                )
        if resolved:
            action_counts[source_check_action(row)] += 1
        else:
            unresolved_count += 1

    return {
        "status": "passed" if unresolved_count == 0 else "failed",
        "checked_count": len(rows),
        "pass_count": action_counts["pass"],
        "fixed_count": action_counts["fixed"],
        "replaced_count": action_counts["replaced"],
        "unresolved_count": unresolved_count,
        "origin_counts": dict(sorted(Counter(str(row.get("provenance") or "") for row in rows).items())),
    }


def build_source_check_summary(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    canonical_path: Path,
) -> dict[str, Any]:
    canonical_path = Path(canonical_path)
    if not canonical_path.is_file():
        not_run = {
            "status": "not_run",
            "checked_count": 0,
            "pass_count": 0,
            "fixed_count": 0,
            "replaced_count": 0,
            "unresolved_count": 0,
            "origin_counts": {},
        }
        return {
            "status": "not_run",
            "reason": "canonical_corpus_missing",
            "candidate": dict(not_run),
            "selected": dict(not_run),
        }

    candidate_summary = source_check_candidates(candidates, canonical_path)
    selected_summary = source_check_candidates(selected, canonical_path)
    status = (
        "passed"
        if candidate_summary["status"] == "passed" and selected_summary["status"] == "passed"
        else "failed"
    )
    return {
        "status": status,
        "candidate": candidate_summary,
        "selected": selected_summary,
    }


def make_case(
    category: str,
    index: int,
    query: str,
    expected_sources: list[str],
    expected_evidence_facets: list[str],
    *,
    expected_intent: str = "definition",
    requires_multi_citation: bool = False,
    is_hard_negative: bool = False,
) -> dict[str, Any]:
    return {
        "query_id": f"en_{category}_{index:03d}",
        "query": query,
        "language": "en",
        "category": category,
        "expected_intent": expected_intent,
        "expected_sources": list(expected_sources),
        "expected_evidence_facets": list(expected_evidence_facets),
        "is_hard_negative": is_hard_negative,
        "requires_web_check": False,
        "requires_multi_citation": requires_multi_citation,
        "notes": "Curated v2 candidate with source targets recorded in expected_sources.",
        "provenance": "curated_candidate_v2",
    }


def build_additional_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (query, sources, facets) in enumerate(DEFINITION_SPECS, start=1):
        rows.append(make_case("definition", index, query, sources, facets))

    formula_templates = [
        "What equation defines {label}, and which variables appear in it?",
        "State the formula for {label} and identify every term used in the expression.",
    ]
    index = 1
    for label, sources, facets in FORMULA_TOPICS:
        for query_template in formula_templates:
            rows.append(make_case("formula_variable", index, query_template.format(label=label), sources, facets, expected_intent="formula"))
            index += 1

    index = 1
    for label, sources, facets, variant_count in NUMERIC_RANGE_TOPICS:
        for query_template in NUMERIC_RANGE_TEMPLATES[:variant_count]:
            rows.append(make_case("numeric_range", index, query_template.format(label=label), sources, facets, expected_intent="frequency_range"))
            index += 1

    condition_templates = [
        "Under what conditions does {label} apply?",
        "What assumptions and limitations govern {label}?",
    ]
    index = 1
    for label, sources, facets in CONDITION_TOPICS:
        for query_template in condition_templates:
            if index > 19:
                break
            rows.append(make_case("condition_limitation", index, query_template.format(label=label), sources, facets, expected_intent="limitations"))
            index += 1

    for index, (symbol, context, meaning, sources) in enumerate(EXACT_SYMBOL_SPECS, start=1):
        rows.append(make_case("exact_symbol", index, f"What does the exact symbol {symbol} denote in {context}?", sources, [symbol, meaning]))

    multiple_templates = [
        "How are {label} related, and what evidence supports each quantity?",
        "Explain {label} together, including the definitions or formulas that connect them.",
    ]
    index = 1
    for label, sources, facets in MULTIPLE_EVIDENCE_TOPICS:
        for query_template in multiple_templates:
            rows.append(make_case("multiple_evidence", index, query_template.format(label=label), sources, facets, expected_intent="relation", requires_multi_citation=True))
            index += 1

    for index, (query, sources, facets) in enumerate(CROSS_SOURCE_SPECS, start=1):
        rows.append(make_case("cross_source", index, query, sources, facets, requires_multi_citation=True))

    for index, query in enumerate(HARD_NEGATIVE_QUERIES, start=1):
        rows.append(make_case("hard_negative", index, query, [], [], expected_intent="out_of_scope", is_hard_negative=True))

    for index, (query, sources, facets, intent) in enumerate(LONG_NOISY_SPECS, start=1):
        rows.append(make_case("long_noisy", index, query, sources, facets, expected_intent=intent, requires_multi_citation=len(facets) > 2))
    return rows


def validate_rows(rows: list[dict[str, Any]], *, expected_category_counts: dict[str, int]) -> None:
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for row in rows:
        query_id = str(row.get("query_id") or "").strip()
        query = str(row.get("query") or "").strip()
        if not query_id:
            raise ValueError("query_id must be non-empty")
        if query_id in seen_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen_ids.add(query_id)
        normalized_query = " ".join(query.lower().split())
        if not query:
            raise ValueError(f"query must be non-empty for {query_id}")
        if normalized_query in seen_queries:
            raise ValueError(f"duplicate query: {query}")
        seen_queries.add(normalized_query)
        if row.get("language") != "en":
            raise ValueError(f"language must be en for {query_id}")
        if contains_han(query):
            raise ValueError(f"Han code point found in query for {query_id}")
        facets = row.get("expected_evidence_facets")
        if not isinstance(facets, list) or not all(isinstance(item, str) for item in facets):
            raise ValueError(f"facets must be a list of strings for {query_id}")
        if any(contains_han(facet) for facet in facets):
            raise ValueError(f"Han code point found in facet for {query_id}")
        is_hard_negative = bool(row.get("is_hard_negative"))
        if is_hard_negative and facets:
            raise ValueError(f"hard negative must have no facets: {query_id}")
        if not is_hard_negative and not facets:
            raise ValueError(f"answerable case must have non-empty facets: {query_id}")
        if is_hard_negative and row.get("expected_sources"):
            raise ValueError(f"hard negative must have no expected sources: {query_id}")
        if not str(row.get("provenance") or "").strip():
            raise ValueError(f"provenance must be recorded for {query_id}")
    actual_counts = Counter(str(row.get("category") or "") for row in rows)
    if actual_counts != Counter(expected_category_counts):
        raise ValueError(f"category counts do not match approved matrix: actual={dict(actual_counts)} expected={expected_category_counts}")


def build_candidate_pool() -> list[dict[str, Any]]:
    candidates = copy.deepcopy(LEGACY_CASES) + build_additional_candidates()
    for row in candidates:
        fixed_facets = LEGACY_FACET_FIXES.get(str(row.get("query_id") or ""))
        if fixed_facets is not None:
            row["expected_evidence_facets"] = list(fixed_facets)
    validate_rows(candidates, expected_category_counts=CANDIDATE_COUNTS)
    return candidates


def stable_rank(row: dict[str, Any], purpose: str) -> str:
    payload = "|".join([SELECTION_SEED, purpose, str(row["category"]), str(row["query_id"]), str(row["query"])])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_cases(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_rows(candidates, expected_category_counts=CANDIDATE_COUNTS)
    selected: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        category_rows = [row for row in candidates if row["category"] == category]
        migrated = sorted([row for row in category_rows if row["provenance"] == "migrated_legacy_84"], key=lambda row: row["query_id"])
        curated = sorted([row for row in category_rows if row["provenance"] != "migrated_legacy_84"], key=lambda row: stable_rank(row, "accept"))
        accepted_count = ACCEPTED_COUNTS[category]
        if len(migrated) > accepted_count:
            raise ValueError(f"migrated cases exceed accepted quota for {category}")
        chosen = migrated + curated[: accepted_count - len(migrated)]
        if len(chosen) != accepted_count:
            raise ValueError(f"not enough candidates to fill accepted quota for {category}")
        ordered = sorted(chosen, key=lambda row: stable_rank(row, "split"))
        offset = 0
        for split in SPLIT_ORDER:
            count = SPLIT_MATRIX[category][split]
            for row in ordered[offset : offset + count]:
                selected_row = copy.deepcopy(row)
                selected_row["split"] = split
                selected.append(selected_row)
            offset += count
        if offset != len(ordered):
            raise ValueError(f"split matrix does not consume accepted cases for {category}")
    validate_rows(selected, expected_category_counts=ACCEPTED_COUNTS)
    actual_split_counts = Counter(row["split"] for row in selected)
    if actual_split_counts != Counter({"development": 80, "regression": 50, "holdout": 50}):
        raise ValueError(f"split counts do not match approved totals: {dict(actual_split_counts)}")
    return sorted(selected, key=lambda row: (SPLIT_ORDER.index(row["split"]), CATEGORY_ORDER.index(row["category"]), row["query_id"]))


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text_checked(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    if path.read_text(encoding="utf-8") != text:
        raise ValueError(f"UTF-8 roundtrip mismatch for {path.as_posix()}")


def validate_holdout_destination(destination: Path, protected_paths: list[Path]) -> Path:
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("holdout destination must not be a symlink")
    if destination.exists() and not destination.is_file():
        raise ValueError("holdout destination must be a regular file or a new path")

    existing_parent = destination.parent
    while not existing_parent.exists() and not existing_parent.is_symlink():
        if existing_parent == existing_parent.parent:
            break
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir():
        raise ValueError("holdout destination parent must resolve to a directory")

    resolved_destination = destination.resolve(strict=destination.exists())
    for protected_path in protected_paths:
        protected_path = Path(protected_path)
        resolved_protected = protected_path.resolve(strict=protected_path.exists())
        collision = (
            resolved_destination == resolved_protected
            or resolved_destination.is_relative_to(resolved_protected)
            or resolved_protected.is_relative_to(resolved_destination)
        )
        if not collision and destination.exists() and protected_path.exists():
            try:
                collision = destination.samefile(protected_path)
            except OSError:
                collision = False
        if collision:
            raise ValueError("holdout destination collides with a protected project path")
    return resolved_destination


def aggregate_source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get("expected_sources") or [])
    return dict(sorted(counts.items()))


def build_manifest(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    active_text: str,
    holdout_text: str,
    source_check: dict[str, Any],
) -> dict[str, Any]:
    selection_lines = [f"{row['query_id']}|{row['category']}|{row['split']}" for row in selected]
    return {
        "schema_version": "retrieval_quality_v2",
        "language": "en",
        "selection_seed": SELECTION_SEED,
        "candidate_count": len(candidates),
        "accepted_count": len(selected),
        "active_count": sum(row["split"] != "holdout" for row in selected),
        "holdout_count": sum(row["split"] == "holdout" for row in selected),
        "migrated_legacy_count": sum(row["provenance"] == "migrated_legacy_84" for row in candidates),
        "candidate_category_counts": CANDIDATE_COUNTS,
        "accepted_category_counts": ACCEPTED_COUNTS,
        "split_counts": dict(Counter(row["split"] for row in selected)),
        "category_split_matrix": SPLIT_MATRIX,
        "candidate_source_counts": aggregate_source_counts(candidates),
        "accepted_source_counts": aggregate_source_counts(selected),
        "active_dataset_path": ACTIVE_DATASET.as_posix(),
        "active_dataset_sha256": sha256_text(active_text),
        "holdout_dataset_sha256": sha256_text(holdout_text),
        "accepted_selection_sha256": sha256_text("\n".join(selection_lines) + "\n"),
        "holdout_policy": "Holdout JSONL is omitted from the repository, but the deterministic builder can materialize it with --write-holdout for an authorized local write; stdout remains aggregate-only.",
        "source_check_fields": ["expected_sources", "expected_evidence_facets", "provenance"],
        "source_check": source_check,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the English-only retrieval quality v2 dataset.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    parser.add_argument("--write-holdout", default="", help="Optional private JSONL destination for the 50 holdout cases.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    candidates = build_candidate_pool()
    selected = select_cases(candidates)
    canonical_path = project_root / CANONICAL_CORPUS
    source_check = build_source_check_summary(candidates, selected, canonical_path)
    if source_check["status"] == "failed":
        raise ValueError(
            "source check failed: "
            f"candidate_unresolved={source_check['candidate']['unresolved_count']} "
            f"selected_unresolved={source_check['selected']['unresolved_count']}"
        )
    active = [row for row in selected if row["split"] != "holdout"]
    holdout = [row for row in selected if row["split"] == "holdout"]
    active_counts = {category: SPLIT_MATRIX[category]["development"] + SPLIT_MATRIX[category]["regression"] for category in CATEGORY_ORDER}
    holdout_counts = {category: SPLIT_MATRIX[category]["holdout"] for category in CATEGORY_ORDER}
    validate_rows(active, expected_category_counts=active_counts)
    validate_rows(holdout, expected_category_counts=holdout_counts)
    active_text = jsonl_text(active)
    holdout_text = jsonl_text(holdout)
    manifest = build_manifest(candidates, selected, active_text, holdout_text, source_check)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    active_path = project_root / ACTIVE_DATASET
    manifest_path = project_root / MANIFEST_PATH
    holdout_path = None
    if args.write_holdout:
        requested = Path(args.write_holdout)
        destination = requested if requested.is_absolute() else project_root / requested
        holdout_path = validate_holdout_destination(
            destination,
            [active_path, manifest_path, canonical_path],
        )
    write_text_checked(active_path, active_text)
    write_text_checked(manifest_path, manifest_text)
    if holdout_path is not None:
        write_text_checked(holdout_path, holdout_text)
    summary = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "candidate_count": len(candidates),
        "accepted_count": len(selected),
        "active_count": len(active),
        "holdout_count": len(holdout),
        "candidate_category_counts": CANDIDATE_COUNTS,
        "split_counts": dict(Counter(row["split"] for row in selected)),
        "source_check": source_check,
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
