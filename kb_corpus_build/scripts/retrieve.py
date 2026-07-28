#!/usr/bin/env python3
"""Phase-12 hybrid retrieval over BM25, optional vectors, and structured SQLite indexes."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_bm25_index import normalize_rf_notation, score_bm25, tokenize_for_bm25
from unicode_han import contains_han


RRF_K = 60
DEFAULT_VECTOR_TIMEOUT_SECONDS = 45
DEFAULT_HF_CACHE_RELATIVE = Path("kb_corpus_build") / ".cache" / "huggingface"

QUERY_ALIASES = [
    ("s11", "S11 reflection coefficient scattering parameter"),
    ("s21", "S21 transmission coefficient scattering parameter"),
    ("s-parameters", "S parameters scattering parameters scattering matrix S11 S21 microwave network"),
    ("s parameters", "S parameters scattering parameters scattering matrix S11 S21 microwave network"),
    ("scattering matrix", "S parameters scattering parameters S11 S21 microwave network"),
    ("z0", "Z0 characteristic impedance transmission line"),
    ("radiation efficiency", "antenna efficiency eta input power radiated power directivity antenna gain"),
    ("scattering parameters", "S parameters S11 S21 transmission coefficient reflection coefficient microwave network"),
    ("maxwell equations", "curl div divergence Faraday Ampere Gauss partial rho"),
    ("itu-r p.1411 frequency range", "Scope outdoor short-range propagation frequency range 300 MHz 300 GHz"),
    ("itu-r p.1411 300 mhz 300 ghz frequency range", "Scope outdoor short-range propagation frequency range 300 MHz 300 GHz"),
    ("short-range outdoor propagation", "outdoor short-range propagation Scope frequency range 300 MHz 300 GHz"),
    ("propagation scenario in itu-r p.1411", "physical operating environments propagation impairments street canyon over rooftops LoS NLoS"),
    ("input parameters for itu-r p.1411 los model", "LoS situation input parameters distance frequency antenna heights x1 x2"),
    ("limitations of itu-r p.1411 models", "valid for distance range frequency range recommended for use limitations"),
    ("street canyon nlos model", "Propagation along street canyons NLoS dense urban micro-cellular NLoS2 w1 w2 x1 x2"),
    ("above rooftop propagation", "Propagation over rooftops roof-tops non-line-of-sight NLoS1 average building height"),
    ("urban microcell propagation", "urban micro-cellular micro-cells street canyons NLoS dense urban"),
    ("los path loss", "LoS line-of-sight LLoS basic transmission loss exponent distance atmospheric attenuation"),
    ("nlos path loss", "NLoS non-line-of-sight LNLoS basic transmission loss reflection loss diffraction loss"),
]

OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"\borbital\s+mechanics\b", re.IGNORECASE),
    re.compile(r"\borbital\s+transfer\b", re.IGNORECASE),
    re.compile(r"\bhohmann\b", re.IGNORECASE),
    re.compile(r"\bfinite\s+element\s+meshing\b", re.IGNORECASE),
    re.compile(r"\bmesh\s+generation\b", re.IGNORECASE),
    re.compile(r"\bsolid\s+mechanics\b", re.IGNORECASE),
    re.compile(r"\bpython\s+api\b", re.IGNORECASE),
    re.compile(r"\bpython\b.*\bapi\b.*\bmesh\b", re.IGNORECASE),
    re.compile(r"\bpyaedt\b", re.IGNORECASE),
    re.compile(r"\bhfss\s+project\b", re.IGNORECASE),
    re.compile(r"\bcst\b.*\bvba\b", re.IGNORECASE),
    re.compile(r"\bopenems\b", re.IGNORECASE),
    re.compile(r"\bquantum\s+computing\b", re.IGNORECASE),
    re.compile(r"\bqubit\b", re.IGNORECASE),
    re.compile(r"\bmedical\s+mri\b", re.IGNORECASE),
    re.compile(r"\bansys\s+license\b", re.IGNORECASE),
    re.compile(r"\blicense\s+server\b", re.IGNORECASE),
    re.compile(r"\bmatlab\s+antenna\s+toolbox\b", re.IGNORECASE),
    re.compile(r"\bcuda\b.*\btransformer\b", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve cited evidence from the local EM RAG indexes.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    parser.add_argument("--query", required=True, help="User query.")
    parser.add_argument("--top-k", type=int, default=12, help="Maximum evidence rows to return.")
    parser.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "bm25_structured"],
        default="hybrid",
        help=(
            "Use BM25+structured retrieval by default. In hybrid mode, vector scoring only starts "
            "when EM_RAG_ENABLE_VECTOR=1."
        ),
    )
    return parser.parse_args()


def normalize_scalar(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def expand_query(query: str) -> str:
    lowered = normalize_scalar(query).lower()
    additions = []
    for needle, expansion in QUERY_ALIASES:
        if needle.lower() in lowered:
            additions.append(expansion)
    return " ".join([normalize_scalar(query), *additions]).strip()


def is_out_of_scope_query(query: str) -> bool:
    text = normalize_scalar(query)
    return any(pattern.search(text) for pattern in OUT_OF_SCOPE_PATTERNS)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_text_checked(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    read_back = path.read_text(encoding="utf-8")
    if read_back != text:
        raise ValueError(f"UTF-8 roundtrip mismatch for {path.as_posix()}")


def default_huggingface_cache_dir(project_root: Path | None = None) -> Path:
    root = project_root.resolve() if project_root is not None else Path.cwd().resolve()
    return root / DEFAULT_HF_CACHE_RELATIVE


def ensure_huggingface_cache_env(project_root: Path | None = None) -> Path:
    existing = os.environ.get("HF_HOME")
    if existing:
        return Path(existing)
    cache_dir = default_huggingface_cache_dir(project_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    return cache_dir


def vector_worker_env(project_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if env.get("HF_HOME"):
        cache_dir = Path(env["HF_HOME"])
    else:
        cache_dir = default_huggingface_cache_dir(project_root)
        cache_dir.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(cache_dir)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def vector_timeout_seconds() -> int:
    raw = os.environ.get("EM_RAG_VECTOR_TIMEOUT_SECONDS", str(DEFAULT_VECTOR_TIMEOUT_SECONDS))
    try:
        timeout = int(raw)
    except ValueError:
        return DEFAULT_VECTOR_TIMEOUT_SECONDS
    return max(1, timeout)


def vector_runtime_enabled() -> bool:
    return os.environ.get("EM_RAG_ENABLE_VECTOR", "").strip().lower() in {"1", "true", "yes", "on"}


def append_jsonl_checked(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    new_text = existing + json.dumps(row, ensure_ascii=False) + "\n"
    write_text_checked(path, new_text)


def load_bm25(build_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    index_path = build_root / "indexes" / "bm25" / "bm25_index.pkl"
    docstore_path = build_root / "indexes" / "bm25" / "bm25_docstore.jsonl"
    with index_path.open("rb") as fh:
        index = pickle.load(fh)
    docstore = {row["chunk_id"]: row for row in load_jsonl(docstore_path)}
    return index, docstore


def ranked_scores(items: list[tuple[str, float]]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for rank, (chunk_id, raw_score) in enumerate(items, start=1):
        scores[chunk_id] = {"score": float(raw_score), "rank": float(rank)}
    return scores


def query_terms(query: str) -> list[str]:
    tokens = []
    seen: set[str] = set()
    for token in tokenize_for_bm25(query):
        if len(token) < 3:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def score_structured_text(search_text: str, tokens: list[str]) -> float:
    lowered = search_text.lower()
    score = 0.0
    for token in tokens:
        token_text = token.replace("_", " ")
        if token in lowered or token_text in lowered:
            score += 1.0
    return score


def normalize_intent_text(value: Any) -> str:
    text = normalize_rf_notation(normalize_scalar(value)).lower()
    text = text.replace("roof-tops", "rooftops").replace("roof-top", "rooftop")
    text = text.replace("micro-cellular", "microcellular").replace("micro-cells", "microcells")
    text = text.replace("s-parameters", "s parameters").replace("s-parameter", "s parameter")
    text = re.sub(r"\s+", " ", text)
    return text


def document_intent_text(doc: dict[str, Any]) -> str:
    parts = [
        doc.get("source_id"),
        doc.get("source_title"),
        doc.get("chapter"),
        doc.get("section"),
        " ".join(doc.get("keywords") or []),
        doc.get("content_md") or doc.get("retrieval_text"),
    ]
    return normalize_intent_text(" ".join(normalize_scalar(part) for part in parts))


def is_itu_p1411_intent(query_text: str) -> bool:
    return any(
        marker in query_text
        for marker in [
            "itu-r p.1411",
            "p.1411",
            "short-range outdoor",
            "street canyon",
            "rooftop",
            "roof top",
            "urban microcell",
            "los path loss",
            "nlos path loss",
        ]
    )


def is_acoustic_context(doc_text: str) -> bool:
    return "acoustic waveguide" in doc_text or "chapter 13: acoustics" in doc_text or "chapter 13 acoustics" in doc_text


def is_path_loss_metric_noise(doc_text: str) -> bool:
    return any(
        marker in doc_text
        for marker in [
            "delay spread",
            "angular profile",
            "arrival angle",
            "multipath delay",
            "path morphology",
        ]
    )


def is_s_parameter_intent(query_text: str) -> bool:
    return any(
        marker in query_text
        for marker in [
            "s11",
            "s21",
            "s parameter",
            "scattering parameter",
            "scattering matrix",
        ]
    )


def is_s_parameter_definition_context(doc_text: str) -> bool:
    return any(
        marker in doc_text
        for marker in [
            "scattering matrix",
            "scattering parameter",
            "s parameter",
            "s-parameter",
            "two-port network",
            "k-port microwave network",
            "port voltages and currents",
            "incident waves",
            "reflected waves",
        ]
    )


def is_s_parameter_definition_locus(chapter_text: str, section_text: str) -> bool:
    return any(
        marker in section_text
        for marker in [
            "microwave networks",
            "s parameters",
            "scattering parameters",
        ]
    ) or ("microwave circuits" in chapter_text and "network" in section_text)


def is_s_parameter_application_locus(chapter_text: str, section_text: str, doc_text: str) -> bool:
    metadata_text = f"{chapter_text} {section_text}"
    return any(
        marker in metadata_text
        for marker in [
            "mutual coupling",
            "phased array",
            "phased arrays",
            "microwave amplifiers",
            "power combiners",
            "low-noise amplifier",
            "power amplifier",
        ]
    ) or "active reflection coefficient" in doc_text


def is_generic_s_parameter_query(query_text: str) -> bool:
    if not is_s_parameter_intent(query_text):
        return False
    return not any(
        marker in query_text
        for marker in [
            "mutual coupling",
            "active reflection",
            "phased array",
            "phased-array",
            "amplifier",
            "power combiner",
        ]
    )


def score_s_parameter_intent(
    query_text: str,
    doc_text: str,
    source_id: str,
    chapter_text: str = "",
    section_text: str = "",
) -> float:
    if not is_s_parameter_intent(query_text):
        return 0.0

    score = 0.0
    mentions_s11 = "s11" in doc_text
    mentions_s21 = "s21" in doc_text
    has_scattering_context = any(
        marker in doc_text
        for marker in [
            "scattering matrix",
            "scattering parameter",
            "microwave network",
            "vector network analyzer",
            "two-port network",
        ]
    )
    has_s_parameter_definition = is_s_parameter_definition_context(doc_text) and (mentions_s11 or mentions_s21)
    definition_locus = is_s_parameter_definition_locus(chapter_text, section_text)
    application_locus = is_s_parameter_application_locus(chapter_text, section_text, doc_text)

    if "s11" in query_text:
        if mentions_s11 and has_s_parameter_definition:
            score += 14.0
        elif mentions_s11 and has_scattering_context:
            score += 8.0
        elif mentions_s11:
            score += 8.0
        elif "reflection coefficient" in doc_text:
            score -= 6.0

    if "s21" in query_text:
        if mentions_s21 and has_s_parameter_definition:
            score += 14.0
        elif mentions_s21 and has_scattering_context:
            score += 8.0
        elif mentions_s21:
            score += 8.0
        elif "transmission coefficient" in doc_text or "basic transmission loss" in doc_text:
            score -= 6.0

    if ("s parameter" in query_text or "scattering parameter" in query_text or "scattering matrix" in query_text):
        if has_s_parameter_definition:
            score += 10.0
        elif "reflection coefficient" in doc_text or "transmission coefficient" in doc_text:
            score -= 4.0

    if has_scattering_context and not has_s_parameter_definition and any(
        marker in doc_text for marker in ["mutual coupling", "phased array", "power amplifier", "low-noise amplifier"]
    ):
        score -= 4.0

    if source_id == "modern_antennas_microwave_circuits" and has_s_parameter_definition:
        score += 2.0
    if is_generic_s_parameter_query(query_text) and has_s_parameter_definition:
        if definition_locus:
            score += 4.0
        elif application_locus:
            score -= 8.0
    if "standing wave ratio" in doc_text or "vswr" in doc_text:
        score -= 8.0
    if is_acoustic_context(doc_text):
        score -= 12.0
    return score


def has_vswr_gamma_formula(doc_text: str) -> bool:
    has_swr = "swr" in doc_text or "vswr" in doc_text or "standing wave ratio" in doc_text
    has_gamma = "gamma" in doc_text or "reflection coefficient" in doc_text
    symbolic_formula = has_swr and has_gamma and any(
        marker in doc_text
        for marker in [
            "swr =",
            "vswr =",
            "mbox{swr}",
            "left|gamma",
            "|gamma|",
            "swr - 1",
            "swr + 1",
        ]
    )
    word_formula = has_swr and (
        "one plus the magnitude of gamma" in doc_text and "one minus the magnitude of gamma" in doc_text
    )
    return symbolic_formula or word_formula


def is_vswr_definition_section(section_text: str) -> bool:
    return "standing wave ratio" in section_text or section_text == "vswr"


def is_p1411_overall_frequency_query(query_text: str) -> bool:
    if not (
        ("p.1411" in query_text or "itu-r p.1411" in query_text)
        and ("frequency range" in query_text or "300 mhz" in query_text or "300 ghz" in query_text)
    ):
        return False
    submodel_markers = [
        "limitations",
        "input parameters",
        "los path loss",
        "nlos path loss",
        "basic transmission loss",
        "street canyon",
        "urban area",
        "submodel",
        "millimetre-wave",
        "multipath",
        "delay spread",
        "scenario",
    ]
    return not any(marker in query_text for marker in submodel_markers)


def is_p1411_scope_doc(doc_text: str) -> bool:
    return "scope" in doc_text and "300 mhz to 300 ghz" in doc_text


def is_p1411_overall_frequency_noise_doc(doc_text: str) -> bool:
    return not is_p1411_scope_doc(doc_text)


def is_p1411_submodel_frequency_doc(doc_text: str) -> bool:
    if is_p1411_scope_doc(doc_text):
        return False
    return "frequency range" in doc_text and any(
        marker in doc_text
        for marker in [
            "frequency range from",
            "800 to 2 000 mhz",
            "2 to 38 ghz",
            "up to 26 ghz",
            "up to 38 ghz",
            "4.1.3.1",
            "4.2.2.1",
            "nlos2",
            "urban area",
        ]
    )


def query_intent_score(query: str, doc: dict[str, Any]) -> float:
    query_text = normalize_intent_text(query)
    doc_text = document_intent_text(doc)
    chapter_text = normalize_intent_text(doc.get("chapter"))
    section_text = normalize_intent_text(doc.get("section"))
    source_id = normalize_scalar(doc.get("source_id"))
    score = 0.0
    scenario_overview_query = "propagation scenario" in query_text and "p.1411" in query_text

    if is_itu_p1411_intent(query_text):
        if source_id != "itu_r_p1411_13":
            return 0.0
        score += 1.0

    score += score_s_parameter_intent(
        query_text,
        doc_text,
        source_id,
        chapter_text,
        section_text,
    )

    if "vswr" in query_text or "standing wave ratio" in query_text:
        has_vswr = "vswr" in doc_text or "standing wave ratio" in doc_text or "standing-wave ratio" in doc_text
        definition_like = any(marker in doc_text for marker in ["defined", "definition", "equals", "mbox{vswr}", "vswr ="])
        relates_reflection = any(marker in doc_text for marker in ["gamma", "reflection coefficient", "magnitude"])
        explicit_vswr_gamma_formula = has_vswr_gamma_formula(doc_text)
        direct_vswr_definition = (
            ("voltage-standing-wave ratio" in doc_text or "voltage standing wave ratio" in doc_text)
            and "equals" in doc_text
        )
        if has_vswr and explicit_vswr_gamma_formula:
            score += 12.0
        elif has_vswr and direct_vswr_definition and relates_reflection:
            score += 9.0
        elif has_vswr and definition_like and relates_reflection:
            score += 6.0
        elif has_vswr and definition_like:
            score += 4.0
        if has_vswr and is_vswr_definition_section(section_text):
            score += 3.0
        if has_vswr and "example" in doc_text and not direct_vswr_definition:
            score -= 3.0
        if has_vswr and "smith chart" in doc_text and not explicit_vswr_gamma_formula:
            score -= 5.0
        if has_vswr and is_acoustic_context(doc_text):
            score -= 14.0

    if "frequency range" in query_text and ("p.1411" in query_text or "300 mhz" in query_text):
        if "scope" in doc_text and "300 mhz to 300 ghz" in doc_text:
            score += 6.0

    if "short-range outdoor" in query_text:
        if "scope" in doc_text and ("outdoor short-range propagation" in doc_text or "short-range outdoor" in doc_text):
            score += 6.0

    if "propagation scenario" in query_text and "p.1411" in query_text:
        if "five different environments" in doc_text:
            score += 8.0
        elif "physical operating environments" in doc_text:
            score += 6.0
        elif "propagation over rooftops" in doc_text or "propagation along street canyons" in doc_text:
            score += 4.0

    if "input parameters" in query_text and "los" in query_text:
        if "llos" in doc_text and "basic transmission loss exponent" in doc_text:
            score += 10.0
        elif "4.3.2.1 los situation this situation is depicted" in doc_text:
            score += 8.0
        elif "4.3.2.1 los situation" in doc_text and (
            "input parameters" in doc_text or "relevant parameters" in doc_text or "parameters include" in doc_text
        ):
            score += 6.0
        elif "4.3.2.1 los situation" in doc_text:
            score += 4.0

    if "path loss" in query_text and "los" in query_text and "nlos" not in query_text:
        if "llos" in doc_text and "basic transmission loss exponent" in doc_text:
            score += 14.0
        elif "llos" in doc_text and "basic transmission loss" in doc_text:
            score += 11.0
        elif "line-of-sight" in doc_text and "basic transmission loss" in doc_text:
            score += 8.0
        elif "4.3.2.1 los situation" in doc_text:
            score += 3.0
        if "nlos" in doc_text and "llos" not in doc_text:
            score -= 4.0
        if is_path_loss_metric_noise(doc_text):
            score -= 6.0

    if "path loss" in query_text and "nlos" in query_text:
        if "lnlos" in doc_text and ("reflection loss" in doc_text or "diffraction loss" in doc_text):
            score += 14.0
        elif "lnlos" in doc_text or ("reflection loss" in doc_text and "diffraction loss" in doc_text):
            score += 11.0
        elif "non-line-of-sight" in doc_text and "basic transmission loss" in doc_text:
            score += 8.0
        elif "propagation along street canyons" in doc_text:
            score += 3.0
        if is_path_loss_metric_noise(doc_text):
            score -= 6.0

    if "limitations" in query_text and "p.1411" in query_text:
        if "valid for" in doc_text or "recommended for use" in doc_text:
            score += 6.0
        elif "distance range" in doc_text and "frequency range" in doc_text:
            score += 4.0

    if not scenario_overview_query and ("street canyon" in query_text or ("street" in query_text and "nlos" in query_text)):
        if "3.1.2 propagation along street canyons" in doc_text:
            score += 6.0

    if not scenario_overview_query and (
        "below rooftop" in query_text or "below-rooftop" in query_text or "below rooftops" in query_text
    ):
        if "5.4.2 below-rooftops propagation environments" in doc_text:
            score += 8.0
        elif "both the transmitting and receiving stations are located below-rooftop" in doc_text:
            score += 7.0
        elif "below-rooftop" in doc_text or "below-rooftops" in doc_text:
            score += 5.0
    elif not scenario_overview_query and ("rooftop" in query_text or "roof top" in query_text):
        if "3.1.1 propagation over rooftops" in doc_text:
            score += 8.0
        elif "over rooftops propagation" in doc_text:
            score += 5.0
        elif "rooftop" in doc_text or "rooftops" in doc_text:
            score += 2.0

    if "urban microcell" in query_text or ("urban" in query_text and "microcell" in query_text):
        if "3.1.2 propagation along street canyons" in doc_text and "urban microcellular" in doc_text:
            score += 8.0
        elif "urban microcellular" in doc_text or "urban microcells" in doc_text:
            score += 6.0
        elif "microcell" in doc_text and "urban" in doc_text:
            score += 4.0

    generic_phased_array_query = (
        ("phased array" in query_text or "phased-array" in query_text)
        and "mutual coupling" not in query_text
        and "calibration" not in query_text
        and "beamforming" not in query_text
    )
    if generic_phased_array_query:
        if "linear phased arrays of isotropic antennas" in doc_text:
            score += 8.0
        elif "phased array" in doc_text and ("array factor" in doc_text or "beam" in doc_text):
            score += 5.0
        if "mutual coupling" in doc_text:
            score -= 6.0
        if "calibration of phased-arrays" in doc_text:
            score -= 4.0

    return score


def collect_structured_scores(build_root: Path, query: str) -> dict[str, float]:
    structured_dir = build_root / "indexes" / "structured"
    tokens = query_terms(query)
    scores: defaultdict[str, float] = defaultdict(float)
    if not tokens:
        return {}

    table_specs = [
        (structured_dir / "formula.sqlite", "formulas"),
        (structured_dir / "terms.sqlite", "terms"),
        (structured_dir / "propagation_models.sqlite", "propagation_models"),
    ]
    for db_path, table_name in table_specs:
        if not db_path.is_file():
            continue
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"select related_chunk_ids_json, search_text from {table_name}").fetchall()
        for row in rows:
            score = score_structured_text(str(row["search_text"] or ""), tokens)
            if score <= 0.0:
                continue
            try:
                related_chunk_ids = json.loads(row["related_chunk_ids_json"] or "[]")
            except json.JSONDecodeError:
                related_chunk_ids = []
            for chunk_id in related_chunk_ids:
                normalized = normalize_scalar(chunk_id)
                if normalized:
                    scores[normalized] += score
    return dict(scores)


def collect_vector_scores_with_status(
    build_root: Path,
    query: str,
    timeout_seconds: int | None = None,
) -> tuple[dict[str, float], str, str]:
    vector_dir = build_root / "indexes" / "vector"
    if not (vector_dir / "faiss.index").is_file():
        return {}, "missing_index", "vector faiss.index is not available"
    metadata_path = vector_dir / "index_metadata.json"
    docstore_path = vector_dir / "docstore.jsonl"
    if not metadata_path.is_file() or not docstore_path.is_file():
        return {}, "missing_metadata", "vector metadata or docstore is not available"
    worker_path = Path(__file__).resolve().with_name("vector_query_worker.py")
    if not worker_path.is_file():
        return {}, "missing_worker", f"vector worker not found: {worker_path.as_posix()}"
    timeout = timeout_seconds if timeout_seconds is not None else vector_timeout_seconds()
    argv = [
        sys.executable,
        str(worker_path),
        "--build-root",
        str(build_root),
        "--query",
        query,
    ]
    try:
        completed = subprocess.run(
            argv,
            cwd=build_root.parent,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=vector_worker_env(build_root.parent),
        )
    except subprocess.TimeoutExpired:
        return {}, "timeout", f"vector worker timed out after {timeout} seconds"
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "").strip()
        return {}, "error", error[:1000] or f"vector worker exited with code {completed.returncode}"
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {}, "invalid_output", f"vector worker returned invalid JSON: {exc}"
    status = normalize_scalar(payload.get("status")) or "error"
    if status != "ok":
        return {}, status, normalize_scalar(payload.get("error"))
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict):
        return {}, "invalid_output", "vector worker returned non-object scores"
    scores: dict[str, float] = {}
    for chunk_id, score in raw_scores.items():
        normalized_id = normalize_scalar(chunk_id)
        if not normalized_id:
            continue
        try:
            scores[normalized_id] = float(score)
        except (TypeError, ValueError):
            continue
    return scores, "ok", ""


def collect_batch_vector_scores_with_status(
    build_root: Path,
    query_by_id: dict[str, str],
    timeout_seconds: int | None = None,
) -> tuple[dict[str, dict[str, float]], str, str]:
    if not query_by_id:
        return {}, "ok", ""
    vector_dir = build_root / "indexes" / "vector"
    if not (vector_dir / "faiss.index").is_file():
        return {}, "missing_index", "vector faiss.index is not available"
    metadata_path = vector_dir / "index_metadata.json"
    docstore_path = vector_dir / "docstore.jsonl"
    if not metadata_path.is_file() or not docstore_path.is_file():
        return {}, "missing_metadata", "vector metadata or docstore is not available"
    worker_path = Path(__file__).resolve().with_name("vector_query_worker.py")
    if not worker_path.is_file():
        return {}, "missing_worker", f"vector worker not found: {worker_path.as_posix()}"
    timeout = timeout_seconds if timeout_seconds is not None else vector_timeout_seconds()
    argv = [
        sys.executable,
        str(worker_path),
        "--build-root",
        str(build_root),
        "--queries-json",
        json.dumps(query_by_id, ensure_ascii=False),
    ]
    try:
        completed = subprocess.run(
            argv,
            cwd=build_root.parent,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=vector_worker_env(build_root.parent),
        )
    except subprocess.TimeoutExpired:
        return {}, "timeout", f"vector batch worker timed out after {timeout} seconds"
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "").strip()
        return {}, "error", error[:1000] or f"vector batch worker exited with code {completed.returncode}"
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {}, "invalid_output", f"vector batch worker returned invalid JSON: {exc}"
    status = normalize_scalar(payload.get("status")) or "error"
    if status != "ok":
        return {}, status, normalize_scalar(payload.get("error"))
    raw_scores_by_query = payload.get("scores_by_query")
    if not isinstance(raw_scores_by_query, dict):
        return {}, "invalid_output", "vector batch worker returned non-object scores_by_query"

    scores_by_query: dict[str, dict[str, float]] = {}
    for query_id in query_by_id:
        raw_scores = raw_scores_by_query.get(query_id, {})
        if not isinstance(raw_scores, dict):
            scores_by_query[query_id] = {}
            continue
        scores: dict[str, float] = {}
        for chunk_id, score in raw_scores.items():
            normalized_id = normalize_scalar(chunk_id)
            if not normalized_id:
                continue
            try:
                scores[normalized_id] = float(score)
            except (TypeError, ValueError):
                continue
        scores_by_query[query_id] = scores
    return scores_by_query, "ok", ""


def run_vector_runtime_preflight(
    project_root: Path,
    output_root: Path,
    timeout_seconds: int | None = None,
) -> tuple[str, str]:
    script_path = Path(__file__).resolve().with_name("vector_runtime_preflight.py")
    if not script_path.is_file():
        return "missing_preflight", f"vector preflight script not found: {script_path.as_posix()}"
    timeout = timeout_seconds if timeout_seconds is not None else vector_timeout_seconds()
    argv = [
        sys.executable,
        str(script_path),
        "--project-root",
        str(project_root),
        "--output-dir",
        str(output_root),
    ]
    try:
        completed = subprocess.run(
            argv,
            cwd=project_root,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=vector_worker_env(project_root),
        )
    except subprocess.TimeoutExpired:
        return "timeout", f"vector runtime preflight timed out after {timeout} seconds"
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "").strip()
        return "error", error[:1000] or f"vector runtime preflight exited with code {completed.returncode}"
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return "invalid_output", f"vector runtime preflight returned invalid JSON: {exc}"
    status = normalize_scalar(payload.get("status")) or "error"
    if status != "ok":
        return status, normalize_scalar(payload.get("error"))
    return "ok", ""


def collect_vector_scores(build_root: Path, query: str) -> dict[str, float]:
    scores, _status, _error = collect_vector_scores_with_status(build_root, query)
    return scores


def normalize_score_map(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    max_score = max(score_map.values())
    if max_score <= 0.0:
        return {key: 0.0 for key in score_map}
    return {key: value / max_score for key, value in score_map.items()}


def rank_from_score_map(score_map: dict[str, float]) -> dict[str, int]:
    ranked = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
    return {chunk_id: rank for rank, (chunk_id, _score) in enumerate(ranked, start=1)}


def content_preview(text: str, limit: int = 360) -> str:
    normalized = " ".join(normalize_scalar(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def query_preview_needles(query: str) -> list[str]:
    query_text = normalize_intent_text(query)
    needles: list[str] = []
    if "path loss" in query_text and "los" in query_text and "nlos" not in query_text:
        needles.extend(["LLoS", "basic transmission loss exponent", "Lgas", "Lrain"])
    if "path loss" in query_text and "nlos" in query_text:
        needles.extend(["LNLoS", "reflection loss", "diffraction loss", "basic transmission loss"])
    if "frequency range" in query_text or "300 mhz" in query_text:
        needles.extend(["300 MHz to 300 GHz", "frequency range"])
    if "limitations" in query_text:
        needles.extend(["recommended for use", "valid for", "distance range", "frequency range"])
    if "input parameters" in query_text:
        needles.extend(["Input parameters", "relevant parameters", "parameters include", "where"])
    if "street canyon" in query_text or ("nlos" in query_text and "path loss" not in query_text):
        needles.append("Propagation along street canyons")
    if "rooftop" in query_text or "roof top" in query_text:
        needles.extend(["Propagation over rooftops", "over roof-tops propagation", "over rooftops propagation"])
    if "urban microcell" in query_text or ("urban" in query_text and "microcell" in query_text):
        needles.extend(["urban micro-cellular", "urban microcellular", "urban high-rise environment for micro-cells"])
    if "scenario" in query_text:
        needles.extend(
            [
                "Environment Description and propagation impairments",
                "Urban very high-rise",
                "Five different environments",
                "Physical operating environments",
            ]
        )
    if "s11" in query_text:
        needles.extend(["S_{11}", "S11", "reflection coefficient at port 1"])
    if "s21" in query_text:
        needles.extend(["S_{21}", "S21", "forward gain", "transmission coefficient"])
    if "s parameter" in query_text or "scattering parameter" in query_text or "scattering matrix" in query_text:
        needles.extend(["scattering matrix", "scattering parameters", "microwave network"])
    if "reflection coefficient" in query_text:
        needles.append("reflection coefficient")

    for token in query_terms(query_text):
        if token not in {"itu", "1411", "model", "models", "propagation", "what", "with"}:
            needles.append(token)

    unique_needles: list[str] = []
    seen: set[str] = set()
    for needle in needles:
        key = needle.lower()
        if key and key not in seen:
            seen.add(key)
            unique_needles.append(needle)
    return unique_needles


def content_preview_for_query(text: str, query: str, limit: int = 360) -> str:
    normalized = " ".join(normalize_scalar(text).split())
    if len(normalized) <= limit:
        return normalized

    lowered = normalized.lower()
    best_index: int | None = None
    for needle in query_preview_needles(query):
        index = lowered.find(needle.lower())
        if index < 0:
            continue
        best_index = index
        break

    if best_index is None:
        return content_preview(normalized, limit)

    start = max(0, best_index - 80)
    end = min(len(normalized), start + limit)
    preview = normalized[start:end].strip()
    if start > 0:
        preview = "..." + preview
    if end < len(normalized):
        preview = preview.rstrip() + "..."
    return preview


def build_result(
    rank: int,
    chunk_id: str,
    doc: dict[str, Any],
    query: str,
    final_score: float,
    bm25_score: float,
    vector_score: float,
    structured_score: float,
    intent_score: float,
    rrf_score: float,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "final_score": final_score,
        "chunk_id": chunk_id,
        "source_id": doc.get("source_id"),
        "source_title": doc.get("source_title"),
        "chapter": doc.get("chapter"),
        "section": doc.get("section"),
        "page_start": doc.get("page_start"),
        "page_end": doc.get("page_end"),
        "content_type": doc.get("content_type"),
        "domain_tags": doc.get("domain_tags") or [],
        "keywords": doc.get("keywords") or [],
        "content_preview": content_preview_for_query(str(doc.get("content_md") or doc.get("retrieval_text") or ""), query),
        "scores": {
            "bm25": bm25_score,
            "vector": vector_score,
            "structured": structured_score,
            "intent": intent_score,
            "rrf": rrf_score,
        },
        "citation": doc.get("citation") or "",
    }


def retrieve(
    project_root: Path,
    query: str,
    top_k: int,
    retrieval_mode: str = "hybrid",
    vector_scores_override: dict[str, float] | None = None,
    vector_status_override: str = "",
    vector_error_override: str = "",
) -> dict[str, Any]:
    if retrieval_mode not in {"hybrid", "bm25_structured"}:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
    normalized_query = normalize_scalar(query)
    if contains_han(normalized_query):
        return {
            "query": query,
            "expanded_query": normalized_query,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "vector_index_used": False,
            "vector_status": "skipped_unsupported_language",
            "vector_error": "",
            "out_of_scope": False,
            "error": "unsupported_language",
            "results": [],
        }
    expanded_query = expand_query(normalized_query)
    if is_out_of_scope_query(query):
        return {
            "query": query,
            "expanded_query": expanded_query,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "vector_index_used": False,
            "vector_status": "skipped_out_of_scope",
            "vector_error": "",
            "out_of_scope": True,
            "results": [],
        }
    build_root = project_root / "kb_corpus_build"
    bm25_index, docstore = load_bm25(build_root)
    bm25_ranked = ranked_scores(score_bm25(bm25_index, expanded_query))
    vector_status = "disabled"
    vector_error = ""
    if retrieval_mode == "bm25_structured":
        vector_scores = {}
    elif vector_scores_override is not None:
        vector_scores = vector_scores_override
        vector_status = vector_status_override or "ok"
        vector_error = vector_error_override
    elif not vector_runtime_enabled():
        vector_scores = {}
        vector_status = "disabled_runtime_guard"
        vector_error = "Set EM_RAG_ENABLE_VECTOR=1 to run the timeout-isolated vector worker."
    else:
        vector_scores, vector_status, vector_error = collect_vector_scores_with_status(build_root, expanded_query)
    structured_scores = collect_structured_scores(build_root, expanded_query)

    normalized_bm25 = normalize_score_map({chunk_id: data["score"] for chunk_id, data in bm25_ranked.items()})
    normalized_vector = normalize_score_map(vector_scores)
    normalized_structured = normalize_score_map(structured_scores)
    vector_ranks = rank_from_score_map(vector_scores)
    structured_ranks = rank_from_score_map(structured_scores)

    candidate_ids = set(normalized_bm25) | set(normalized_vector) | set(normalized_structured)
    normalized_query_intent = normalize_intent_text(expanded_query)
    suppress_p1411_overall_frequency_noise = is_p1411_overall_frequency_query(normalized_query_intent) and any(
        is_p1411_scope_doc(document_intent_text(docstore[chunk_id]))
        for chunk_id in candidate_ids
        if chunk_id in docstore
    )
    scored: list[tuple[str, float, float, float]] = []
    for chunk_id in candidate_ids:
        if chunk_id not in docstore:
            continue
        rrf = 0.0
        if chunk_id in bm25_ranked:
            rrf += 1.0 / (RRF_K + bm25_ranked[chunk_id]["rank"])
        if chunk_id in vector_ranks:
            rrf += 1.0 / (RRF_K + vector_ranks[chunk_id])
        if chunk_id in structured_ranks:
            rrf += 1.0 / (RRF_K + structured_ranks[chunk_id])
        intent = query_intent_score(expanded_query, docstore[chunk_id])
        scored.append((chunk_id, rrf + intent, rrf, intent))
    scored.sort(key=lambda item: (-item[1], item[0]))
    max_intent = max((intent for _chunk_id, _final, _rrf, intent in scored), default=0.0)
    prune_zero_intent_tail = max_intent >= 6.0

    deduped: list[tuple[str, float, float, float]] = []
    seen_groups: set[str] = set()
    for chunk_id, final, rrf, intent in scored:
        if suppress_p1411_overall_frequency_noise and is_p1411_overall_frequency_noise_doc(
            document_intent_text(docstore[chunk_id])
        ):
            continue
        if prune_zero_intent_tail and intent <= 0.0:
            continue
        group_id = normalize_scalar(docstore[chunk_id].get("duplicate_group_id") or chunk_id)
        if group_id in seen_groups:
            continue
        seen_groups.add(group_id)
        deduped.append((chunk_id, final, rrf, intent))
        if len(deduped) >= top_k:
            break

    results = [
        build_result(
            rank,
            chunk_id,
            docstore[chunk_id],
            expanded_query,
            final,
            normalized_bm25.get(chunk_id, 0.0),
            normalized_vector.get(chunk_id, 0.0),
            normalized_structured.get(chunk_id, 0.0),
            intent,
            rrf,
        )
        for rank, (chunk_id, final, rrf, intent) in enumerate(deduped, start=1)
    ]
    return {
        "query": query,
        "expanded_query": expanded_query,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "vector_index_used": bool(vector_scores),
        "vector_status": vector_status,
        "vector_error": vector_error,
        "out_of_scope": False,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output = retrieve(project_root, args.query, args.top_k, retrieval_mode=args.retrieval_mode)
    trace_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": args.query,
        "top_k": args.top_k,
        "retrieval_mode": args.retrieval_mode,
        "vector_index_used": output["vector_index_used"],
        "vector_status": output.get("vector_status", ""),
        "vector_error": output.get("vector_error", ""),
        "results": output["results"],
    }
    append_jsonl_checked(project_root / "kb_corpus_build" / "rag" / "retrieval_trace.jsonl", trace_row)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
