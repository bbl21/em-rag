#!/usr/bin/env python3
"""Stage-14 retrieval quality evaluation with layered reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("kb_corpus_build/audit/retrieval_quality_evaluation")
EVAL_CASE_DATASET = Path("kb_corpus_build/eval/datasets/retrieval_quality_eval_cases.jsonl")
CODEX_WEB_ASSISTED_REPORT = Path(
    "kb_corpus_build/audit/minimal_human_review_audit_web_assisted/web_assisted_machine_review.md"
)


FIX_DIRECTIONS = {
    "retrieve_failed": "先修 retrieve.py 可运行性或依赖环境。",
    "parse_failed": "修命令输出和 JSON 解析，保证评测输入可信。",
    "no_results_for_answerable": "修召回：query expansion、结构化索引或语料缺口识别。",
    "hard_negative_false_positive": "修 out-of-scope gate / evidence suppression，不能给越界问题硬凑证据。",
    "missing_citation": "修 citation 生成和 docstore 元数据，不优先调排序。",
    "missing_chunk": "修索引/docstore 与 canonical corpus 的一致性。",
    "source_mismatch": "修 source routing、结构化字段重排或 expected source 配置。",
    "intent_facet_mismatch": "修 intent-facet 解析、术语别名和 query expansion。",
    "answerability_gap": "top-5 不能支撑回答，先判断语料缺口，再修召回。",
    "weak_ranking": "top-5 有答案但 rank1 弱，修 intent rerank 和弱证据降权。",
    "citation_support_gap": "citation 不足以支撑核心结论，修 citation selection 或回答阶段 citation planner。",
    "duplicate_evidence": "修 duplicate_group_id 去重或重复证据降权。",
    "preview_gap": "修 query-aware preview 或让评测读取完整 chunk。",
    "multi_citation_required": "回答阶段必须做多 citation 规划，不一定要改 retriever。",
    "off_domain_top_evidence": "修领域过滤和弱证据降权；公式符号相似但学科错误的 top evidence 不能进入前列。",
    "supporting_evidence_pollution": "修 query-aware rerank 和 citation selection；top1 可答但后续证据会污染强制多引用回答。",
    "submodel_scope_risk": "修范围类意图重排；总范围问题不能把子模型频段排在高位造成误导。",
    "path_loss_formula_gap": "修 LoS/NLoS path loss 意图解析；优先召回直接列出 basic transmission loss 公式的 chunk。",
    "metric_confusion_risk": "修传播模型指标识别；path loss 查询要降权 delay spread、angular profile 等非损耗指标。",
}

FACET_ALIASES = {
    "input parameters": [
        "input parameters",
        "relevant parameters",
        "parameters include",
        "where",
        "basic transmission loss exponent",
        "distance between station",
    ],
    "limitations": [
        "limitations",
        "valid for",
        "only valid",
        "recommended for use",
        "may depend",
        "applicable to",
        "frequency range",
        "distance range",
    ],
    "itu-r p.1411": [
        "itu-r p.1411",
        "p.1411",
        "itu_r_p1411_13",
        "recommendation itu-r p.1411",
    ],
    "microcell": [
        "microcell",
        "micro-cell",
        "micro-cellular",
        "micro cellular",
        "microcells",
        "micro-cells",
    ],
    "los": [
        "los",
        "line-of-sight",
        "line of sight",
        "llos",
    ],
    "nlos": [
        "nlos",
        "non-line-of-sight",
        "non line of sight",
        "lnlos",
    ],
    "path loss": [
        "path loss",
        "propagation loss",
        "basic transmission loss",
        "transmission loss",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality using layered stage-14 reports.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k evidence rows to retrieve per query.")
    parser.add_argument(
        "--cases",
        default="",
        help="JSONL eval cases. Defaults to the active English-only dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR.as_posix(),
        help="Output directory, relative to project root unless absolute.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "bm25_structured"],
        default="bm25_structured",
        help="Use full hybrid retrieval or disable vector scoring for offline quality evaluation.",
    )
    parser.add_argument(
        "--reuse-retrieval-results",
        default="",
        help="Optional rag_retrieval_results.jsonl to score without rerunning retrieve.py.",
    )
    return parser.parse_args()


def normalize_scalar(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_match_text(value: Any) -> str:
    text = normalize_scalar(value).lower().replace("‑", "-").replace("–", "-")
    text = re.sub(r"\b([syz])\s*_\s*\{?\s*([0-9]{1,2})\s*\}?", r"\1\2", text)
    text = text.replace("γ", "gamma").replace("Γ", "gamma")
    return text


def md_cell(value: Any) -> str:
    return normalize_scalar(value).replace("|", "\\|")


def percent(part: int, whole: int) -> str:
    return "0.00%" if whole <= 0 else f"{part / whole * 100:.2f}%"


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
    if path.read_text(encoding="utf-8") != text:
        raise ValueError(f"UTF-8 roundtrip mismatch for {path.as_posix()}")


def write_jsonl_checked(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_text_checked(path, text)


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def detect_web_reference_status(project_root: Path, output_root: Path) -> str:
    web_report = project_root / CODEX_WEB_ASSISTED_REPORT
    if web_report.is_file():
        return "codex_web_spotcheck_limited"
    existing_local_report = output_root / "web_assisted_consistency_check.md"
    if existing_local_report.is_file():
        text = existing_local_report.read_text(encoding="utf-8")
        if "codex_web_spotcheck_limited" in text:
            return "codex_web_spotcheck_limited"
    return "not_run_by_local_script"


def infer_expected_intent(row: dict[str, Any]) -> str:
    query = normalize_match_text(row.get("query"))
    category = normalize_scalar(row.get("category"))
    if category == "hard_negative" or row.get("should_have_direct_answer") is False:
        return "out_of_scope"
    if "frequency range" in query:
        return "frequency_range"
    if "input parameter" in query:
        return "input_parameters"
    if "limitation" in query:
        return "limitations"
    if "scenario" in query or "street canyon" in query or "rooftop" in query:
        return "scenario"
    if "formula" in query or "equation" in query:
        return "formula"
    if "relationship" in query or "relation" in query:
        return "relation"
    return "definition"


def requires_multi_citation(row: dict[str, Any]) -> bool:
    query = normalize_match_text(row.get("query"))
    padded = f" {query} "
    if " los " in padded and " nlos" in padded:
        return True
    if "s11" in query and "s21" in query:
        return True
    if "directivity" in query and ("gain" in query or "effective aperture" in query):
        return True
    explicit_markers = [" and ", "relationship between", "compare", "difference between"]
    return any(marker in padded for marker in explicit_markers)


def case_from_legacy_row(row: dict[str, Any]) -> dict[str, Any]:
    is_hard_negative = row.get("should_have_direct_answer") is False or row.get("category") == "hard_negative"
    converted = {
        "query_id": row["query_id"],
        "query": row["query"],
        "language": row["language"],
        "category": row.get("category", "unknown"),
        "expected_intent": row.get("expected_intent") or infer_expected_intent(row),
        "expected_sources": row.get("expected_sources") or row.get("expected_source_hint") or [],
        "expected_evidence_facets": row.get("expected_evidence_facets") or row.get("expected_key_terms") or [],
        "is_hard_negative": bool(row.get("is_hard_negative", is_hard_negative)),
        "requires_web_check": bool(row.get("requires_web_check", False)),
        "requires_multi_citation": bool(row.get("requires_multi_citation", requires_multi_citation(row))),
        "notes": row.get("notes", ""),
    }
    return converted


def default_cases_path(project_root: Path) -> Path:
    return (project_root / EVAL_CASE_DATASET).resolve()


def load_eval_cases(project_root: Path, cases_arg: str) -> list[dict[str, Any]]:
    path = resolve_project_path(project_root, cases_arg) if cases_arg else default_cases_path(project_root)
    rows = load_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"No eval cases found at {path.as_posix()}")
    for row in rows:
        language = normalize_scalar(row.get("language")).lower()
        if language != "en":
            query_id = normalize_scalar(row.get("query_id")) or "<unknown>"
            raise ValueError(f"Eval case {query_id} language must be 'en'; got {language or '<missing>'}")
    return [case_from_legacy_row(row) for row in rows]


def canonical_maps(project_root: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    rows = load_jsonl(project_root / "kb_corpus_build" / "corpus" / "chunks.canonical.jsonl")
    by_id = {row["chunk_id"]: row for row in rows if row.get("chunk_id")}
    return by_id, set(by_id)


def configure_retrieve_module(retrieve_module: Any, retrieval_mode: str) -> None:
    if retrieval_mode == "bm25_structured":
        retrieve_module.collect_vector_scores = lambda build_root, query: {}


def hybrid_vector_overrides(
    retrieve_module: Any,
    project_root: Path,
    cases: list[dict[str, Any]],
    output_root: Path,
    retrieval_mode: str,
) -> tuple[dict[str, dict[str, float]], str, str] | None:
    if retrieval_mode != "hybrid" or not retrieve_module.vector_runtime_enabled():
        return None

    preflight_status, preflight_error = retrieve_module.run_vector_runtime_preflight(
        project_root,
        output_root / "vector_runtime_preflight",
    )
    if preflight_status != "ok":
        return {}, f"preflight_{preflight_status}", preflight_error

    query_by_id = {
        normalize_scalar(case["query_id"]): retrieve_module.expand_query(case["query"])
        for case in cases
        if normalize_scalar(case.get("query_id"))
    }
    batch_scores, batch_status, batch_error = retrieve_module.collect_batch_vector_scores_with_status(
        project_root / "kb_corpus_build",
        query_by_id,
    )
    if batch_status != "ok":
        return {}, f"batch_{batch_status}", batch_error
    return batch_scores, "ok", ""


def enrich_results(results: list[dict[str, Any]], canonical_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in results:
        chunk_id = normalize_scalar(row.get("chunk_id"))
        canonical = canonical_by_id.get(chunk_id, {})
        item = dict(row)
        item["judge_text"] = normalize_scalar(canonical.get("content_md") or canonical.get("retrieval_text"))[:6000]
        item["duplicate_group_id"] = normalize_scalar(canonical.get("duplicate_group_id") or chunk_id)
        item["chunk_id_in_canonical"] = chunk_id in canonical_by_id
        if "tex_file" not in item:
            item["tex_file"] = canonical.get("tex_file")
        enriched.append(item)
    return enriched


def run_retrieval(
    project_root: Path,
    cases: list[dict[str, Any]],
    top_k: int,
    output_root: Path,
    retrieval_mode: str,
) -> list[dict[str, Any]]:
    scripts_dir = project_root / "kb_corpus_build" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import retrieve as retrieve_module

    configure_retrieve_module(retrieve_module, retrieval_mode)
    canonical_by_id, _canonical_ids = canonical_maps(project_root)
    raw_dir = output_root / "retrieval_raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vector_overrides = hybrid_vector_overrides(retrieve_module, project_root, cases, output_root, retrieval_mode)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['query_id']} {case['query']}", flush=True)
        started = time.perf_counter()
        exit_code = 0
        parse_failed = False
        error = ""
        output: dict[str, Any] = {}
        try:
            retrieve_kwargs: dict[str, Any] = {}
            if vector_overrides is not None:
                scores_by_query, vector_status, vector_error = vector_overrides
                retrieve_kwargs = {
                    "vector_scores_override": scores_by_query.get(case["query_id"], {}),
                    "vector_status_override": vector_status,
                    "vector_error_override": vector_error,
                }
            output = retrieve_module.retrieve(
                project_root,
                case["query"],
                top_k,
                retrieval_mode=retrieval_mode,
                **retrieve_kwargs,
            )
            output["results"] = enrich_results(output.get("results") or [], canonical_by_id)
        except Exception as exc:  # pragma: no cover - real environment failures only
            exit_code = 1
            parse_failed = True
            error = f"{type(exc).__name__}: {exc}"
            output = {"query": case["query"], "top_k": top_k, "results": []}
        runtime = round(time.perf_counter() - started, 3)
        write_text_checked(
            raw_dir / f"{case['query_id']}.stdout.json",
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        )
        write_text_checked(raw_dir / f"{case['query_id']}.stderr.txt", error + ("\n" if error else ""))
        rows.append(
            {
                "query_id": case["query_id"],
                "query": case["query"],
                "category": case["category"],
                "exit_code": exit_code,
                "parse_failed": parse_failed,
                "runtime_seconds": runtime,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
                "vector_index_used": bool(output.get("vector_index_used")),
                "vector_status": normalize_scalar(output.get("vector_status")),
                "vector_error": normalize_scalar(output.get("vector_error")),
                "out_of_scope": bool(output.get("out_of_scope")),
                "results": output.get("results") or [],
            }
        )
    return rows


def result_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("chunk_id"),
        row.get("source_id"),
        row.get("source_title"),
        row.get("chapter"),
        row.get("section"),
        row.get("citation"),
        row.get("content_preview"),
        row.get("judge_text"),
    ]
    return normalize_match_text(" ".join(normalize_scalar(part) for part in parts))


def tokens_for(term: str) -> list[str]:
    normalized = normalize_match_text(term).replace("-", " ")
    return [token for token in re.findall(r"[a-z0-9]+", normalized) if len(token) > 1]


def text_matches_term(text: str, term: str) -> bool:
    normalized = normalize_match_text(term)
    aliases = FACET_ALIASES.get(normalized, [term])
    for alias in aliases:
        alias_normalized = normalize_match_text(alias)
        if not alias_normalized:
            continue
        if alias_normalized in text or alias_normalized.replace("-", " ") in text:
            return True
        tokens = tokens_for(alias)
        if tokens and all(token in text for token in tokens):
            return True
    return False


def matched_facets(case: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
    text = "\n".join(result_text(row) for row in results)
    facets = case.get("expected_evidence_facets") or []
    return [facet for facet in facets if text_matches_term(text, facet)]


def matched_sources(case: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
    sources = case.get("expected_sources") or []
    found: list[str] = []
    source_text = " ".join(normalize_match_text(row.get("source_id")) for row in results)
    for source in sources:
        if normalize_match_text(source) in source_text:
            found.append(source)
    return found


def citation_complete(results: list[dict[str, Any]]) -> bool:
    if not results:
        return False
    for row in results:
        citation = normalize_scalar(row.get("citation"))
        if not citation or "|" not in citation or not normalize_scalar(row.get("source_id")):
            return False
    return True


def direct_evidence_rank(case: dict[str, Any], results: list[dict[str, Any]]) -> int | None:
    sources = case.get("expected_sources") or []
    for index, row in enumerate(results[:5], start=1):
        facet_ok = result_supports_case_facets(case, row)
        source_ok = True
        if sources:
            source = normalize_match_text(row.get("source_id"))
            source_ok = any(normalize_match_text(expected) in source for expected in sources)
        if facet_ok and source_ok:
            return index
    return None


def duplicate_problem(results: list[dict[str, Any]]) -> bool:
    groups = [normalize_scalar(row.get("duplicate_group_id") or row.get("chunk_id")) for row in results if row.get("chunk_id")]
    return len(groups) != len(set(groups))


def preview_problem(results: list[dict[str, Any]]) -> bool:
    if not results:
        return True
    return any(len(normalize_scalar(row.get("content_preview"))) < 30 for row in results[:3])


def result_supports_case_facets(case: dict[str, Any], row: dict[str, Any]) -> bool:
    query = normalize_match_text(case.get("query"))
    text = result_text(row)
    if "radiation efficiency" in query:
        if "antenna efficiency" in text or "radiation efficiency" in text:
            return True
        if "radiated power" in text and ("input power" in text or "p_{in}" in text or "p_in" in text):
            return True
        if "p_{t}" in text and ("p_{in}" in text or "p_in" in text):
            return True
        return False
    facets = case.get("expected_evidence_facets") or []
    if not facets:
        return bool(text)
    return all(text_matches_term(text, facet) for facet in facets)


def off_domain_top_evidence_problem(case: dict[str, Any], results: list[dict[str, Any]]) -> bool:
    query = normalize_match_text(case.get("query"))
    if "acoustic" in query or "acoustics" in query:
        return False
    rf_standing_wave_query = any(
        marker in query
        for marker in [
            "vswr",
            "standing wave ratio",
            "reflection coefficient",
            "scattering parameter",
            "s11",
            "s21",
        ]
    )
    if not rf_standing_wave_query:
        return False
    for row in results[:3]:
        text = result_text(row)
        if "acoustic waveguide" in text or "chapter 13: acoustics" in text or "chapter 13 acoustics" in text:
            return True
    return False


def supporting_evidence_pollution_problem(case: dict[str, Any], results: list[dict[str, Any]], direct_rank: int | None) -> bool:
    if direct_rank != 1 or len(results) < 3:
        return False
    weak_support_count = sum(1 for row in results[1:3] if not result_supports_case_facets(case, row))
    return weak_support_count >= 2


def p1411_submodel_scope_risk(case: dict[str, Any], results: list[dict[str, Any]]) -> bool:
    query = normalize_match_text(case.get("query"))
    frequency_range_query = "frequency range" in query
    if "p.1411" not in query or not frequency_range_query:
        return False
    top1_text = result_text(results[0]) if results else ""
    if "300 mhz to 300 ghz" not in top1_text:
        return False
    submodel_patterns = [
        r"\b2\s*(?:to|-)\s*38\s*ghz\b",
        r"\b800\s*(?:to|-)\s*2\s*000\s*mhz\b",
        r"\b800\s*(?:to|-)\s*2000\s*mhz\b",
        r"\b26\s*ghz\b",
        r"\b38\s*ghz\b",
    ]
    for row in results[1:5]:
        text = result_text(row)
        if any(re.search(pattern, text) for pattern in submodel_patterns) and "300 mhz to 300 ghz" not in text:
            return True
    return False


def is_path_loss_query(query: str) -> bool:
    return "path loss" in query


def p1411_path_loss_formula_gap(case: dict[str, Any], results: list[dict[str, Any]]) -> bool:
    query = normalize_match_text(case.get("query"))
    if not is_path_loss_query(query):
        return False
    top_text = "\n".join(result_text(row) for row in results[:3])
    if "los" in query and "nlos" not in query:
        return not any(marker in top_text for marker in ["llos", "l_los", "basic transmission loss exponent"])
    if "nlos" in query:
        return not any(marker in top_text for marker in ["lnlos", "l_nlos", "diffraction loss", "reflection loss"])
    return False


def p1411_metric_confusion_risk(case: dict[str, Any], results: list[dict[str, Any]]) -> bool:
    query = normalize_match_text(case.get("query"))
    if not is_path_loss_query(query):
        return False
    confusion_markers = [
        "delay spread",
        "angular profile",
        "arrival angle",
        "multipath delay",
        "path morphology",
    ]
    for row in results[:3]:
        text = result_text(row)
        if any(marker in text for marker in confusion_markers):
            return True
    return False


def mechanical_issues(case: dict[str, Any], retrieval: dict[str, Any], canonical_ids: set[str]) -> list[str]:
    issues: list[str] = []
    results = retrieval.get("results") or []
    if retrieval.get("exit_code") != 0:
        issues.append("retrieve_failed")
    if retrieval.get("parse_failed"):
        issues.append("parse_failed")
    if case.get("is_hard_negative"):
        if results:
            issues.append("hard_negative_false_positive")
    elif not results:
        issues.append("no_results_for_answerable")
    for row in results[:5]:
        if not normalize_scalar(row.get("citation")) or not normalize_scalar(row.get("source_id")):
            issues.append("missing_citation")
        chunk_id = normalize_scalar(row.get("chunk_id"))
        if chunk_id and chunk_id not in canonical_ids:
            issues.append("missing_chunk")
    if duplicate_problem(results):
        issues.append("duplicate_evidence")
    return sorted(set(issues))


def vector_status_fields(retrieval: dict[str, Any]) -> dict[str, Any]:
    return {
        "vector_index_used": bool(retrieval.get("vector_index_used")),
        "vector_status": normalize_scalar(retrieval.get("vector_status")),
        "vector_error": normalize_scalar(retrieval.get("vector_error")),
    }


def score_retrieval_case(case: dict[str, Any], retrieval: dict[str, Any], canonical_ids: set[str]) -> dict[str, Any]:
    results = retrieval.get("results") or []
    top5 = results[:5]
    issues = mechanical_issues(case, retrieval, canonical_ids)
    vector_fields = vector_status_fields(retrieval)
    if case.get("is_hard_negative"):
        if results:
            return {
                **case,
                "retrieval_mode": retrieval.get("retrieval_mode", ""),
                **vector_fields,
                "quality_label": "NEEDS_FIX",
                "quality_score": 25,
                "score_breakdown": {
                    "answerability_at_5": 0,
                    "directness_rank": 0,
                    "citation_support": 0,
                    "intent_facet_match": 0,
                    "negative_safety": 0,
                    "diversity_preview": 5,
                },
                "mechanical_issues": issues,
                "failure_types": sorted(set(issues + ["hard_negative_false_positive"])),
                "matched_evidence_facets": [],
                "matched_sources": [],
                "direct_evidence_rank": None,
                "top_chunk_ids": [row.get("chunk_id") for row in top5],
                "needs_human_spotcheck": True,
                "judgment_reason": "hard negative returned evidence; this is a false positive.",
            }
        return {
            **case,
            "retrieval_mode": retrieval.get("retrieval_mode", ""),
            **vector_fields,
            "quality_label": "PASS",
            "quality_score": 100,
            "score_breakdown": {
                "answerability_at_5": 30,
                "directness_rank": 20,
                "citation_support": 15,
                "intent_facet_match": 15,
                "negative_safety": 10,
                "diversity_preview": 10,
            },
            "mechanical_issues": [],
            "failure_types": [],
            "matched_evidence_facets": [],
            "matched_sources": [],
            "direct_evidence_rank": None,
            "top_chunk_ids": [],
            "needs_human_spotcheck": False,
            "judgment_reason": "hard negative returned no evidence.",
        }

    facets = case.get("expected_evidence_facets") or []
    sources = case.get("expected_sources") or []
    found_facets = matched_facets(case, top5)
    found_sources = matched_sources(case, top5)
    direct_rank = direct_evidence_rank(case, top5)
    facets_ok = len(found_facets) >= len(facets) if facets else bool(top5)
    sources_ok = bool(found_sources) if sources else True
    answerable = bool(top5) and facets_ok and sources_ok
    citations_ok = citation_complete(top5)
    duplicate_bad = duplicate_problem(top5)
    preview_bad = preview_problem(top5)
    off_domain_bad = off_domain_top_evidence_problem(case, top5)
    supporting_pollution_bad = supporting_evidence_pollution_problem(case, top5, direct_rank)
    submodel_scope_bad = p1411_submodel_scope_risk(case, top5)
    path_loss_formula_bad = p1411_path_loss_formula_gap(case, top5)
    metric_confusion_bad = p1411_metric_confusion_risk(case, top5)

    answerability_score = 30 if answerable else (15 if found_facets and sources_ok else 0)
    if direct_rank == 1:
        directness_score = 20
    elif direct_rank and direct_rank <= 3:
        directness_score = 14
    elif direct_rank and direct_rank <= 5:
        directness_score = 8
    elif answerable:
        directness_score = 4
    else:
        directness_score = 0
    citation_score = 15 if citations_ok and answerable else (8 if citations_ok else 0)
    if facets and len(found_facets) == len(facets):
        intent_score = 15
    elif facets and found_facets:
        intent_score = 8
    elif not facets and top5:
        intent_score = 10
    else:
        intent_score = 0
    negative_score = 10
    diversity_score = 10
    if duplicate_bad:
        diversity_score -= 5
    if preview_bad:
        diversity_score -= 5
    diversity_score = max(0, diversity_score)
    score = answerability_score + directness_score + citation_score + intent_score + negative_score + diversity_score

    failure_types: list[str] = list(issues)
    if not sources_ok:
        failure_types.append("source_mismatch")
    if facets and not facets_ok:
        failure_types.append("intent_facet_mismatch")
    if not answerable:
        failure_types.append("answerability_gap")
    if answerable and direct_rank != 1:
        failure_types.append("weak_ranking")
    if not citations_ok:
        failure_types.append("citation_support_gap")
    if duplicate_bad:
        failure_types.append("duplicate_evidence")
    if preview_bad:
        failure_types.append("preview_gap")
    if case.get("requires_multi_citation") and answerable:
        failure_types.append("multi_citation_required")
    if off_domain_bad:
        failure_types.append("off_domain_top_evidence")
    if supporting_pollution_bad:
        failure_types.append("supporting_evidence_pollution")
    if submodel_scope_bad:
        failure_types.append("submodel_scope_risk")
    if path_loss_formula_bad:
        failure_types.append("path_loss_formula_gap")
    if metric_confusion_bad:
        failure_types.append("metric_confusion_risk")

    if off_domain_bad or path_loss_formula_bad:
        label = "NEEDS_FIX"
    elif supporting_pollution_bad or submodel_scope_bad or metric_confusion_bad:
        label = "WEAK_PASS" if answerable else "UNSURE"
    elif score >= 80 and direct_rank == 1 and not issues:
        label = "PASS"
    elif answerable and score >= 60:
        label = "WEAK_PASS"
    elif score >= 60:
        label = "UNSURE"
    else:
        label = "NEEDS_FIX"

    needs_human = label in {"UNSURE", "NEEDS_FIX"} or "multi_citation_required" in failure_types
    reason_parts = [
        f"score={score}",
        f"answerable={answerable}",
        f"direct_rank={direct_rank or '<none>'}",
        f"facets={len(found_facets)}/{len(facets)}",
        f"sources={','.join(found_sources) or '<none>'}",
    ]
    if failure_types:
        reason_parts.append("failure_types=" + ", ".join(sorted(set(failure_types))))
    return {
        **case,
        "retrieval_mode": retrieval.get("retrieval_mode", ""),
        **vector_fields,
        "quality_label": label,
        "quality_score": score,
        "score_breakdown": {
            "answerability_at_5": answerability_score,
            "directness_rank": directness_score,
            "citation_support": citation_score,
            "intent_facet_match": intent_score,
            "negative_safety": negative_score,
            "diversity_preview": diversity_score,
        },
        "mechanical_issues": sorted(set(issues)),
        "failure_types": sorted(set(failure_types)),
        "matched_evidence_facets": found_facets,
        "matched_sources": found_sources,
        "direct_evidence_rank": direct_rank,
        "top_chunk_ids": [row.get("chunk_id") for row in top5],
        "needs_human_spotcheck": needs_human,
        "judgment_reason": "; ".join(reason_parts),
    }


def top_evidence_lines(retrieval: dict[str, Any], limit: int = 3) -> list[str]:
    lines: list[str] = []
    for row in (retrieval.get("results") or [])[:limit]:
        preview = md_cell(row.get("content_preview"))[:300].rstrip()
        lines.append(
            f"- rank {row.get('rank')}: `{md_cell(row.get('chunk_id'))}` / "
            f"`{md_cell(row.get('source_id'))}` / {md_cell(row.get('citation'))} / {preview}"
        )
    return lines or ["- <无证据>"]


def build_quality_scoring_spec() -> str:
    return "\n".join(
        [
            "# 检索质量评分标准",
            "",
            "本文件定义第十四阶段 `retrieval_quality_evaluation` 的评分口径。机械硬门槛和语义质量评分分开记录；脚本评分是规则辅助初稿，不等同人工 qrels、真实 Recall/MRR/Accuracy 或最终 LLM judge。",
            "",
            "## 硬门槛",
            "",
            "- JSON 可解析，检索命令成功。",
            "- 非 hard negative 问题应返回 evidence。",
            "- hard negative / out-of-scope 问题不应返回 evidence。",
            "- top-k evidence 的 `source_id`、`chunk_id`、`citation` 字段完整。",
            "- 返回 chunk 能在 canonical corpus 中找到。",
            "- duplicate group 不应在 top-k 中重复。",
            "",
            "## 100 分评分",
            "",
            "| 维度 | 权重 | 含义 |",
            "|---|---:|---|",
            "| answerability@5 | 30 | top-5 是否包含足够证据回答问题 |",
            "| directness / rank | 20 | 最直接证据是否排在 rank1；rank2-5 会降级 |",
            "| citation support | 15 | citation 指向内容是否支撑核心结论 |",
            "| intent-facet match | 15 | 是否命中定义、公式、输入参数、限制、场景、频段、关系等查询意图 |",
            "| negative safety | 10 | hard negative / out-of-scope 是否拒答 |",
            "| diversity / preview quality | 10 | 是否重复、是否只有弱 preview |",
            "",
            "## 标签",
            "",
            "- `PASS`：无硬门槛失败，分数 >= 80，rank1 是直接证据。",
            "- `WEAK_PASS`：top-5 能回答，但 rank1 不够直接、证据分散、存在弱相关污染、总范围混入子模型范围或需要多 citation。",
            "- `NEEDS_FIX`：top-5 不能回答、citation 不支撑、hard negative 返回证据或分数 < 60。",
            "- `UNSURE`：规则辅助评分不能区分语料缺口和检索失败，需 agent 或人工抽查。",
            "",
            "## 高风险失败归因",
            "",
            "- `off_domain_top_evidence`：top evidence 中出现跨学科硬负例，例如 RF/VSWR 查询召回 Acoustics 章节。",
            "- `supporting_evidence_pollution`：rank1 可回答，但 rank2/3 不支撑同一查询意图，强制多引用会污染回答。",
            "- `submodel_scope_risk`：总范围查询召回了子模型频段，可能诱导模型答错范围。",
            "- `path_loss_formula_gap`：LoS/NLoS path loss 查询没有在 top evidence 中召回直接损耗公式。",
            "- `metric_confusion_risk`：path loss 查询混入 delay spread、angular profile 等非损耗指标。",
            "",
        ]
    )


def build_mechanical_integrity_audit(cases: list[dict[str, Any]], judgments: list[dict[str, Any]]) -> str:
    issue_counts: Counter[str] = Counter()
    for row in judgments:
        issue_counts.update(row.get("mechanical_issues") or [])
    lines = [
        "# 机械完整性检查",
        "",
        "本报告只记录确定性硬门槛，不判断 evidence 是否真正能回答问题。",
        "",
        f"- 总问题数：{len(cases)}",
        f"- 存在机械问题的问题数：{sum(1 for row in judgments if row.get('mechanical_issues'))}",
        "",
        "## 机械问题统计",
        "",
        "| issue | count |",
        "|---|---:|",
    ]
    if issue_counts:
        for issue, count in sorted(issue_counts.items()):
            lines.append(f"| `{issue}` | {count} |")
    else:
        lines.append("| `<none>` | 0 |")
    lines.extend(["", "## 问题明细", "", "| query_id | query | mechanical_issues |", "|---|---|---|"])
    for row in judgments:
        issues = ", ".join(f"`{item}`" for item in row.get("mechanical_issues") or []) or "<none>"
        lines.append(f"| {row['query_id']} | {md_cell(row['query'])} | {issues} |")
    lines.append("")
    return "\n".join(lines)


def build_source_grounded_quality_judgments(
    judgments: list[dict[str, Any]], retrieval_by_id: dict[str, dict[str, Any]]
) -> str:
    counts = Counter(row["quality_label"] for row in judgments)
    lines = [
        "# 本地语料溯源检索质量判断",
        "",
        "本报告是规则辅助的 source-grounded quality judgment 初稿：它读取 top-k、本地 chunk 文本和 citation 字段做评分。需要联网或主观语义判断的地方，仍由 Codex/agent 在本报告基础上复核。",
        "",
        f"- PASS：{counts['PASS']}",
        f"- WEAK_PASS：{counts['WEAK_PASS']}",
        f"- NEEDS_FIX：{counts['NEEDS_FIX']}",
        f"- UNSURE：{counts['UNSURE']}",
        "",
        "| query_id | label | score | direct_rank | failure_types | reason |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in judgments:
        failures = ", ".join(f"`{item}`" for item in row.get("failure_types") or []) or "<none>"
        direct = row.get("direct_evidence_rank") or ""
        lines.append(
            f"| {row['query_id']} | `{row['quality_label']}` | {row['quality_score']} | {direct} | "
            f"{failures} | {md_cell(row.get('judgment_reason'))} |"
        )
    lines.extend(["", "## Top Evidence 摘要", ""])
    for row in judgments:
        if row["quality_label"] == "PASS" and not row.get("needs_human_spotcheck"):
            continue
        retrieval = retrieval_by_id.get(row["query_id"], {})
        lines.extend(
            [
                f"### {row['query_id']} - {md_cell(row['query'])}",
                "",
                f"- label：`{row['quality_label']}`",
                f"- score：{row['quality_score']}",
                f"- failure_types：{', '.join(row.get('failure_types') or []) or '<none>'}",
                *top_evidence_lines(retrieval, 5),
                "",
            ]
        )
    return "\n".join(lines)


def build_web_assisted_consistency_check(judgments: list[dict[str, Any]], web_reference_status: str) -> str:
    needs_web = [row for row in judgments if row.get("requires_web_check")]
    lines = [
        "# 联网一致性辅助检查",
        "",
        f"- web_reference_status：`{web_reference_status}`",
        "- 本地脚本不联网；联网只能由 Codex 在审查时辅助执行。",
        "- 联网结果不能替代本地 citation，也不能证明知识库已经召回对应证据。",
    ]
    if web_reference_status == "codex_web_spotcheck_limited":
        lines.append(f"- Codex 联网辅助报告：`{CODEX_WEB_ASSISTED_REPORT.as_posix()}`")
    lines.extend(
        [
            "",
            "## 需要联网辅助的样例",
            "",
            "| query_id | query | label | reason |",
            "|---|---|---|---|",
        ]
    )
    if needs_web:
        for row in needs_web:
            lines.append(f"| {row['query_id']} | {md_cell(row['query'])} | `{row['quality_label']}` | {md_cell(row.get('judgment_reason'))} |")
    else:
        lines.append("| <none> | <none> | <none> | 当前 eval cases 未显式要求联网辅助。 |")
    lines.append("")
    return "\n".join(lines)


def select_spotcheck_samples(judgments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    quotas = [("PASS", 3), ("WEAK_PASS", 3), ("NEEDS_FIX", 3), ("UNSURE", 3)]
    for label, quota in quotas:
        rows = sorted((row for row in judgments if row["quality_label"] == label), key=lambda row: (row["category"], row["query_id"]))
        for row in rows[:quota]:
            selected[row["query_id"]] = row
    high_risk = [
        row
        for row in judgments
        if row.get("is_hard_negative")
        or row.get("requires_multi_citation")
        or row.get("requires_web_check")
        or row["quality_label"] in {"NEEDS_FIX", "UNSURE"}
    ]
    for row in sorted(high_risk, key=lambda item: (item["quality_label"], item["query_id"]))[:4]:
        selected.setdefault(row["query_id"], row)
    return list(selected.values())[:12]


def build_human_spotcheck_samples(
    judgments: list[dict[str, Any]], retrieval_by_id: dict[str, dict[str, Any]]
) -> str:
    samples = select_spotcheck_samples(judgments)
    lines = [
        "# 人工随机抽查样例",
        "",
        "人工审查是抽查和仲裁，不是批量主审。样例按固定规则选取，覆盖 PASS、WEAK_PASS、NEEDS_FIX、UNSURE 和高风险边界。",
        "",
    ]
    for row in samples:
        retrieval = retrieval_by_id.get(row["query_id"], {})
        lines.extend(
            [
                f"## {row['query_id']} - {md_cell(row['query'])}",
                "",
                f"- 机器标签：`{row['quality_label']}`",
                f"- 分数：{row['quality_score']}",
                f"- 失败归因：{', '.join(row.get('failure_types') or []) or '<none>'}",
                "Top evidence：",
                *top_evidence_lines(retrieval, 3),
                "人工抽查：",
                "[] top-k 是否真的能回答这个问题？",
                "[] citation 指向的内容是否支撑核心结论？",
                "[] 是否存在弱相关、误召回或 hard negative 硬凑证据？",
                "人工结论：",
                "[] 通过",
                "[] 需要修复",
                "[] 不确定",
                "",
            ]
        )
    return "\n".join(lines)


def build_failure_taxonomy_and_fix_plan(judgments: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        for failure in row.get("failure_types") or []:
            grouped[failure].append(row)
    lines = [
        "# 失败归因与修复计划",
        "",
        "本文件按失败类型聚合，用来决定后续是否需要改 retriever、answer prompt、citation planner 或上游语料。",
        "",
    ]
    if not grouped:
        lines.extend(["当前没有失败归因项。", ""])
        return "\n".join(lines)
    for failure in sorted(grouped):
        rows = grouped[failure]
        lines.extend(
            [
                f"## `{failure}`",
                "",
                f"- 数量：{len(rows)}",
                f"- 修复方向：{FIX_DIRECTIONS.get(failure, '先人工判定根因，再决定修复位置。')}",
                "",
                "| query_id | label | score | query |",
                "|---|---|---:|---|",
            ]
        )
        for row in rows[:20]:
            lines.append(f"| {row['query_id']} | `{row['quality_label']}` | {row['quality_score']} | {md_cell(row['query'])} |")
        lines.append("")
    return "\n".join(lines)


def build_quality_gate_summary(judgments: list[dict[str, Any]], retrieval_mode: str, web_reference_status: str) -> str:
    counts = Counter(row["quality_label"] for row in judgments)
    hybrid_vector_unavailable = [
        row
        for row in judgments
        if retrieval_mode == "hybrid"
        and not row.get("is_hard_negative")
        and (not row.get("vector_index_used") or normalize_scalar(row.get("vector_status")) != "ok")
    ]
    vector_status_counts = Counter(normalize_scalar(row.get("vector_status")) or "<blank>" for row in hybrid_vector_unavailable)
    if counts["NEEDS_FIX"]:
        gate = "NEEDS_RETRIEVAL_FIX"
    elif counts["UNSURE"]:
        gate = "NEEDS_AGENT_OR_HUMAN_ADJUDICATION"
    elif hybrid_vector_unavailable:
        gate = "HYBRID_VECTOR_UNAVAILABLE"
    elif retrieval_mode != "hybrid":
        gate = "BM25_STRUCTURED_PASS_NEEDS_HYBRID_VALIDATION"
    else:
        gate = "PASS"
    lines = [
        "# 检索质量门禁摘要",
        "",
        f"- quality_gate：`{gate}`",
        f"- retrieval_mode：`{retrieval_mode}`",
        f"- web_reference_status：`{web_reference_status}`",
        f"- hybrid_vector_unavailable_cases：{len(hybrid_vector_unavailable)}",
        f"- 总问题数：{len(judgments)}",
        f"- PASS：{counts['PASS']} ({percent(counts['PASS'], len(judgments))})",
        f"- WEAK_PASS：{counts['WEAK_PASS']} ({percent(counts['WEAK_PASS'], len(judgments))})",
        f"- NEEDS_FIX：{counts['NEEDS_FIX']} ({percent(counts['NEEDS_FIX'], len(judgments))})",
        f"- UNSURE：{counts['UNSURE']} ({percent(counts['UNSURE'], len(judgments))})",
        "",
        "## 结论",
        "",
    ]
    if vector_status_counts:
        status_text = ", ".join(f"`{status}`={count}" for status, count in sorted(vector_status_counts.items()))
        lines.insert(11, f"- vector_status_counts：{status_text}")
    if gate == "NEEDS_RETRIEVAL_FIX":
        lines.append("本轮存在明确修复项；先看 `failure_taxonomy_and_fix_plan.md`，再决定是否修改 retriever。")
    elif gate == "NEEDS_AGENT_OR_HUMAN_ADJUDICATION":
        lines.append("本轮没有明确失败，但存在不确定项；先由 Codex 做 source-grounded 复核，必要时给用户抽查。")
    elif gate == "HYBRID_VECTOR_UNAVAILABLE":
        lines.append("本轮请求了 `hybrid`，但至少一个可回答样例没有实际使用 vector 分数；只能视为退化检索结果，不能宣布完整 Hybrid/vector 通过。")
    elif gate == "BM25_STRUCTURED_PASS_NEEDS_HYBRID_VALIDATION":
        lines.append("本轮 `bm25_structured` 通过，但不能代表完整 Hybrid/vector 质量通过。")
    else:
        lines.append("本轮完整 Hybrid/vector 评测通过。")
    lines.append("")
    return "\n".join(lines)


def write_quality_outputs(
    output_root: Path,
    cases: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    retrieval_mode: str,
    web_reference_status: str,
) -> None:
    retrieval_by_id = {row["query_id"]: row for row in retrievals}
    write_jsonl_checked(output_root / "retrieval_quality_eval_cases.jsonl", cases)
    write_jsonl_checked(output_root / "rag_retrieval_results.jsonl", retrievals)
    write_jsonl_checked(output_root / "quality_judgments.jsonl", judgments)
    write_text_checked(output_root / "quality_scoring_spec.md", build_quality_scoring_spec())
    write_text_checked(output_root / "mechanical_integrity_audit.md", build_mechanical_integrity_audit(cases, judgments))
    write_text_checked(
        output_root / "source_grounded_quality_judgments.md",
        build_source_grounded_quality_judgments(judgments, retrieval_by_id),
    )
    write_text_checked(
        output_root / "web_assisted_consistency_check.md",
        build_web_assisted_consistency_check(judgments, web_reference_status),
    )
    write_text_checked(output_root / "human_spotcheck_samples.md", build_human_spotcheck_samples(judgments, retrieval_by_id))
    write_text_checked(output_root / "failure_taxonomy_and_fix_plan.md", build_failure_taxonomy_and_fix_plan(judgments))
    write_text_checked(output_root / "quality_gate_summary.md", build_quality_gate_summary(judgments, retrieval_mode, web_reference_status))


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = resolve_project_path(project_root, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = load_eval_cases(project_root, args.cases)
    default_eval_path = (project_root / EVAL_CASE_DATASET).resolve()
    if not args.cases or resolve_project_path(project_root, args.cases).resolve() == default_eval_path:
        write_jsonl_checked(project_root / EVAL_CASE_DATASET, cases)
    canonical_by_id, canonical_ids = canonical_maps(project_root)
    if args.reuse_retrieval_results:
        retrievals = load_jsonl(resolve_project_path(project_root, args.reuse_retrieval_results))
        for row in retrievals:
            row["results"] = enrich_results(row.get("results") or [], canonical_by_id)
    else:
        retrievals = run_retrieval(project_root, cases, args.top_k, output_root, args.retrieval_mode)
    by_id = {row["query_id"]: row for row in retrievals}
    judgments = [score_retrieval_case(case, by_id.get(case["query_id"], {"results": []}), canonical_ids) for case in cases]
    web_reference_status = detect_web_reference_status(project_root, output_root)
    write_quality_outputs(output_root, cases, retrievals, judgments, args.retrieval_mode, web_reference_status)
    counts = Counter(row["quality_label"] for row in judgments)
    print(
        "Retrieval quality evaluation complete: "
        f"PASS={counts['PASS']} WEAK_PASS={counts['WEAK_PASS']} "
        f"NEEDS_FIX={counts['NEEDS_FIX']} UNSURE={counts['UNSURE']}"
    )
    print(f"Open: {output_root / 'quality_gate_summary.md'}")
    print(f"Open: {output_root / 'failure_taxonomy_and_fix_plan.md'}")
    print(f"Open: {output_root / 'human_spotcheck_samples.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
