#!/usr/bin/env python3
"""Build a deterministic manual review pack for canonical chunks."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable


RANDOM_SEED = "20260629"
PER_SOURCE_LIMIT = 15
FORMULA_LIMIT = 30
ITU_LIMIT = 30
METADATA_LIMIT = 72
NOISE_LIMIT = 40
METADATA_FIELDS = ["source_id", "chapter", "section", "page_start", "page_end", "tex_file"]
CITATION_BASE_FIELDS = ["source_id", "raw_path"]

FREQUENCY_RANGE_RE = re.compile(
    r"\b(?:from\s+|between\s+)?"
    r"\d+(?:[\s,]\d{3})*(?:\.\d+)?\s*"
    r"(?:Hz|kHz|MHz|GHz|THz)?\s+"
    r"(?:to|and)\s+"
    r"\d+(?:[\s,]\d{3})*(?:\.\d+)?\s*"
    r"(?:Hz|kHz|MHz|GHz|THz)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deterministic manual review pack.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    return parser.parse_args()


def normalize_scalar(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_content_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_scalar(item) for item in value if normalize_scalar(item)]
    scalar = normalize_scalar(value)
    return [scalar] if scalar else []


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    alignments = []
    for header in headers:
        alignments.append("---:" if "#" in header or "数量" in header or "比例" in header or "token" in header else "---")
    lines.append("|" + "|".join(alignments) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def bool_cn(value: bool) -> str:
    return "是" if value else "否"


def percent(part: int, whole: int) -> str:
    if whole <= 0:
        return "0.00%"
    return f"{(part / whole) * 100:.2f}%"


def parse_report_count(report_text: str, key: str) -> int | None:
    match = re.search(rf"^{re.escape(key)}:\s*(\d+)\s*$", report_text, flags=re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def dedup_ratio_from_report(path: Path) -> tuple[str, str]:
    if not path.is_file():
        return "unknown", "未找到 dedup_report.md"
    report_text = path.read_text(encoding="utf-8")
    input_chunks = parse_report_count(report_text, "input_chunks")
    canonical_chunks = parse_report_count(report_text, "canonical_chunks")
    if not input_chunks or not canonical_chunks:
        return "unknown", "dedup_report.md 缺少 input_chunks/canonical_chunks"
    duplicate_count = max(0, input_chunks - canonical_chunks)
    return percent(duplicate_count, canonical_chunks), f"{duplicate_count}/{canonical_chunks}"


def sample_key(row: dict[str, Any], label: str) -> str:
    chunk_id = normalize_scalar(row.get("chunk_id"))
    return sha256(f"{RANDOM_SEED}|{label}|{chunk_id}".encode("utf-8")).hexdigest()


def stable_sample(rows: list[dict[str, Any]], limit: int, label: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (sample_key(row, label), normalize_scalar(row.get("chunk_id"))))[:limit]


def missing_fields(row: dict[str, Any]) -> list[str]:
    return [field for field in METADATA_FIELDS if not normalize_scalar(row.get(field))]


def is_pdf_row(row: dict[str, Any]) -> bool:
    return normalize_scalar(row.get("source_type")).lower() in {"pdf_course", "pdf_standard"}


def is_latex_row(row: dict[str, Any]) -> bool:
    return normalize_scalar(row.get("source_type")).lower() == "latex_book"


def applicable_citation_fields(row: dict[str, Any]) -> list[str]:
    fields = list(CITATION_BASE_FIELDS)
    if is_pdf_row(row):
        fields.extend(["page_start", "page_end"])
    if is_latex_row(row):
        fields.append("tex_file")
    return fields


def applicable_missing_fields(row: dict[str, Any]) -> list[str]:
    return [field for field in applicable_citation_fields(row) if not normalize_scalar(row.get(field))]


def chapter_section_missing_fields(row: dict[str, Any]) -> list[str]:
    return [field for field in ("chapter", "section") if not normalize_scalar(row.get(field))]


def formula_score(row: dict[str, Any]) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 0
    content_type = normalize_scalar(row.get("content_type")).lower()
    equations = normalize_list(row.get("equations"))
    content = normalize_content_text(row.get("content_md"))
    if content_type == "formula":
        score += 5
        reasons.append("内容类型为 formula")
    if equations:
        score += 5
        reasons.append("equations 字段非空")
    if re.search(r"[=\\]", " ".join(equations) or content):
        score += 3
        reasons.append("命中公式模式")
    if re.search(r"\b(where|denote|represents|is)\b", content, flags=re.IGNORECASE):
        score += 2
        reasons.append("命中变量解释线索")
    if re.search(r"\b(valid|applies|condition|assumptions?)\b", content, flags=re.IGNORECASE):
        score += 2
        reasons.append("命中适用条件线索")
    if 80 <= len(content.split()) <= 1400:
        score += 2
        reasons.append("长度适合审查")
    return score, reasons


def itu_flags(row: dict[str, Any]) -> dict[str, bool]:
    content = normalize_content_text(row.get("content_md"))
    equations = normalize_list(row.get("equations"))
    lower = content.lower()
    return {
        "公式": bool(equations),
        "频率范围": bool(FREQUENCY_RANGE_RE.search(content)),
        "场景": bool(re.search(r"\b(urban|suburban|rural|street canyon|rooftop|indoor|outdoor|microcell|macrocell|campus)\b", lower)),
        "LoS/NLoS": bool(re.search(r"\b(?:los|nlos|line-of-sight|non-line-of-sight)\b", lower)),
        "输入/输出": bool(re.search(r"\b(input|output|parameter|distance|height|loss)\b", lower)),
        "限制条件": bool(re.search(r"\b(valid|applicable|assumption|limitation|shall|must)\b", lower)),
    }


def itu_score(row: dict[str, Any]) -> int:
    flags = itu_flags(row)
    return sum(2 for value in flags.values() if value) - (0 if normalize_list(row.get("equations")) else 1)


def detect_noise_types(row: dict[str, Any]) -> list[str]:
    content = normalize_content_text(row.get("content_md"))
    lower = content.lower()
    types: list[str] = []
    if re.search(r"(\\[A-Za-z]+|\{[^{}]{0,40}\}|markup residue|latex)", content):
        types.append("LaTeX 残留")
    if re.search(r"\b(header|footer|page boundary|page \d+|pdf)\b", lower):
        types.append("PDF 页眉/页脚")
    if re.search(r"\b(index|contents|table of contents|catalogue|appendix)\b", lower):
        types.append("疑似目录/索引")
    if re.search(r"\bfigure|table\b", lower):
        types.append("图表残留")
    if re.search(r"\b(repeated|short lines|artifact)\b", lower):
        types.append("异常空格/断行")
    return types


def noise_score(row: dict[str, Any], noise_types: list[str]) -> int:
    return len(noise_types) * 2 + len(normalize_list(row.get("quality_flags")))


def render_chunk_detail(row: dict[str, Any], extra_lines: list[str] | None = None) -> str:
    extra_lines = extra_lines or []
    chunk_id = normalize_scalar(row.get("chunk_id"))
    source_id = normalize_scalar(row.get("source_id")) or "unknown"
    chapter = normalize_scalar(row.get("chapter")) or "null"
    section = normalize_scalar(row.get("section")) or "null"
    page_start = row.get("page_start")
    page_end = row.get("page_end")
    tex_file = normalize_scalar(row.get("tex_file")) or "null"
    content_type = normalize_scalar(row.get("content_type")) or "unknown"
    token_estimate = row.get("token_estimate") or 0
    quality_flags = ", ".join(normalize_list(row.get("quality_flags"))) or "<none>"
    lines = [
        f"## {chunk_id}",
        "",
        f"- chunk_id（块 ID）: {chunk_id}",
        f"- source_id（来源 ID）: {source_id}",
        f"- chapter（章）: {chapter}",
        f"- section（节）: {section}",
        f"- page_start/page_end（起止页）: {page_start} / {page_end}",
        f"- tex_file（LaTeX 文件）: {tex_file}",
        f"- content_type（内容类型）: {content_type}",
        f"- token_estimate（估算 tokens）: {token_estimate}",
        f"- quality_flags（质量标记）: {quality_flags}",
        "",
    ]
    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")
    equations = normalize_list(row.get("equations"))
    if equations:
        lines.extend(["### 公式字段 equations", "", "~~~text", "\n\n".join(equations), "~~~", ""])
    lines.extend(["### 原始内容 content_md", "", "~~~markdown", normalize_content_text(row.get("content_md")), "~~~", ""])
    return "\n".join(lines)


def render_front_matter(title: str, total_rows: int, description: str) -> list[str]:
    return [
        f"# {title}",
        "",
        "- 数据来源: `kb_corpus_build/corpus/chunks.canonical.jsonl`",
        f"- 随机种子: `{RANDOM_SEED}`",
        f"- 已加载 canonical 行数: {total_rows}",
        f"- 说明: {description}",
        "",
    ]


def build_per_source_samples(rows: list[dict[str, Any]]) -> tuple[str, int, list[str]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[normalize_scalar(row.get("source_id")) or "unknown"].append(row)
    lines = render_front_matter(
        "按 source_id 随机抽样",
        len(rows),
        "每个 source_id 最多抽样 15 个 canonical chunk；每条样本保留完整 content_md 供人工审查。",
    )
    chunk_ids: list[str] = []
    total = 0
    for source_id in sorted(by_source):
        selected = stable_sample(by_source[source_id], PER_SOURCE_LIMIT, f"per-source:{source_id}")
        total += len(selected)
        chunk_ids.extend(normalize_scalar(row.get("chunk_id")) for row in selected)
        lines.extend(
            [
                f"## source_id：{source_id}",
                "",
                f"- 可用 chunk 数: {len(by_source[source_id])}",
                "",
                f"- 抽样 chunk 数: {len(selected)}",
                "",
            ]
        )
        for index, row in enumerate(selected, start=1):
            lines.append(render_chunk_detail(row).replace(f"## {normalize_scalar(row.get('chunk_id'))}", f"## {index}. {normalize_scalar(row.get('chunk_id'))}", 1))
    return "\n".join(lines), total, chunk_ids


def build_formula_samples(rows: list[dict[str, Any]]) -> tuple[str, int, list[str]]:
    candidates: list[tuple[dict[str, Any], int, list[str]]] = []
    for row in rows:
        if normalize_scalar(row.get("content_type")).lower() == "formula" or normalize_list(row.get("equations")):
            score, reasons = formula_score(row)
            candidates.append((row, score, reasons))
    selected = sorted(
        candidates,
        key=lambda item: (-item[1], sample_key(item[0], "formula"), normalize_scalar(item[0].get("chunk_id"))),
    )[:FORMULA_LIMIT]
    lines = render_front_matter(
        "公式 chunk 抽样",
        len(rows),
        "从 content_type 包含 formula 或 equations 非空的 chunk 中抽取高优先级公式样本。",
    )
    summary_rows = []
    chunk_ids: list[str] = []
    for index, (row, score, reasons) in enumerate(selected, start=1):
        chunk_ids.append(normalize_scalar(row.get("chunk_id")))
        summary_rows.append(
            [
                index,
                normalize_scalar(row.get("chunk_id")),
                normalize_scalar(row.get("source_id")) or "unknown",
                normalize_scalar(row.get("content_type")) or "unknown",
                score,
                ", ".join(reasons) or "基础命中",
            ]
        )
    lines.extend(
        [
            markdown_table(["#", "chunk_id", "source_id", "content_type", "分数", "优先原因"], summary_rows),
            "",
        ]
    )
    for index, (row, score, reasons) in enumerate(selected, start=1):
        lines.append(
            render_chunk_detail(
                row,
                extra_lines=[
                    f"- 公式优先级分数: {score}",
                    f"- 优先原因: {', '.join(reasons) or '基础命中'}",
                ],
            ).replace(f"## {normalize_scalar(row.get('chunk_id'))}", f"## {index}. {normalize_scalar(row.get('chunk_id'))}", 1)
        )
    return "\n".join(lines), len(selected), chunk_ids


def build_itu_samples(rows: list[dict[str, Any]]) -> tuple[str, int, list[str]]:
    candidates = [row for row in rows if normalize_scalar(row.get("content_type")).lower() == "standard_model"]
    selected = sorted(candidates, key=lambda row: (-itu_score(row), sample_key(row, "itu"), normalize_scalar(row.get("chunk_id"))))[:ITU_LIMIT]
    lines = render_front_matter(
        "ITU 模型 chunk 抽样",
        len(rows),
        "从 standard_model chunk 中抽取样本；覆盖项列仅表示启发式命中或人工待确认，不代表事实已验证。",
    )
    headers = [
        "#",
        "chunk_id",
        "页码",
        "content_type",
        "公式（启发式命中）",
        "频率范围（启发式命中）",
        "场景（人工待确认）",
        "LoS/NLoS（人工待确认）",
        "输入/输出（人工待确认）",
        "限制条件（人工待确认）",
        "是否全部同一 chunk（启发式命中）",
    ]
    summary_rows = []
    chunk_ids: list[str] = []
    for index, row in enumerate(selected, start=1):
        flags = itu_flags(row)
        chunk_ids.append(normalize_scalar(row.get("chunk_id")))
        all_in_same_chunk = all(flags.values())
        summary_rows.append(
            [
                index,
                normalize_scalar(row.get("chunk_id")),
                f"{row.get('page_start')} / {row.get('page_end')}",
                normalize_scalar(row.get("content_type")) or "unknown",
                bool_cn(flags["公式"]),
                bool_cn(flags["频率范围"]),
                bool_cn(flags["场景"]),
                bool_cn(flags["LoS/NLoS"]),
                bool_cn(flags["输入/输出"]),
                bool_cn(flags["限制条件"]),
                bool_cn(all_in_same_chunk),
            ]
        )
    lines.extend([markdown_table(headers, summary_rows), ""])
    for index, row in enumerate(selected, start=1):
        flags = itu_flags(row)
        lines.append(
            render_chunk_detail(
                row,
                extra_lines=[
                    f"- ITU 优先级分数: {itu_score(row)}",
                    "- 覆盖项（仅供人工复核）:",
                    f"  - 公式（启发式命中）: {bool_cn(flags['公式'])}",
                    f"  - 频率范围（启发式命中）: {bool_cn(flags['频率范围'])}",
                    f"  - 场景（人工待确认）: {bool_cn(flags['场景'])}",
                    f"  - LoS/NLoS（人工待确认）: {bool_cn(flags['LoS/NLoS'])}",
                    f"  - 输入/输出（人工待确认）: {bool_cn(flags['输入/输出'])}",
                    f"  - 限制条件（人工待确认）: {bool_cn(flags['限制条件'])}",
                ],
            ).replace(f"## {normalize_scalar(row.get('chunk_id'))}", f"## {index}. {normalize_scalar(row.get('chunk_id'))}", 1)
        )
    return "\n".join(lines), len(selected), chunk_ids


def build_metadata_samples(rows: list[dict[str, Any]]) -> tuple[str, int, list[str]]:
    candidates = []
    field_missing_counts = {field: 0 for field in METADATA_FIELDS}
    citation_fields = CITATION_BASE_FIELDS + ["page_start", "page_end", "tex_file"]
    citation_missing_counts = {field: 0 for field in citation_fields}
    for row in rows:
        missing = missing_fields(row)
        citation_missing = applicable_missing_fields(row)
        for field in missing:
            field_missing_counts[field] += 1
        for field in citation_missing:
            citation_missing_counts[field] += 1
        if missing or citation_missing:
            candidates.append((row, missing, citation_missing))
    selected = sorted(
        candidates,
        key=lambda item: (-(len(item[1]) + len(item[2])), sample_key(item[0], "metadata"), normalize_scalar(item[0].get("chunk_id"))),
    )[:METADATA_LIMIT]
    lines = render_front_matter(
        "metadata 缺失样本",
        len(rows),
        "严格缺失字段用于定位原始字段空值；适用 citation 缺失只统计该来源类型应具备的追溯字段。",
    )
    stats_rows = [[field, field_missing_counts[field], percent(field_missing_counts[field], len(rows))] for field in METADATA_FIELDS]
    citation_stats_rows = [[field, citation_missing_counts[field], percent(citation_missing_counts[field], len(rows))] for field in citation_fields]
    lines.extend(
        [
            "## 严格缺失字段统计",
            "",
            markdown_table(["字段", "缺失 chunk 数", "缺失比例"], stats_rows),
            "",
            "## 适用 citation 缺失字段统计",
            "",
            markdown_table(["字段", "适用 citation 缺失 chunk 数", "占全部 chunk 比例"], citation_stats_rows),
            "",
            "## 缺失字段样本",
            "",
        ]
    )
    sample_rows = []
    chunk_ids: list[str] = []
    for index, (row, missing, citation_missing) in enumerate(selected, start=1):
        chunk_ids.append(normalize_scalar(row.get("chunk_id")))
        sample_rows.append(
            [
                index,
                normalize_scalar(row.get("chunk_id")),
                normalize_scalar(row.get("source_id")) or "unknown",
                ", ".join(missing) or "<none>",
                ", ".join(citation_missing) or "<none>",
                len(missing) + len(citation_missing),
            ]
        )
    lines.extend([markdown_table(["#", "chunk_id", "source_id", "严格缺失字段", "适用 citation 缺失字段", "缺失数量"], sample_rows), ""])
    for index, (row, missing, citation_missing) in enumerate(selected, start=1):
        lines.append(
            render_chunk_detail(
                row,
                extra_lines=[
                    f"- 严格缺失字段: {', '.join(missing) or '<none>'}",
                    f"- 适用 citation 缺失字段: {', '.join(citation_missing) or '<none>'}",
                    f"- 缺失数量: {len(missing) + len(citation_missing)}",
                ],
            ).replace(f"## {normalize_scalar(row.get('chunk_id'))}", f"## {index}. {normalize_scalar(row.get('chunk_id'))}", 1)
        )
    return "\n".join(lines), len(selected), chunk_ids


def build_noise_samples(rows: list[dict[str, Any]]) -> tuple[str, int, list[str]]:
    strata: dict[str, dict[str, list[tuple[dict[str, Any], list[str]]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        types = detect_noise_types(row)
        if not types:
            continue
        source_id = normalize_scalar(row.get("source_id")) or "unknown"
        for noise_type in types:
            strata[source_id][noise_type].append((row, types))
    selections: dict[str, dict[str, list[tuple[dict[str, Any], list[str]]]]] = defaultdict(dict)
    flat_selected: list[tuple[dict[str, Any], list[str]]] = []
    for source_id in sorted(strata):
        for noise_type in sorted(strata[source_id]):
            chosen = sorted(
                strata[source_id][noise_type],
                key=lambda item: (
                    -noise_score(item[0], item[1]),
                    sample_key(item[0], f"noise:{source_id}:{noise_type}"),
                    normalize_scalar(item[0].get("chunk_id")),
                ),
            )[: max(1, NOISE_LIMIT // max(1, sum(len(group) for group in strata[source_id].values())))]
            selections[source_id][noise_type] = chosen
            flat_selected.extend(chosen)
    dedup: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for row, types in flat_selected:
        dedup[normalize_scalar(row.get("chunk_id"))] = (row, types)
    ordered_ids = sorted(dedup, key=lambda chunk_id: sample_key({"chunk_id": chunk_id}, "noise-final"))
    ordered = [dedup[chunk_id] for chunk_id in ordered_ids[:NOISE_LIMIT]]
    keep_ids = {normalize_scalar(row.get("chunk_id")) for row, _ in ordered}
    lines = render_front_matter(
        "疑似噪声样本",
        len(rows),
        "按 source_id 和噪声类型分层抽样，覆盖页眉页脚、目录/索引、LaTeX 残留等疑似噪声。",
    )
    chunk_ids: list[str] = []
    for source_id in sorted(selections):
        source_lines = [f"## source_id：{source_id}", ""]
        emitted = False
        for noise_type in sorted(selections[source_id]):
            chosen = [item for item in selections[source_id][noise_type] if normalize_scalar(item[0].get("chunk_id")) in keep_ids]
            if not chosen:
                continue
            emitted = True
            source_lines.extend([f"### 噪声类型：{noise_type}", ""])
            table_rows = []
            for index, (row, types) in enumerate(chosen, start=1):
                chunk_ids.append(normalize_scalar(row.get("chunk_id")))
                table_rows.append(
                    [
                        index,
                        normalize_scalar(row.get("chunk_id")),
                        normalize_scalar(row.get("source_id")) or "unknown",
                        noise_score(row, types),
                        ", ".join(types),
                    ]
                )
            source_lines.extend([markdown_table(["#", "chunk_id", "source_id", "分数", "疑似原因"], table_rows), ""])
            for index, (row, types) in enumerate(chosen, start=1):
                source_lines.append(
                    render_chunk_detail(
                        row,
                        extra_lines=[
                            f"- 疑似噪声分数: {noise_score(row, types)}",
                            f"- 疑似原因: {', '.join(types)}",
                        ],
                    ).replace(f"## {normalize_scalar(row.get('chunk_id'))}", f"## {index}. {normalize_scalar(row.get('chunk_id'))}", 1)
                )
        if emitted:
            lines.extend(source_lines)
    return "\n".join(lines), len(set(chunk_ids)), chunk_ids


def build_checklist(rows_by_id: dict[str, dict[str, Any]], sampled_chunk_ids: list[str]) -> tuple[str, int]:
    ordered_ids = sorted(dict.fromkeys(sampled_chunk_ids))
    lines = render_front_matter(
        "人工审查打分表",
        len(rows_by_id),
        "本审查包内每个去重后的抽样 chunk 对应一行；评分列由人工填写。",
    )
    lines.extend([f"- 去重后抽样 chunk 数: {len(ordered_ids)}", ""])
    table_rows = [[chunk_id, normalize_scalar(rows_by_id[chunk_id].get("source_id")) or "unknown", "", "", "", "", "", "", ""] for chunk_id in ordered_ids]
    lines.append(
        markdown_table(
            ["chunk_id", "来源", "citation 可追溯", "内容完整", "噪声少", "公式完整", "metadata 正确", "是否通过", "问题说明"],
            table_rows,
        )
    )
    lines.append("")
    return "\n".join(lines), len(ordered_ids)


def build_summary(
    rows: list[dict[str, Any]],
    per_source_count: int,
    formula_count: int,
    itu_count: int,
    metadata_count: int,
    noise_count: int,
    checklist_count: int,
    dedup_report_path: Path | None = None,
) -> str:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[normalize_scalar(row.get("source_id")) or "unknown"].append(row)
    strict_metadata_missing_total = sum(1 for row in rows if missing_fields(row))
    citation_missing_total = sum(1 for row in rows if applicable_missing_fields(row))
    chapter_section_missing_total = sum(1 for row in rows if chapter_section_missing_fields(row))
    duplicate_ratio, duplicate_detail = dedup_ratio_from_report(dedup_report_path) if dedup_report_path else ("unknown", "未提供 dedup_report.md")
    quality_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        flags = normalize_list(row.get("quality_flags"))
        if not flags:
            quality_counts["<none>"] += 1
            continue
        for flag in flags:
            quality_counts[flag] += 1
    lines = render_front_matter(
        "汇总指标",
        len(rows),
        "指标来自 canonical chunks；当前脚本只写入人工审查包，不修改其他 audit 输出。",
    )
    lines.extend(
        [
            "## 语料计数",
            "",
            f"- canonical chunk 数: {len(rows)}",
            "",
            f"- 严格 metadata 缺失 chunk 数: {strict_metadata_missing_total}",
            "",
            f"- 适用 citation 缺失 chunk 数: {citation_missing_total}",
            "",
            f"- chapter/section 缺失 chunk 数: {chapter_section_missing_total}",
            "",
            f"- duplicate/canonical 比例（来自 dedup_report: input-canonical/canonical）: {duplicate_ratio}",
            "",
            f"- duplicate/canonical 明细: {duplicate_detail}",
            "",
            "## 分 source 指标",
            "",
            markdown_table(
                [
                    "source_id",
                    "canonical chunk 数",
                    "平均 token_estimate",
                    "<200 tokens 比例",
                    ">1500 tokens 比例",
                    "严格 metadata 缺失比例",
                    "适用 citation 缺失比例",
                    "chapter/section 缺失比例",
                ],
                [
                    [
                        source_id,
                        len(source_rows),
                        f"{(sum(int(row.get('token_estimate') or 0) for row in source_rows) / len(source_rows)):.1f}",
                        percent(sum(1 for row in source_rows if int(row.get("token_estimate") or 0) < 200), len(source_rows)),
                        percent(sum(1 for row in source_rows if int(row.get("token_estimate") or 0) > 1500), len(source_rows)),
                        percent(sum(1 for row in source_rows if missing_fields(row)), len(source_rows)),
                        percent(sum(1 for row in source_rows if applicable_missing_fields(row)), len(source_rows)),
                        percent(sum(1 for row in source_rows if chapter_section_missing_fields(row)), len(source_rows)),
                    ]
                    for source_id, source_rows in sorted(by_source.items())
                ],
            ),
            "",
            "## quality_flags 分布",
            "",
            markdown_table(
                ["quality_flag", "chunk 数", "占 canonical 比例"],
                [[flag, count, percent(count, len(rows))] for flag, count in sorted(quality_counts.items())],
            ),
            "",
            "## 抽样文件计数",
            "",
            markdown_table(
                ["文件", "抽样/列出 chunk 数"],
                [
                    ["per_source_random_samples.md", per_source_count],
                    ["formula_chunk_samples.md", formula_count],
                    ["itu_model_chunk_samples.md", itu_count],
                    ["metadata_missing_samples.md", metadata_count],
                    ["suspicious_noise_samples.md", noise_count],
                    ["manual_review_checklist.md", checklist_count],
                    ["summary_metrics.md", 0],
                ],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    build_root = project_root / "kb_corpus_build"
    corpus_path = build_root / "corpus" / "chunks.canonical.jsonl"
    output_root = build_root / "audit" / "manual_review_pack"
    rows = load_jsonl(corpus_path)
    rows_by_id = {normalize_scalar(row.get("chunk_id")): row for row in rows}

    per_source_text, per_source_count, per_source_ids = build_per_source_samples(rows)
    formula_text, formula_count, formula_ids = build_formula_samples(rows)
    itu_text, itu_count, itu_ids = build_itu_samples(rows)
    metadata_text, metadata_count, metadata_ids = build_metadata_samples(rows)
    noise_text, noise_count, noise_ids = build_noise_samples(rows)
    checklist_text, checklist_count = build_checklist(
        rows_by_id,
        per_source_ids + formula_ids + itu_ids + metadata_ids + noise_ids,
    )
    summary_text = build_summary(
        rows,
        per_source_count,
        formula_count,
        itu_count,
        metadata_count,
        noise_count,
        checklist_count,
        build_root / "reports" / "dedup_report.md",
    )

    write_text_checked(output_root / "per_source_random_samples.md", per_source_text)
    write_text_checked(output_root / "formula_chunk_samples.md", formula_text)
    write_text_checked(output_root / "itu_model_chunk_samples.md", itu_text)
    write_text_checked(output_root / "metadata_missing_samples.md", metadata_text)
    write_text_checked(output_root / "suspicious_noise_samples.md", noise_text)
    write_text_checked(output_root / "manual_review_checklist.md", checklist_text)
    write_text_checked(output_root / "summary_metrics.md", summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
