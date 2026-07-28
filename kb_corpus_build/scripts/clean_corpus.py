#!/usr/bin/env python3
"""Phase-4 cleaning and normalization for the EM knowledge corpus build."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


LATEX_LAYOUT_TOKENS = {
    "blue",
    "black",
    "white",
    "red",
    "green",
    "gray",
    "grey",
    "small",
    "large",
    "Large",
    "LARGE",
    "Huge",
    "center",
    "flushleft",
    "flushright",
}

LATEX_LAYOUT_ENVIRONMENTS = {
    "center",
    "flushleft",
    "flushright",
}

PDF_DIRECTORY_OR_INDEX_HEADINGS = {
    "contents",
    "table of contents",
    "index",
    "list of figures",
    "list of tables",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and standardize extracted LaTeX/PDF corpus units.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    return parser.parse_args()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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


def write_jsonl_checked(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    text = "\n".join(lines) + ("\n" if lines else "")
    write_text_checked(path, text)


def normalize_inline_space(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def join_paragraph_lines(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return normalize_inline_space("\n\n".join(paragraphs))


def clean_latex_content(content_md: str, headings: set[str]) -> tuple[str, bool]:
    original = content_md or ""
    normalized = original.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    changed = False

    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        cleaned_line = re.sub(r"\\color\{[A-Za-z]+\}", "", line)
        cleaned_line = re.sub(r"\\?(?:begin|end)\{mdframed\}", "", cleaned_line)
        cleaned_line = re.sub(r"\\?mdframed(?:\[[^\]]*\])?", "", cleaned_line)
        cleaned_line = re.sub(
            r"\\(?:begin|end)\{(" + "|".join(sorted(LATEX_LAYOUT_ENVIRONMENTS)) + r")\}",
            "",
            cleaned_line,
        )
        if cleaned_line != line:
            changed = True
            line = cleaned_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        remove_line = False
        if line == r"\relax":
            remove_line = True
        elif re.fullmatch(r"\{[A-Za-z][A-Za-z0-9_-]*\}", line) and line[1:-1] in LATEX_LAYOUT_TOKENS:
            remove_line = True
        elif line.startswith("{") and line.endswith("}") and line[1:-1].strip() in headings:
            remove_line = True
        elif re.fullmatch(r"\d+\.[A-Za-z]{1,4}", line):
            remove_line = True
        elif re.fullmatch(r"\[[^\]]*(?:backgroundcolor|linecolor|nobreak)[^\]]*\]", line):
            remove_line = True

        if remove_line:
            changed = True
            continue

        if line.startswith("{") and line.endswith("}"):
            inner = line[1:-1].strip()
            if inner and (" " in inner or re.search(r"[.;:,]", inner)) and not re.search(r"[$\\_^]", inner):
                line = inner
                changed = True

        if line != raw_line:
            changed = True
        cleaned_lines.append(line)

    cleaned = join_paragraph_lines(cleaned_lines)
    if cleaned != normalize_inline_space(original):
        changed = True
    return cleaned, changed


def is_pdf_noise_line(line: str, line_index: int, line_count: int) -> bool:
    if not line:
        return False
    near_page_edge = line_index < 4 or line_index >= max(0, line_count - 3)
    if line.casefold() in PDF_DIRECTORY_OR_INDEX_HEADINGS:
        return True
    if near_page_edge and re.fullmatch(r"\d{1,3}", line):
        return True
    if re.fullmatch(r"-\s*[A-Za-z]*\d+\s*-", line):
        return True
    if near_page_edge and re.match(r"^Rec\.\s+ITU-R\b", line):
        return True
    if re.fullmatch(r"(?:FIGURE|TABLE)\s+\d+[A-Z]?", line):
        return True
    return False


def stitch_pdf_lines(lines: list[str]) -> str:
    output: list[str] = []
    for line in lines:
        if not output:
            output.append(line)
            continue

        previous = output[-1]
        if previous.endswith("-") and re.match(r"^[a-z]", line):
            output[-1] = previous + line
            continue

        if not re.search(r"[.!?:;]\)?$", previous) and re.match(r"^[A-Za-z0-9$(]", line):
            output[-1] = f"{previous} {line}"
            continue

        output.append(line)

    return normalize_inline_space("\n".join(output))


def clean_pdf_content(content_md: str) -> tuple[str, bool]:
    original = content_md or ""
    normalized = original.replace("\r\n", "\n").replace("\r", "\n")
    filtered_lines: list[str] = []
    changed = False

    raw_lines = normalized.split("\n")
    for line_index, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            continue
        if is_pdf_noise_line(line, line_index, len(raw_lines)):
            changed = True
            continue
        if line != raw_line:
            changed = True
        filtered_lines.append(line)

    cleaned = stitch_pdf_lines(filtered_lines)
    if cleaned != normalize_inline_space(original):
        changed = True
    return cleaned, changed


def normalized_quality_flags(flags: Any, changed: bool, content_md: str) -> list[str]:
    values = [str(flag) for flag in flags or []]
    if not content_md:
        values.append("empty_content")
    return sorted(set(values))


def build_latex_unit(row: dict[str, Any], ordinal: int) -> dict[str, Any]:
    headings = {
        str(value).strip()
        for key in ("part", "chapter", "section", "subsection", "subsubsection")
        for value in [row.get(key)]
        if isinstance(value, str) and value.strip()
    }
    content_md, changed = clean_latex_content(str(row.get("content_md") or ""), headings)
    source_id = str(row.get("source_id") or "")
    return {
        "unit_id": f"{source_id}:latex_section:{ordinal:05d}",
        "source_id": source_id,
        "source_title": str(row.get("source_title") or ""),
        "source_type": str(row.get("source_type") or ""),
        "raw_path": str(row.get("raw_path") or ""),
        "tex_file": row.get("tex_file"),
        "origin_type": "latex_section",
        "chapter": row.get("chapter"),
        "section": row.get("section"),
        "subsection": row.get("subsection"),
        "page_start": None,
        "page_end": None,
        "content_md": content_md,
        "equations": list(row.get("equations") or []),
        "figures": list(row.get("figures") or []),
        "tables": list(row.get("tables") or []),
        "quality_flags": normalized_quality_flags(row.get("quality_flags"), changed, content_md),
    }


def build_pdf_unit(row: dict[str, Any], ordinal: int) -> dict[str, Any]:
    content_md, changed = clean_pdf_content(str(row.get("content_md") or ""))
    source_id = str(row.get("source_id") or "")
    page = row.get("page")
    return {
        "unit_id": f"{source_id}:pdf_page:{ordinal:05d}",
        "source_id": source_id,
        "source_title": str(row.get("source_title") or ""),
        "source_type": str(row.get("source_type") or ""),
        "raw_path": str(row.get("raw_path") or ""),
        "origin_type": "pdf_page",
        "chapter": row.get("chapter"),
        "section": row.get("section"),
        "subsection": None,
        "page_start": page,
        "page_end": page,
        "content_md": content_md,
        "equations": list(row.get("equations") or []),
        "figures": list(row.get("figures") or []),
        "tables": list(row.get("tables") or []),
        "quality_flags": normalized_quality_flags(row.get("quality_flags"), changed, content_md),
    }


def collect_units(project_root: Path) -> tuple[list[dict[str, Any]], int, int]:
    build_root = project_root / "kb_corpus_build"
    latex_dir = build_root / "intermediate" / "latex"
    pdf_dir = build_root / "intermediate" / "pdf"
    units: list[dict[str, Any]] = []
    latex_units = 0
    pdf_units = 0

    latex_ordinal = 0
    for path in sorted(latex_dir.glob("*.jsonl")):
        for row in load_jsonl(path):
            latex_ordinal += 1
            latex_units += 1
            units.append(build_latex_unit(row, latex_ordinal))

    pdf_ordinal = 0
    for path in sorted(pdf_dir.glob("*.jsonl")):
        for row in load_jsonl(path):
            pdf_ordinal += 1
            pdf_units += 1
            units.append(build_pdf_unit(row, pdf_ordinal))

    return units, latex_units, pdf_units


def write_report(project_root: Path, latex_units: int, pdf_units: int, cleaned_units: int) -> None:
    build_root = project_root / "kb_corpus_build"
    report_path = build_root / "reports" / "cleaning_report.md"
    units_path = build_root / "intermediate" / "cleaned" / "cleaned_units.jsonl"
    lines = [
        "# Phase 4 Cleaning Report",
        "",
        "Phase status: completed",
        f"latex_units: {latex_units}",
        f"pdf_units: {pdf_units}",
        f"cleaned_units: {cleaned_units}",
        f"cleaned_units_path: {safe_rel(units_path, project_root)}",
    ]
    write_text_checked(report_path, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    build_root = project_root / "kb_corpus_build"
    units_path = build_root / "intermediate" / "cleaned" / "cleaned_units.jsonl"

    units, latex_units, pdf_units = collect_units(project_root)
    write_jsonl_checked(units_path, units)
    write_report(project_root, latex_units, pdf_units, len(units))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
