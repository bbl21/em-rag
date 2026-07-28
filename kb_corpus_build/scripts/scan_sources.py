#!/usr/bin/env python3
"""Scan first-stage source inventory for the EM RAG corpus build."""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only on minimal runtimes.
    yaml = None


BUILD_DIRS = [
    "config",
    "metadata",
    "scripts",
    "logs",
    "intermediate/latex",
    "intermediate/pdf",
    "intermediate/cleaned",
    "corpus/by_source",
    "indexes/bm25",
    "indexes/vector",
    "indexes/structured",
    "rag",
    "eval/datasets",
    "eval/reports",
    "reports",
]


SOURCE_SPECS = [
    {
        "source_id": "ellingson_em_vol1",
        "title": "Electromagnetics Volume 1",
        "source_type": "latex_book",
        "plan_raw_path": "electromagnetics-vol-1-latex",
        "priority": "high",
        "domain_tags": ["electromagnetics", "transmission_line", "fields", "waves"],
    },
    {
        "source_id": "modern_antennas_microwave_circuits",
        "title": "Modern Antennas and Microwave Circuits",
        "source_type": "latex_book",
        "plan_raw_path": "Modern_Antennas_Microwave_Circuits_Jan_2022_arxiv_latex",
        "priority": "high",
        "domain_tags": ["antennas", "microwave_circuits", "phased_array", "transmission_line"],
    },
    {
        "source_id": "mit_em_applications",
        "title": "Electromagnetics and Applications",
        "source_type": "pdf_course",
        "plan_raw_path": "Electromagnetics-and-Applications.pdf",
        "priority": "high",
        "domain_tags": ["electromagnetics", "radiation", "guided_waves", "waves"],
    },
    {
        "source_id": "itu_r_p1411_13",
        "title": "ITU-R P.1411-13",
        "source_type": "pdf_standard",
        "plan_raw_path": "R-REC-P.1411-13-202509-I!!PDF-E.pdf",
        "priority": "high",
        "domain_tags": ["propagation_model", "path_loss", "short_range", "outdoor_radio_propagation", "itu_r"],
    },
]


@dataclass(frozen=True)
class PdfInspection:
    page_count: int | None
    text_layer_pages: int | None
    text_layer_page_ratio: float | None
    method: str
    quality_flags: list[str]
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan raw sources for phase one.")
    parser.add_argument("--project-root", default=".", help="Project root containing reference/ and RAG-plan.md.")
    parser.add_argument("--output-dir", default="kb_corpus_build", help="Build output directory relative to project root.")
    return parser.parse_args()


def posix_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def create_build_tree(build_root: Path) -> None:
    for rel in BUILD_DIRS:
        (build_root / rel).mkdir(parents=True, exist_ok=True)


def resolve_source_path(project_root: Path, plan_raw_path: str) -> Path:
    reference_path = project_root / "reference" / plan_raw_path
    if reference_path.exists():
        return reference_path
    return project_root / plan_raw_path


def safe_read_text(path: Path, max_bytes: int = 1_000_000) -> str:
    data = path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="ignore")


def detect_main_tex(source_path: Path) -> tuple[str | None, list[str], int]:
    tex_files = sorted(source_path.rglob("*.tex"), key=lambda p: p.relative_to(source_path).as_posix().lower())
    by_name = {path.name.lower(): path for path in tex_files}
    for name in ("main.tex", "book.tex", "root.tex"):
        if name in by_name:
            return by_name[name].as_posix(), [by_name[name].as_posix()], len(tex_files)

    documentclass_candidates: list[Path] = []
    for tex_file in tex_files:
        try:
            if "\\documentclass" in safe_read_text(tex_file):
                documentclass_candidates.append(tex_file)
        except OSError:
            continue

    candidates = [path.as_posix() for path in documentclass_candidates]
    selected = candidates[0] if candidates else None
    return selected, candidates, len(tex_files)


def inspect_pdf_with_fitz(path: Path) -> PdfInspection | None:
    if importlib.util.find_spec("fitz") is None:
        return None
    try:
        import fitz  # type: ignore

        with fitz.open(path) as document:
            page_count = document.page_count
            text_pages = sum(1 for page in document if page.get_text("text").strip())
        return PdfInspection(page_count, text_pages, ratio(text_pages, page_count), "pymupdf", [])
    except Exception as exc:  # pragma: no cover - depends on optional library.
        return PdfInspection(None, None, None, "pymupdf", ["pdf_library_error"], str(exc))


def inspect_pdf_with_pypdf(path: Path) -> PdfInspection | None:
    module_name = "pypdf" if importlib.util.find_spec("pypdf") is not None else None
    if module_name is None and importlib.util.find_spec("PyPDF2") is not None:
        module_name = "PyPDF2"
    if module_name is None:
        return None
    try:
        module = __import__(module_name)
        reader = module.PdfReader(str(path))
        page_count = len(reader.pages)
        text_pages = sum(1 for page in reader.pages if (page.extract_text() or "").strip())
        return PdfInspection(page_count, text_pages, ratio(text_pages, page_count), module_name, [])
    except Exception as exc:  # pragma: no cover - depends on optional library.
        return PdfInspection(None, None, None, module_name, ["pdf_library_error"], str(exc))


def inspect_pdf_with_tools(path: Path) -> PdfInspection | None:
    pdfinfo = shutil.which("pdfinfo")
    pdftotext = shutil.which("pdftotext")
    if not pdfinfo or not pdftotext:
        return None

    try:
        info = subprocess.run([pdfinfo, str(path)], text=True, capture_output=True, check=True)
        page_match = re.search(r"^Pages:\s+(\d+)", info.stdout, re.MULTILINE)
        page_count = int(page_match.group(1)) if page_match else None
        if not page_count:
            return PdfInspection(None, None, None, "poppler", ["pdf_page_count_unknown"])

        text_pages = 0
        for page in range(1, page_count + 1):
            extracted = subprocess.run(
                [pdftotext, "-f", str(page), "-l", str(page), str(path), "-"],
                text=True,
                capture_output=True,
                check=False,
            )
            if extracted.stdout.strip():
                text_pages += 1
        return PdfInspection(page_count, text_pages, ratio(text_pages, page_count), "poppler", [])
    except Exception as exc:  # pragma: no cover - depends on optional tools.
        return PdfInspection(None, None, None, "poppler", ["pdf_tool_error"], str(exc))


def extract_pdf_streams(data: bytes) -> list[bytes]:
    streams: list[bytes] = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL):
        raw = match.group(1).strip()
        streams.append(raw)
        try:
            streams.append(zlib.decompress(raw))
        except zlib.error:
            continue
    return streams


def inspect_pdf_builtin(path: Path) -> PdfInspection:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return PdfInspection(None, None, None, "builtin_pdf_scan", ["pdf_read_error"], str(exc))

    page_count = len(re.findall(rb"/Type\s*/Page\b", data))
    text_blob = b"\n".join([data, *extract_pdf_streams(data)])
    has_text_layer_markers = bool(re.search(rb"\bBT\b|\bTj\b|\bTJ\b|/Font\b", text_blob))
    text_pages = page_count if page_count and has_text_layer_markers else 0
    flags = ["builtin_pdf_scan_estimated"]
    if not page_count:
        flags.append("pdf_page_count_unknown")
    if has_text_layer_markers:
        flags.append("text_layer_detected_by_markers")
    else:
        flags.append("text_layer_not_detected_by_markers")
    return PdfInspection(page_count or None, text_pages if page_count else None, ratio(text_pages, page_count), "builtin_pdf_scan", flags)


def inspect_pdf(path: Path) -> PdfInspection:
    for inspector in (inspect_pdf_with_fitz, inspect_pdf_with_pypdf, inspect_pdf_with_tools):
        result = inspector(path)
        if result is not None and result.page_count is not None:
            return result
    return inspect_pdf_builtin(path)


def ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return round(numerator / denominator, 4)


def inspect_source(spec: dict[str, Any], project_root: Path) -> dict[str, Any]:
    source_path = resolve_source_path(project_root, spec["plan_raw_path"])
    raw_path = posix_relative(source_path, project_root) if source_path.exists() else spec["plan_raw_path"]
    record: dict[str, Any] = {
        "source_id": spec["source_id"],
        "title": spec["title"],
        "source_type": spec["source_type"],
        "raw_path": raw_path,
        "detected_main_file": None,
        "license": "unknown_or_detected",
        "allowed_use": "internal_rag_pending_license_review",
        "priority": spec["priority"],
        "domain_tags": spec["domain_tags"],
        "exists": source_path.exists(),
        "path_type": "directory" if source_path.is_dir() else "file" if source_path.is_file() else "missing",
        "size_bytes": source_size(source_path) if source_path.exists() else 0,
        "quality_flags": [],
    }

    if not source_path.exists():
        record["quality_flags"].append("source_missing")
        return record

    if spec["source_type"] == "latex_book":
        selected, candidates, tex_count = detect_main_tex(source_path)
        record["tex_file_count"] = tex_count
        record["main_tex_candidates"] = [posix_relative(Path(candidate), project_root) for candidate in candidates]
        record["detected_main_file"] = posix_relative(Path(selected), project_root) if selected else None
        if not selected:
            record["quality_flags"].append("main_tex_not_found")
    else:
        inspection = inspect_pdf(source_path)
        record["pdf_page_count"] = inspection.page_count
        record["text_layer_pages"] = inspection.text_layer_pages
        record["text_layer_page_ratio"] = inspection.text_layer_page_ratio
        record["pdf_inspection_method"] = inspection.method
        record["quality_flags"].extend(inspection.quality_flags)
        if inspection.error:
            record["pdf_inspection_error"] = inspection.error

    return record


def source_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    if yaml is None:
        path.write_text(render_simple_yaml(payload), encoding="utf-8", newline="\n")
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")


def render_simple_yaml(payload: dict[str, Any]) -> str:
    lines = ["sources:"]
    for source in payload["sources"]:
        lines.append("  -")
        for key, value in source.items():
            if isinstance(value, list):
                lines.append(f"    {key}:")
                for item in value:
                    lines.append(f"      - {item}")
            else:
                rendered = "null" if value is None else value
                lines.append(f"    {key}: {rendered}")
    return "\n".join(lines) + "\n"


def write_inventory(path: Path, sources: list[dict[str, Any]]) -> None:
    lines = [
        "# Source Inventory",
        "",
        "| source_id | type | raw_path | exists | key scan result | flags |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for source in sources:
        if source["source_type"] == "latex_book":
            key = f"tex files: {source.get('tex_file_count', 0)}; main: {source.get('detected_main_file') or 'unknown'}"
        else:
            key = (
                f"pages: {source.get('pdf_page_count')}; "
                f"text pages: {source.get('text_layer_pages')}; "
                f"ratio: {source.get('text_layer_page_ratio')}; "
                f"method: {source.get('pdf_inspection_method')}"
            )
        flags = ", ".join(source.get("quality_flags", [])) or "none"
        lines.append(
            f"| {source['source_id']} | {source['source_type']} | {source['raw_path']} | "
            f"{source['exists']} | {key} | {flags} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_report(path: Path, sources: list[dict[str, Any]], build_root: Path) -> None:
    all_present = all(source["exists"] for source in sources)
    has_warnings = any(source.get("quality_flags") for source in sources)
    needs_pdf_validation = any(
        "builtin_pdf_scan_estimated" in source.get("quality_flags", [])
        for source in sources
        if source["source_type"].startswith("pdf")
    )
    if not all_present:
        phase_status = "blocked"
    elif has_warnings:
        phase_status = "completed_with_warnings"
    else:
        phase_status = "completed"
    pdf_precision_status = "needs_validation" if needs_pdf_validation else "verified"

    lines = [
        "# Phase 1 Scan Report",
        "",
        f"Phase status: {phase_status}",
        f"PDF precision status: {pdf_precision_status}",
        "",
        "## Summary",
        "",
        f"- Sources expected: {len(sources)}",
        f"- Sources found: {sum(1 for source in sources if source['exists'])}",
        f"- Output root: {build_root.as_posix()}",
        "",
        "## Source Checks",
        "",
    ]
    for source in sources:
        lines.extend([
            f"### {source['source_id']}",
            "",
            f"- title: {source['title']}",
            f"- type: {source['source_type']}",
            f"- raw_path: {source['raw_path']}",
            f"- exists: {source['exists']}",
            f"- size_bytes: {source['size_bytes']}",
        ])
        if source["source_type"] == "latex_book":
            lines.extend([
                f"- tex_file_count: {source.get('tex_file_count')}",
                f"- detected_main_file: {source.get('detected_main_file')}",
                f"- main_tex_candidates: {source.get('main_tex_candidates', [])}",
            ])
        else:
            lines.extend([
                f"- pdf_page_count: {source.get('pdf_page_count')}",
                f"- text_layer_pages: {source.get('text_layer_pages')}",
                f"- text_layer_page_ratio: {source.get('text_layer_page_ratio')}",
                f"- pdf_inspection_method: {source.get('pdf_inspection_method')}",
            ])
        flags = source.get("quality_flags", [])
        lines.append(f"- quality_flags: {flags if flags else 'none'}")
        lines.append("")

    lines.extend([
        "## Dependency Notes",
        "",
        f"- fitz_available: {importlib.util.find_spec('fitz') is not None}",
        f"- pypdf_available: {importlib.util.find_spec('pypdf') is not None}",
        f"- PyPDF2_available: {importlib.util.find_spec('PyPDF2') is not None}",
        f"- pdfinfo_available: {shutil.which('pdfinfo') is not None}",
        f"- pdftotext_available: {shutil.which('pdftotext') is not None}",
        "- If only builtin_pdf_scan is available, page and text-layer values are approximate and must be revisited before PDF extraction.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_requirements(build_root: Path) -> None:
    requirements = build_root / "requirements.txt"
    requirements.write_text(
        "# Phase-one scan dependencies. Use a virtual environment; do not break system Python.\n"
        "PyYAML>=6.0\n"
        "pypdf>=4.0.0\n"
        "PyMuPDF>=1.24.0\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    build_root = (project_root / args.output_dir).resolve()

    create_build_tree(build_root)
    sources = [inspect_source(spec, project_root) for spec in SOURCE_SPECS]

    write_yaml(build_root / "metadata" / "sources.yaml", {"sources": sources})
    write_inventory(build_root / "metadata" / "source_inventory.md", sources)
    write_report(build_root / "reports" / "scan_report.md", sources, build_root)
    write_requirements(build_root)

    print(f"Scanned {len(sources)} sources; found {sum(1 for source in sources if source['exists'])}.")
    print(f"Wrote {build_root / 'metadata' / 'sources.yaml'}")
    print(f"Wrote {build_root / 'metadata' / 'source_inventory.md'}")
    print(f"Wrote {build_root / 'reports' / 'scan_report.md'}")
    return 0 if all(source["exists"] for source in sources) else 1


if __name__ == "__main__":
    sys.exit(main())
