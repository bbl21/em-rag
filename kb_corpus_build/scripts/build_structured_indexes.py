#!/usr/bin/env python3
"""Phase-11 SQLite indexes for deterministic structured retrieval."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQLite indexes from structured JSONL outputs.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build.")
    return parser.parse_args()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_scalar(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


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


def json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def list_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(normalize_scalar(item) for item in value)
    return normalize_scalar(value)


def related_chunk_ids_json(row: dict[str, Any]) -> str:
    related = row.get("related_chunk_ids")
    if isinstance(related, list):
        return json.dumps([normalize_scalar(item) for item in related if normalize_scalar(item)], ensure_ascii=False)
    return "[]"


def recreate_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return sqlite3.connect(path)


def build_formula_db(path: Path, rows: list[dict[str, Any]]) -> None:
    with recreate_database(path) as conn:
        conn.execute(
            """
            create table formulas (
                formula_id text primary key,
                source_id text not null,
                chapter text,
                section text,
                formula_latex text,
                variables_json text,
                meaning text,
                applicable_conditions text,
                related_chunk_ids_json text,
                content_type text not null,
                domain_tags_text text not null,
                search_text text not null
            )
            """
        )
        for row in rows:
            search_text = " ".join(
                [
                    normalize_scalar(row.get("formula_id")),
                    normalize_scalar(row.get("source_id")),
                    normalize_scalar(row.get("chapter")),
                    normalize_scalar(row.get("section")),
                    normalize_scalar(row.get("formula_latex")),
                    json_dumps(row.get("variables")),
                    normalize_scalar(row.get("meaning")),
                    normalize_scalar(row.get("applicable_conditions")),
                ]
            )
            conn.execute(
                """
                insert into formulas values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalize_scalar(row.get("formula_id")),
                    normalize_scalar(row.get("source_id")),
                    normalize_scalar(row.get("chapter")),
                    normalize_scalar(row.get("section")),
                    normalize_scalar(row.get("formula_latex")),
                    json_dumps(row.get("variables")),
                    normalize_scalar(row.get("meaning")),
                    normalize_scalar(row.get("applicable_conditions")),
                    related_chunk_ids_json(row),
                    "formula",
                    "",
                    search_text,
                ),
            )
        conn.execute("create index formulas_source_idx on formulas(source_id)")
        conn.execute("create index formulas_content_type_idx on formulas(content_type)")


def build_terms_db(path: Path, rows: list[dict[str, Any]]) -> None:
    with recreate_database(path) as conn:
        conn.execute(
            """
            create table terms (
                term text primary key,
                english_term text,
                definition text,
                source_id text not null,
                chapter text,
                section text,
                related_chunk_ids_json text,
                content_type text not null,
                domain_tags_text text not null,
                search_text text not null
            )
            """
        )
        for row in rows:
            search_text = " ".join(
                [
                    normalize_scalar(row.get("term")),
                    normalize_scalar(row.get("english_term")),
                    normalize_scalar(row.get("definition")),
                    normalize_scalar(row.get("source_id")),
                    normalize_scalar(row.get("chapter")),
                    normalize_scalar(row.get("section")),
                ]
            )
            conn.execute(
                "insert into terms values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    normalize_scalar(row.get("term")),
                    normalize_scalar(row.get("english_term")),
                    normalize_scalar(row.get("definition")),
                    normalize_scalar(row.get("source_id")),
                    normalize_scalar(row.get("chapter")),
                    normalize_scalar(row.get("section")),
                    related_chunk_ids_json(row),
                    "term",
                    "",
                    search_text,
                ),
            )
        conn.execute("create index terms_source_idx on terms(source_id)")
        conn.execute("create index terms_content_type_idx on terms(content_type)")


def build_propagation_models_db(path: Path, rows: list[dict[str, Any]]) -> None:
    with recreate_database(path) as conn:
        conn.execute(
            """
            create table propagation_models (
                model_id text primary key,
                source_id text not null,
                model_name text,
                frequency_range text,
                scenario text,
                los_or_nlos text,
                input_parameters_json text,
                output_parameters_json text,
                formula_latex text,
                assumptions text,
                limitations text,
                related_chunk_ids_json text,
                content_type text not null,
                domain_tags_text text not null,
                search_text text not null
            )
            """
        )
        for row in rows:
            domain_tags_text = "itu_r_p1411 propagation_model"
            search_text = " ".join(
                [
                    normalize_scalar(row.get("model_id")),
                    normalize_scalar(row.get("source_id")),
                    normalize_scalar(row.get("model_name")),
                    normalize_scalar(row.get("frequency_range")),
                    normalize_scalar(row.get("scenario")),
                    normalize_scalar(row.get("los_or_nlos")),
                    json_dumps(row.get("input_parameters")),
                    json_dumps(row.get("output_parameters")),
                    normalize_scalar(row.get("formula_latex")),
                    normalize_scalar(row.get("assumptions")),
                    normalize_scalar(row.get("limitations")),
                    domain_tags_text,
                ]
            )
            conn.execute(
                "insert into propagation_models values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    normalize_scalar(row.get("model_id")),
                    normalize_scalar(row.get("source_id")),
                    normalize_scalar(row.get("model_name")),
                    normalize_scalar(row.get("frequency_range")),
                    normalize_scalar(row.get("scenario")),
                    normalize_scalar(row.get("los_or_nlos")),
                    json_dumps(row.get("input_parameters")),
                    json_dumps(row.get("output_parameters")),
                    normalize_scalar(row.get("formula_latex")),
                    normalize_scalar(row.get("assumptions")),
                    normalize_scalar(row.get("limitations")),
                    related_chunk_ids_json(row),
                    "standard_model",
                    domain_tags_text,
                    search_text,
                ),
            )
        conn.execute("create index propagation_source_idx on propagation_models(source_id)")
        conn.execute("create index propagation_content_type_idx on propagation_models(content_type)")
        conn.execute("create index propagation_domain_tags_idx on propagation_models(domain_tags_text)")


def write_report(project_root: Path, formula_rows: int, term_rows: int, propagation_rows: int) -> None:
    build_root = project_root / "kb_corpus_build"
    structured_dir = build_root / "indexes" / "structured"
    report_path = build_root / "reports" / "structured_sqlite_build_report.md"
    lines = [
        "# Phase 11 Structured SQLite Build Report",
        "",
        "Phase status: completed",
        f"formula_rows: {formula_rows}",
        f"term_rows: {term_rows}",
        f"propagation_model_rows: {propagation_rows}",
        f"formula_db: {safe_rel(structured_dir / 'formula.sqlite', project_root)}",
        f"terms_db: {safe_rel(structured_dir / 'terms.sqlite', project_root)}",
        f"propagation_models_db: {safe_rel(structured_dir / 'propagation_models.sqlite', project_root)}",
    ]
    write_text_checked(report_path, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    build_root = project_root / "kb_corpus_build"
    corpus_dir = build_root / "corpus"
    structured_dir = build_root / "indexes" / "structured"

    formula_rows = load_jsonl(corpus_dir / "formula_index.jsonl")
    term_rows = load_jsonl(corpus_dir / "term_index.jsonl")
    propagation_rows = load_jsonl(corpus_dir / "propagation_model_index.jsonl")

    build_formula_db(structured_dir / "formula.sqlite", formula_rows)
    build_terms_db(structured_dir / "terms.sqlite", term_rows)
    build_propagation_models_db(structured_dir / "propagation_models.sqlite", propagation_rows)
    write_report(project_root, len(formula_rows), len(term_rows), len(propagation_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
