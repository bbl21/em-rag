#!/usr/bin/env python3
"""Phase-2 LaTeX extractor for the EM knowledge corpus build."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


STRUCTURE_COMMANDS = {
    "part": "part",
    "chapter": "chapter",
    "section": "section",
    "subsection": "subsection",
    "subsubsection": "subsubsection",
}
ROW_LEVELS = {"section", "subsection", "subsubsection"}
BLOCK_ENVS = {"equation", "equation*", "align", "align*", "figure", "table", "example", "definition", "theorem"}
KNOWN_TEXT_MACROS = {
    "emph",
    "textbf",
    "textit",
    "texttt",
    "underline",
    "mathrm",
    "mathbf",
    "mathit",
    "textrm",
    "textsc",
    "url",
    "footnote",
    "caption",
    "mbox",
}
SKIP_MACROS = {
    "label",
    "ref",
    "eqref",
    "pageref",
    "cite",
    "vfill",
    "break",
    "bigskip",
    "medskip",
    "smallskip",
    "noindent",
    "clearpage",
    "newpage",
    "thispagestyle",
    "fancyhf",
    "fancyfoot",
    "tableofcontents",
    "frontmatter",
    "mainmatter",
    "backmatter",
    "pagenumbering",
    "setcounter",
    "begingroup",
    "endgroup",
    "fontfamily",
    "selectfont",
}
SPECIAL_REPLACEMENTS = {
    "~": " ",
    r"\%": "%",
    r"\_": "_",
    r"\&": "&",
    r"\#": "#",
    r"\$": "$",
    r"\{": "{",
    r"\}": "}",
}


@dataclass
class SectionRow:
    source_id: str
    source_title: str
    source_type: str
    raw_path: str
    tex_file: str
    part: str | None
    chapter: str | None
    section: str | None
    subsection: str | None
    subsubsection: str | None
    content_parts: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)

    def has_content(self) -> bool:
        if self.content_parts:
            return True
        return bool(self.equations or self.figures or self.tables)

    def to_record(self) -> dict[str, Any]:
        content_md = normalize_text("\n\n".join(part for part in self.content_parts if part))
        return {
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_type": self.source_type,
            "raw_path": self.raw_path,
            "tex_file": self.tex_file,
            "part": self.part,
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "subsubsection": self.subsubsection,
            "content_md": content_md,
            "equations": self.equations,
            "figures": self.figures,
            "tables": self.tables,
            "quality_flags": sorted(set(self.quality_flags)),
        }


class LatexExtractor:
    def __init__(self, project_root: Path, build_root: Path) -> None:
        self.project_root = project_root
        self.build_root = build_root
        self.warnings: list[str] = []
        self.source_reports: list[dict[str, Any]] = []
        self.processed_latex_books = 0
        self._warned_messages: set[tuple[str, str, str]] = set()
        self._relative_path_cache: dict[Path, str] = {}
        self._sources_by_file: dict[Path, dict[str, Any]] = {}

    def extract_all(self, sources: list[dict[str, Any]]) -> int:
        processed = 0
        for source in sources:
            if source.get("source_type") != "latex_book":
                continue
            main_file_rel = source.get("detected_main_file")
            if not main_file_rel:
                self.source_reports.append(
                    {
                        "source_id": source["source_id"],
                        "sections_path": str(
                            (self.build_root / "intermediate" / "latex" / f"{source['source_id']}.sections.jsonl").as_posix()
                        ),
                        "cleaned_path": str((self.build_root / "corpus" / "by_source" / f"{source['source_id']}.cleaned.md").as_posix()),
                        "sections": 0,
                        "equations": 0,
                        "figures": 0,
                        "tables": 0,
                        "output_written": False,
                        "status": "missing_detected_main_file",
                    }
                )
                continue
            self.extract_source(source, self.project_root / main_file_rel)
            processed += 1
            self.processed_latex_books += 1
        self.write_warnings()
        self.write_report()
        return processed

    def extract_source(self, source: dict[str, Any], main_file: Path) -> None:
        source_root = self.project_root / source["raw_path"]
        sections_path = self.build_root / "intermediate" / "latex" / f"{source['source_id']}.sections.jsonl"
        cleaned_path = self.build_root / "corpus" / "by_source" / f"{source['source_id']}.cleaned.md"
        sections_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)

        state = ParserState(self, source, source_root)
        state.parse_file(main_file)
        state.finalize_current()
        self._write_sections(sections_path, state.records)
        self._write_cleaned(cleaned_path, state.records, source["title"])

        section_count = len(state.records)
        equation_count = sum(len(row.get("equations", [])) for row in state.records)
        figure_count = sum(len(row.get("figures", [])) for row in state.records)
        table_count = sum(len(row.get("tables", [])) for row in state.records)

        self.source_reports.append(
            {
                "source_id": source["source_id"],
                "sections_path": sections_path.as_posix(),
                "cleaned_path": cleaned_path.as_posix(),
                "sections": section_count,
                "equations": equation_count,
                "figures": figure_count,
                "tables": table_count,
                "output_written": True,
                "status": "ok",
            }
        )

    def warn(self, source_id: str, path: Path, message: str) -> None:
        rel = self._relative_path(path)
        key = (source_id, rel, message)
        if key in self._warned_messages:
            return
        self._warned_messages.add(key)
        self.warnings.append(f"[{source_id}] {rel}: {message}")

    def _relative_path(self, path: Path) -> str:
        if path in self._relative_path_cache:
            return self._relative_path_cache[path]
        rel = safe_rel(path, self.project_root)
        self._relative_path_cache[path] = rel
        return rel

    def write_warnings(self) -> None:
        log_path = self.build_root / "logs" / "latex_warnings.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(self.warnings)
        log_path.write_text(text + ("\n" if text else ""), encoding="utf-8", newline="\n")
        _ = log_path.read_text(encoding="utf-8")

    def write_report(self) -> None:
        report_path = self.build_root / "reports" / "latex_extraction_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        warnings_count = len(self.warnings)
        all_outputs_written = all(item.get("output_written", False) for item in self.source_reports if item.get("sections") is not None)
        if self.processed_latex_books >= 1 and all_outputs_written:
            phase_status = "completed"
        elif self.source_reports:
            phase_status = "completed_with_warnings" if warnings_count else "completed"
        else:
            phase_status = "blocked"

        lines = [
            "# Phase 2 LaTeX Extraction Report",
            "",
            f"Phase status: {phase_status}",
            f"warnings: {warnings_count}",
            f"warnings_log: { (self.build_root / 'logs' / 'latex_warnings.log').as_posix() }",
            "",
            "## Source outputs",
            "",
        ]

        for report in self.source_reports:
            lines.extend(
                [
                    f"### {report['source_id']}",
                    f"- sections: {report['sections']}",
                    f"- equations: {report['equations']}",
                    f"- figures: {report['figures']}",
                    f"- tables: {report['tables']}",
                    f"- sections_jsonl: {report['sections_path']}",
                    f"- cleaned_markdown: {report['cleaned_path']}",
                    f"- output_written: {report['output_written']}",
                    f"- status: {report['status']}",
                    "",
                ]
            )

        if not self.source_reports:
            lines.extend(["No latex_book sources were processed.", ""])

        report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        _ = report_path.read_text(encoding="utf-8")

    def _write_sections(self, path: Path, records: list[dict[str, Any]]) -> None:
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
        _ = path.read_text(encoding="utf-8")

    def _write_cleaned(self, path: Path, records: list[dict[str, Any]], title: str) -> None:
        lines: list[str] = [f"# {title}"]
        seen: dict[str, str | None] = {"part": None, "chapter": None, "section": None, "subsection": None, "subsubsection": None}
        for record in records:
            for field_name, prefix in (
                ("part", "#"),
                ("chapter", "#"),
                ("section", "##"),
                ("subsection", "###"),
                ("subsubsection", "####"),
            ):
                value = record.get(field_name)
                if value and seen[field_name] != value:
                    if field_name == "part":
                        lines.extend(["", f"# {value}"])
                    else:
                        lines.extend(["", f"{prefix} {value}"])
                    seen[field_name] = value
                    if field_name == "chapter":
                        seen["section"] = None
                        seen["subsection"] = None
                        seen["subsubsection"] = None
                    elif field_name == "section":
                        seen["subsection"] = None
                        seen["subsubsection"] = None
                    elif field_name == "subsection":
                        seen["subsubsection"] = None
            if record["content_md"]:
                lines.extend(["", record["content_md"]])
        text = normalize_text("\n".join(lines).strip()) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")
        _ = path.read_text(encoding="utf-8")


class ParserState:
    def __init__(self, extractor: LatexExtractor, source: dict[str, Any], source_root: Path) -> None:
        self.extractor = extractor
        self.source = source
        self.source_root = source_root
        self.records: list[dict[str, Any]] = []
        self.structure: dict[str, str | None] = {
            "part": None,
            "chapter": None,
            "section": None,
            "subsection": None,
            "subsubsection": None,
        }
        self.current_row: SectionRow | None = None
        self.stack: list[Path] = []

    def parse_file(self, path: Path) -> None:
        normalized = path.resolve()
        if normalized in self.stack:
            self.extractor.warn(self.source["source_id"], path, "cyclic include detected")
            return
        if not path.exists():
            self.extractor.warn(self.source["source_id"], path, "missing include target")
            return

        self.stack.append(normalized)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.extractor.warn(self.source["source_id"], path, "utf-8 decode fallback with ignored errors")
        self._parse_text(text, path)
        self.stack.pop()

    def finalize_current(self) -> None:
        if self.current_row and self.current_row.has_content():
            self.records.append(self.current_row.to_record())
        self.current_row = None

    def _parse_text(self, text: str, current_file: Path) -> None:
        text = strip_comments(text)
        pos = 0
        while pos < len(text):
            match = re.search(
                r"\\(part|chapter|section|subsection|subsubsection)\*?\s*\{|\\(input|include)\s*\{|\\begin\s*\{",
                text[pos:],
            )
            if not match:
                self._append_plain(text[pos:], current_file)
                break
            start = pos + match.start()
            if start > pos:
                self._append_plain(text[pos:start], current_file)
            token = match.group(0)
            if token.startswith(r"\begin"):
                env_name, env_inner, end_pos = extract_environment(text, start)
                if env_name is None:
                    self._append_plain(text[start:start + 6], current_file)
                    pos = start + 6
                    continue
                self._handle_environment(env_name, env_inner, current_file)
                pos = end_pos
                continue

            command_match = re.match(r"\\([A-Za-z]+)\*?\s*\{", text[start:])
            if not command_match:
                self._append_plain(text[start:start + 1], current_file)
                pos = start + 1
                continue
            command = command_match.group(1)
            brace_start = start + command_match.end() - 1
            value, brace_end = read_braced(text, brace_start)
            if value is None:
                self.extractor.warn(self.source["source_id"], current_file, f"failed to parse argument for \\{command}")
                pos = start + len(command_match.group(0))
                continue
            if command in STRUCTURE_COMMANDS:
                self._handle_structure(command, clean_inline_text(value, self.extractor, self.source["source_id"], current_file))
            else:
                self._handle_include(command, value, current_file)
            pos = brace_end

    def _handle_structure(self, command: str, title: str) -> None:
        field_name = STRUCTURE_COMMANDS[command]
        if field_name in ROW_LEVELS:
            self.finalize_current()
        self.structure[field_name] = title or None
        if field_name == "part":
            self.structure["chapter"] = None
            self.structure["section"] = None
            self.structure["subsection"] = None
            self.structure["subsubsection"] = None
        elif field_name == "chapter":
            self.structure["section"] = None
            self.structure["subsection"] = None
            self.structure["subsubsection"] = None
        elif field_name == "section":
            self.structure["subsection"] = None
            self.structure["subsubsection"] = None
        elif field_name == "subsection":
            self.structure["subsubsection"] = None

        if field_name in ROW_LEVELS:
                self.current_row = SectionRow(
                    source_id=self.source["source_id"],
                    source_title=self.source["title"],
                    source_type=self.source["source_type"],
                    raw_path=self.source["raw_path"],
                    tex_file=self.extractor._relative_path(self.source_root),
                    part=self.structure["part"],
                    chapter=self.structure["chapter"],
                    section=self.structure["section"],
                    subsection=self.structure["subsection"],
                    subsubsection=self.structure["subsubsection"],
            )

    def _handle_include(self, command: str, target: str, current_file: Path) -> None:
        include_path = resolve_include_path(target, current_file)
        if include_path is None:
            self.extractor.warn(self.source["source_id"], current_file, f"empty {command} target")
            return
        if not include_path.exists():
            self.extractor.warn(self.source["source_id"], include_path, f"missing {command} target")
            if self.current_row:
                self.current_row.quality_flags.append("missing_include")
            return
        self.parse_file(include_path)

    def _handle_environment(self, env_name: str, inner: str, current_file: Path) -> None:
        if env_name.strip() == "document":
            self._parse_text(inner, current_file)
            return

        target = self._ensure_row(current_file)
        normalized_env = env_name.strip()
        if normalized_env not in BLOCK_ENVS:
            rendered = clean_inline_text(inner, self.extractor, self.source["source_id"], current_file)
            if rendered:
                target.content_parts.append(rendered)
            return

        if normalized_env.startswith("equation") or normalized_env.startswith("align"):
            equation = normalize_text(inner)
            if equation:
                target.equations.append(equation)
                target.content_parts.append(f"$$\n{equation}\n$$")
            return

        if normalized_env == "figure":
            caption = extract_caption(inner, self.extractor, self.source["source_id"], current_file)
            if caption:
                target.figures.append(caption)
                target.content_parts.append(f"Figure: {caption}")
            else:
                target.quality_flags.append("figure_without_caption")
            return

        if normalized_env == "table":
            caption = extract_caption(inner, self.extractor, self.source["source_id"], current_file)
            if caption:
                target.tables.append(caption)
                target.content_parts.append(f"Table: {caption}")
            else:
                rendered = clean_inline_text(inner, self.extractor, self.source["source_id"], current_file)
                if rendered:
                    target.content_parts.append(f"Table:\n{rendered}")
            return

        label = normalized_env.capitalize()
        rendered = clean_inline_text(inner, self.extractor, self.source["source_id"], current_file)
        if rendered:
            target.content_parts.append(f"{label}: {rendered}")

    def _append_plain(self, fragment: str, current_file: Path) -> None:
        rendered = clean_inline_text(fragment, self.extractor, self.source["source_id"], current_file)
        if not rendered:
            return
        if self.current_row is None and not any(self.structure.values()):
            return
        target = self._ensure_row(current_file)
        target.content_parts.append(rendered)

    def _ensure_row(self, current_file: Path) -> SectionRow:
        if self.current_row is None:
            title = self.structure["section"] or self.structure["chapter"] or self.structure["part"] or "Untitled"
            section_value = self.structure["section"] or title
            self.current_row = SectionRow(
                source_id=self.source["source_id"],
                source_title=self.source["title"],
                source_type=self.source["source_type"],
                raw_path=self.source["raw_path"],
                tex_file=self.extractor._relative_path(current_file),
                part=self.structure["part"],
                chapter=self.structure["chapter"],
                section=section_value,
                subsection=self.structure["subsection"],
                subsubsection=self.structure["subsubsection"],
            )
        elif self.current_row.tex_file == self.extractor._relative_path(self.source_root):
            self.current_row.tex_file = self.extractor._relative_path(current_file)
        return self.current_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured text from LaTeX sources.")
    parser.add_argument("--project-root", default=".", help="Project root containing kb_corpus_build and reference.")
    return parser.parse_args()


def load_sources(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = data.get("sources")
    if not isinstance(sources, list):
        raise ValueError("sources.yaml missing top-level 'sources' list")
    return [item for item in sources if isinstance(item, dict)]


def resolve_include_path(target: str, current_file: Path) -> Path | None:
    cleaned = target.strip()
    if not cleaned:
        return None
    if cleaned.endswith(".tex"):
        return (current_file.parent / cleaned).resolve()
    return (current_file.parent / f"{cleaned}.tex").resolve()


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        escaped = False
        result_chars: list[str] = []
        for char in line:
            if char == "%" and not escaped:
                break
            result_chars.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(result_chars))
    return "\n".join(lines)


def extract_environment(text: str, start: int) -> tuple[str | None, str, int]:
    begin_match = re.match(r"\\begin\s*\{([^}]+)\}", text[start:])
    if not begin_match:
        return None, "", start + 1
    env_name = begin_match.group(1)
    content_start = start + begin_match.end()
    end_pattern = re.compile(rf"\\end\s*\{{{re.escape(env_name)}}}")
    end_match = end_pattern.search(text, content_start)
    if not end_match:
        return env_name, text[content_start:], len(text)
    return env_name, text[content_start:end_match.start()], end_match.end()


def read_braced(text: str, brace_start: int) -> tuple[str | None, int]:
    if brace_start >= len(text) or text[brace_start] != "{":
        return None, brace_start
    depth = 0
    chars: list[str] = []
    index = brace_start
    while index < len(text):
        char = text[index]
        if char == "{" and not is_escaped(text, index):
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == "}" and not is_escaped(text, index):
            depth -= 1
            if depth == 0:
                return "".join(chars), index + 1
            chars.append(char)
        else:
            chars.append(char)
        index += 1
    return None, len(text)


def is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def extract_caption(text: str, extractor: LatexExtractor, source_id: str, current_file: Path) -> str:
    match = re.search(r"\\caption\s*\{", text)
    if not match:
        return ""
    caption, _ = read_braced(text, match.end() - 1)
    if caption is None:
        extractor.warn(source_id, current_file, "failed to parse figure/table caption")
        return ""
    return clean_inline_text(caption, extractor, source_id, current_file)


def clean_inline_text(text: str, extractor: LatexExtractor, source_id: str, current_file: Path) -> str:
    math_segments: list[str] = []

    def find_matching_inline_math_end(start: int) -> int | None:
        index = start + 1
        while index < len(text):
            if text[index] == "$" and not is_escaped(text, index):
                return index
            index += 1
        return None

    def find_matching_escaped_math_end(start: int, delimiter: str) -> int | None:
        index = start
        close = "\\" + delimiter
        while index + 1 < len(text):
            if text[index:index + 2] == close and not is_escaped(text, index):
                return index + 1
            index += 1
        return None

    index = 0
    output: list[str] = []
    placeholder_map: dict[str, str] = {}

    # Keep inline math expressions unchanged so LaTeX macros inside them are preserved.
    while index < len(text):
        if text[index] == "$" and not is_escaped(text, index):
            end = find_matching_inline_math_end(index)
            if end is not None:
                placeholder = f"__INLINE_MATH_{len(math_segments)}__"
                math_segments.append(text[index : end + 1])
                placeholder_map[placeholder] = text[index : end + 1]
                output.append(f"<<{placeholder}>>")
                index = end + 1
                continue
            output.append(text[index])
            index += 1
            continue

        if text[index : index + 2] == "\\(" and not is_escaped(text, index):
            end = find_matching_escaped_math_end(index + 2, ")")
            if end is not None:
                placeholder = f"__INLINE_MATH_{len(math_segments)}__"
                math_segments.append(text[index : end + 1])
                placeholder_map[placeholder] = text[index : end + 1]
                output.append(f"<<{placeholder}>>")
                index = end + 1
                continue
            output.append(text[index])
            index += 1
            continue

        if text[index : index + 2] == "\\[" and not is_escaped(text, index):
            end = find_matching_escaped_math_end(index + 2, "]")
            if end is not None:
                placeholder = f"__INLINE_MATH_{len(math_segments)}__"
                math_segments.append(text[index : end + 1])
                placeholder_map[placeholder] = text[index : end + 1]
                output.append(f"<<{placeholder}>>")
                index = end + 1
                continue
            output.append(text[index])
            index += 1
            continue

        output.append(text[index])
        index += 1
    text = "".join(output)

    for src, dst in SPECIAL_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.replace("\\\\", "\n")

    command_pattern = re.compile(r"\\([A-Za-z]+)(\*?)")
    index = 0
    output: list[str] = []
    while index < len(text):
        if text[index] != "\\":
            output.append(text[index])
            index += 1
            continue

        if index + 1 < len(text) and text[index:index + 2] in SPECIAL_REPLACEMENTS:
            output.append(SPECIAL_REPLACEMENTS[text[index:index + 2]])
            index += 2
            continue

        match = command_pattern.match(text, index)
        if not match:
            output.append(text[index])
            index += 1
            continue

        name = match.group(1)
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor < len(text) and text[cursor] == "[":
            _, cursor = read_optional(text, cursor)
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1

        if name in KNOWN_TEXT_MACROS:
            if cursor < len(text) and text[cursor] == "{":
                inner, cursor = read_braced(text, cursor)
                if inner is not None:
                    output.append(clean_inline_text(inner, extractor, source_id, current_file))
                else:
                    extractor.warn(source_id, current_file, f"failed to parse macro argument for \\{name}")
            index = cursor
            continue

        if name == "href":
            if cursor < len(text) and text[cursor] == "{":
                _, cursor = read_braced(text, cursor)
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                if cursor < len(text) and text[cursor] == "{":
                    inner, cursor = read_braced(text, cursor)
                    if inner is not None:
                        output.append(clean_inline_text(inner, extractor, source_id, current_file))
            index = cursor
            continue

        if name in SKIP_MACROS:
            if cursor < len(text) and text[cursor] == "{":
                _, cursor = read_braced(text, cursor)
            index = cursor
            continue

        if cursor < len(text) and text[cursor] == "{":
            inner, cursor = read_braced(text, cursor)
            if inner is not None:
                extractor.warn(source_id, current_file, f"unknown macro \\{name} unwrapped")
                output.append(clean_inline_text(inner, extractor, source_id, current_file))
                index = cursor
                continue

        extractor.warn(source_id, current_file, f"unknown macro \\{name} removed")
        index = cursor

    cleaned = normalize_text("".join(output))
    for placeholder, segment in placeholder_map.items():
        cleaned = cleaned.replace(f"<<{placeholder}>>", segment)
    return cleaned


def read_optional(text: str, bracket_start: int) -> tuple[str | None, int]:
    if bracket_start >= len(text) or text[bracket_start] != "[":
        return None, bracket_start
    depth = 0
    chars: list[str] = []
    index = bracket_start
    while index < len(text):
        char = text[index]
        if char == "[" and not is_escaped(text, index):
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == "]" and not is_escaped(text, index):
            depth -= 1
            if depth == 0:
                return "".join(chars), index + 1
            chars.append(char)
        else:
            chars.append(char)
        index += 1
    return None, len(text)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    build_root = project_root / "kb_corpus_build"
    sources_path = build_root / "metadata" / "sources.yaml"
    if not sources_path.is_file():
        raise FileNotFoundError(f"Missing sources file: {sources_path}")

    extractor = LatexExtractor(project_root, build_root)
    sources = load_sources(sources_path)
    extractor.extract_all(sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
