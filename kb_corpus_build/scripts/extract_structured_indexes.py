#!/usr/bin/env python3
"""Phase-8 deterministic structured index extraction."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


VARIABLE_TOKEN_RE = re.compile(r"\b[A-Za-z](?:[A-Za-z0-9_]{0,15})\b")
FREQUENCY_RANGE_RE = re.compile(
    r"\b(?:from\s+|between\s+)?"
    r"(?P<start>\d+(?:[\s,]\d{3})*(?:\.\d+)?)\s*"
    r"(?P<start_unit>Hz|kHz|MHz|GHz|THz)?\s+"
    r"(?P<separator>to|and)\s+"
    r"(?P<end>\d+(?:[\s,]\d{3})*(?:\.\d+)?)\s*"
    r"(?P<end_unit>Hz|kHz|MHz|GHz|THz)\b",
    re.IGNORECASE,
)
TERM_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b)\s*", re.IGNORECASE)

STOP_VARIABLES = {
    "a",
    "an",
    "and",
    "begin",
    "cos",
    "cdot",
    "dfrac",
    "end",
    "exp",
    "for",
    "frac",
    "if",
    "in",
    "is",
    "label",
    "left",
    "log",
    "log10",
    "max",
    "mathrm",
    "min",
    "of",
    "or",
    "right",
    "sqrt",
    "sin",
    "tan",
    "the",
    "to",
    "where",
}
PARAMETER_NAME_STOPWORDS = STOP_VARIABLES | {
    "as",
    "be",
    "by",
    "components",
    "follows",
    "from",
    "include",
    "includes",
    "into",
    "parts",
    "using",
    "with",
}
GREEK_VARIABLE_COMMANDS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "theta",
    "lambda",
    "mu",
    "nu",
    "rho",
    "sigma",
    "tau",
    "phi",
    "varphi",
    "omega",
}
NOISY_TERMS = {
    "chapter 1",
    "image credits",
    "contents",
    "copyright",
    "figure",
    "index",
    "introduction",
    "navigation",
    "references",
    "table of contents",
    "unknown",
}
TERM_BLACKLIST_EXACT = {"image", "credits", "page", "section", "chapter"}
NOISY_TERM_PREFIXES = (
    "about ",
    "chapter ",
    "review of",
    "introduction",
    "scope",
    "recognizing",
    "recommends",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract deterministic structured indexes.")
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
        items = [normalize_scalar(item) for item in value]
        return [item for item in items if item]
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


def write_jsonl_checked(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows]
    text = "\n".join(lines) + ("\n" if lines else "")
    write_text_checked(path, text)


def sentence_split(text: str) -> list[str]:
    normalized = normalize_content_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def is_itu_standard_model(row: dict[str, Any]) -> bool:
    return (
        normalize_scalar(row.get("source_id")).lower() == "itu_r_p1411_13"
        and normalize_scalar(row.get("content_type")).lower() == "standard_model"
    )


def is_formula_source(row: dict[str, Any]) -> bool:
    source_type = normalize_scalar(row.get("source_type")).lower()
    source_id = normalize_scalar(row.get("source_id")).lower()
    return source_type in {"latex_book", "pdf_course"} and source_id != "itu_r_p1411_13"


def extract_variable_tokens(formula_latex: str) -> list[str]:
    formula_without_labels_units = re.sub(r"\\(?:label|mbox|text|mathrm)\{[^{}]*\}", " ", formula_latex)
    command_variables = [
        match.group(1)
        for match in re.finditer(r"\\([A-Za-z]+)", formula_without_labels_units)
        if match.group(1).lower() in GREEK_VARIABLE_COMMANDS
    ]
    formula_without_commands = re.sub(r"\\[A-Za-z]+", " ", formula_without_labels_units)
    variables: list[str] = []
    seen: set[str] = set()
    for token in command_variables:
        if token not in seen:
            seen.add(token)
            variables.append(token)
    for match in VARIABLE_TOKEN_RE.finditer(formula_without_commands):
        token = match.group(0)
        if token.lower() in STOP_VARIABLES:
            continue
        if token not in seen:
            seen.add(token)
            variables.append(token)
    return variables


def extract_variable_meanings(content: str, variables: list[str]) -> dict[str, str]:
    sentences = sentence_split(content)
    meanings: dict[str, str] = {}
    for variable in variables:
        description = ""
        patterns = [
            rf"\b{re.escape(variable)}\b\s+(?:is|denotes|represents|means|equals)\s+([^.;,]+(?:\s+[^.;,]+)*)",
            rf"\bfor\s+\b{re.escape(variable)}\b\s*,?\s*([^.;,]+(?:\s+[^.;,]+)*)",
        ]
        for sentence in sentences:
            for pattern in patterns:
                match = re.search(pattern, sentence, flags=re.IGNORECASE)
                if match:
                    description = normalize_scalar(match.group(1))
                    description = re.split(
                        r"\s+\band\b\s+[A-Za-z][A-Za-z0-9_]*\s+(?:is|denotes|represents|means|equals)\s+",
                        description,
                        maxsplit=1,
                        flags=re.IGNORECASE,
                    )[0].strip()
                    break
            if description:
                break
        meanings[variable] = description or "unknown"
    return meanings


def extract_formula_meaning(content: str, keywords: list[str], section: str) -> str:
    sentences = sentence_split(content)
    candidate_terms = [term.lower() for term in keywords if term]
    if not candidate_terms:
        return ""
    for sentence in sentences:
        lower = sentence.lower()
        if len(sentence) > 240 or "\\" in sentence or "\n" in sentence or "=" in sentence:
            continue
        if any(term in lower for term in candidate_terms) and re.search(r"\b(is|refers to|defines|means)\b", lower):
            return sentence
    return ""


def extract_formula_conditions(content: str) -> str:
    for sentence in sentence_split(content):
        if "?" in sentence or "\\" in sentence or len(sentence) > 240:
            continue
        if re.search(r"\b(applies to|valid for|used for)\b", sentence, flags=re.IGNORECASE):
            return sentence
    return ""


def normalize_term(term: str) -> str:
    term = normalize_scalar(term)
    term = term.strip(" .,:;()[]{}")
    term = re.sub(r"^\d+(?:\.\d+)*\s+", "", term)
    return re.sub(r"\s+", " ", term)


def is_term_noise(term: str) -> bool:
    lowered = normalize_term(term).lower()
    if not lowered:
        return True
    if lowered in NOISY_TERMS or lowered in TERM_BLACKLIST_EXACT:
        return True
    if any(lowered.startswith(prefix) for prefix in NOISY_TERM_PREFIXES):
        return True
    if "\\" in term or "$" in term:
        return True
    if term.count("(") != term.count(")") or term.count("{") != term.count("}"):
        return True
    if re.search(r"(?:\.{3,}|…{2,})\s*\d*$", term):
        return True
    if lowered.startswith("image credits") or lowered.startswith("figure ") or lowered.startswith("table "):
        return True
    if lowered.isdigit():
        return True
    if len(lowered) < 2:
        return True
    if re.fullmatch(r"[\W_]+", lowered):
        return True
    return False


def extract_term_candidates(row: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for term in normalize_list(row.get("keywords")):
        normalized = normalize_term(term)
        if not is_term_noise(normalized):
            candidates.append(normalized)
    section = normalize_term(row.get("section"))
    if section and not is_term_noise(section):
        candidates.append(section)
    chapter = normalize_term(row.get("chapter"))
    if chapter and not is_term_noise(chapter):
        candidates.append(chapter)

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def extract_term_definition(term: str, content: str) -> str:
    sentences = sentence_split(content)
    escaped = re.escape(term)
    patterns = [
        rf"^(?:the|a|an)?\s*\b{escaped}\b\s+(?:is|are|refers to|means|equals)\s+(.+?)(?:[.;]|$)",
        rf"^(?:the|a|an)?\s*\b{escaped}\b\s*[:\-]\s*(.+?)(?:[.;]|$)",
    ]
    for sentence in sentences:
        for pattern in patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE | re.DOTALL)
            if match:
                definition = normalize_scalar(match.group(1))
                if re.search(r"\b(not required|shown in fig|shown in figure|significantly greater)\b", definition, flags=re.IGNORECASE):
                    return ""
                if definition:
                    return definition
    return ""


def build_english_term(term: str) -> str:
    ascii_letters = re.sub(r"[^A-Za-z0-9 _./-]", "", term)
    if ascii_letters and ascii_letters == term:
        return term
    return ""


def first_non_empty(values: Iterable[str]) -> str:
    for value in values:
        normalized = normalize_scalar(value)
        if normalized:
            return normalized
    return ""


def extract_frequency_range(content: str) -> str:
    match = FREQUENCY_RANGE_RE.search(content)
    if not match:
        return "unknown"
    start = normalize_scalar(match.group("start"))
    end = normalize_scalar(match.group("end"))
    end_unit = normalize_scalar(match.group("end_unit"))
    start_unit = normalize_scalar(match.group("start_unit")) or end_unit
    return f"{start} {start_unit} to {end} {end_unit}"


def extract_los_or_nlos(content: str) -> str:
    has_nlos = re.search(r"\bNLoS\b", content, flags=re.IGNORECASE) is not None
    has_los = re.search(r"\bLoS\b", content, flags=re.IGNORECASE) is not None
    if has_los and has_nlos:
        return "LoS/NLoS"
    if has_los:
        return "LoS"
    if has_nlos:
        return "NLoS"
    return "unknown"


def extract_scenario(content: str, section: str) -> str:
    sentences = sentence_split(content)
    keywords = ("scenario", "urban", "suburban", "rural", "indoor", "outdoor", "microcell", "macrocell", "street")
    for sentence in sentences:
        lower = sentence.lower()
        if is_noisy_model_name(sentence) or re.search(
            r"\b(table|figure|fig\.|shown in fig|listed in table|expressed as follows)\b",
            lower,
        ):
            continue
        if len(sentence) > 260 or "\\" in sentence or "=" in sentence:
            continue
        if any(keyword in lower for keyword in keywords):
            return sentence
    if section and not is_noisy_model_name(section):
        return normalize_scalar(section)
    return "unknown"


def split_parameter_names(text: str) -> list[str]:
    colon_names = re.findall(r"(?:^|\s|[;,\-–])([A-Za-z][A-Za-z0-9_α-ω]*)\s*:", text)
    if colon_names:
        return [normalize_scalar(name) for name in colon_names if normalize_scalar(name)]

    names: list[str] = []
    for piece in TERM_SPLIT_RE.split(text):
        name = normalize_scalar(piece).strip(".:")
        if not name:
            continue
        name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.IGNORECASE)
        if name.lower() in {"include", "includes", "is", "are"}:
            continue
        names.append(name)
    return names


def normalize_parameter_name(name: str) -> str:
    normalized = normalize_scalar(name).strip(".:,;")
    normalized = re.sub(r"^(?:the|a|an)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*\([^)]*\)\s*$", "", normalized)
    parts = normalized.split()
    if len(parts) >= 2:
        suffix = parts[-1]
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,3}", suffix) and len(suffix) <= 3:
            normalized = " ".join(parts[:-1])
    return normalize_scalar(normalized)


def extract_parameter_dict(content: str, prefix: str) -> dict[str, str]:
    patterns = [
        rf"\b{prefix}\s+parameters?\s+include\s+(.+?)(?:[.;]|$)",
        rf"\b{prefix}\s+parameters?\s+are\s+(.+?)(?:[.;]|$)",
        rf"\b{prefix}\s+parameter\s+is\s+(.+?)(?:[.;]|$)",
    ]
    if prefix == "input":
        patterns.append(r"\brelevant\s+parameters?(?:\s+for\s+this\s+(?:situation|model|case))?\s+are\s*:?\s+(.+?)(?:[.;]|$)")
    for sentence in sentence_split(content):
        for pattern in patterns:
            match = re.search(pattern, sentence, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            params: list[str] = []
            for raw_name in split_parameter_names(match.group(1)):
                name = normalize_parameter_name(raw_name)
                if name:
                    params.append(name)
            return {name: "unknown" for name in params}
    return {}


def extract_assumptions(content: str) -> str:
    sentences = sentence_split(content)
    for sentence in sentences:
        if re.search(r"^\s*Assumptions?\s*:", sentence, flags=re.IGNORECASE):
            return sentence
    for sentence in sentences:
        if re.search(r"\b(assume|assumes|assumed)\b", sentence, flags=re.IGNORECASE):
            return sentence
    for sentence in sentences:
        if re.search(r"\bapplicable to situations where\b", sentence, flags=re.IGNORECASE):
            return sentence
    for sentence in sentences:
        if re.search(
            r"\b(?:developed|derived|obtained)\s+(?:based on|from)\s+measure(?:d|ment)\s+data\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            return sentence
    for sentence in sentences:
        if re.search(r"\bsituation is depicted\b", sentence, flags=re.IGNORECASE):
            return sentence
    for sentence in sentences:
        if re.search(r"\bdistance-dependent model\b", sentence, flags=re.IGNORECASE):
            return sentence
    for sentence in sentences:
        if re.search(r"\bmodels? are defined for the situation\b", sentence, flags=re.IGNORECASE):
            return sentence
    return ""


def extract_limitations(content: str) -> str:
    normalized = normalize_content_text(content)
    valid_block = re.search(
        r"\b(?:the\s+)?models?\s+(?:are\s+)?valid for\s*:\s*(.+?)(?:\.\s*(?:Note that[^.]*\.)|$)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if valid_block:
        value = normalize_scalar(valid_block.group(0))
        if len(value) <= 500:
            return value
    sentences = sentence_split(content)
    for sentence in sentences:
        if re.search(r"^\s*Limitations?\s*:", sentence, flags=re.IGNORECASE):
            return sentence
    for sentence in sentences:
        if re.search(r"\b(?:based on|from)\s+measure(?:d|ment)\s+data\b", sentence, flags=re.IGNORECASE):
            continue
        if re.search(
            r"\b(valid for|valid distance range|limited to|not valid|only for|should only be used for|used only for|recommended for use|intended for distances|distances from \d+|range of .{0,80} defined as)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            return sentence
    return ""


def build_formula_index(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    formula_ordinals: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        if not is_formula_source(row):
            continue
        equations = normalize_list(row.get("equations"))
        if not equations:
            continue
        source_id = normalize_scalar(row.get("source_id"))
        chapter = normalize_scalar(row.get("chapter"))
        section = normalize_scalar(row.get("section"))
        content = normalize_content_text(row.get("content_md"))
        keywords = normalize_list(row.get("keywords"))
        chunk_id = normalize_scalar(row.get("chunk_id"))
        for equation in equations:
            key = (source_id, equation)
            if key not in merged:
                formula_ordinals[source_id] += 1
                variables = extract_variable_tokens(equation)
                merged[key] = {
                    "formula_id": f"formula:{source_id}:{formula_ordinals[source_id]:05d}",
                    "source_id": source_id,
                    "chapter": chapter,
                    "section": section,
                    "formula_latex": equation,
                    "variables": extract_variable_meanings(content, variables),
                    "meaning": extract_formula_meaning(content, keywords, section),
                    "applicable_conditions": extract_formula_conditions(content),
                    "related_chunk_ids": [chunk_id] if chunk_id else [],
                }
                continue
            if chunk_id and chunk_id not in merged[key]["related_chunk_ids"]:
                merged[key]["related_chunk_ids"].append(chunk_id)
    return sorted(merged.values(), key=lambda item: item["formula_id"])


def build_term_index(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        content = normalize_content_text(row.get("content_md"))
        chapter = normalize_scalar(row.get("chapter"))
        section = normalize_scalar(row.get("section"))
        source_id = normalize_scalar(row.get("source_id"))
        chunk_id = normalize_scalar(row.get("chunk_id"))
        for term in extract_term_candidates(row):
            key = term.lower()
            definition = extract_term_definition(term, content)
            if key not in merged:
                merged[key] = {
                    "term": term,
                    "english_term": build_english_term(term),
                    "definition": definition,
                    "source_id": source_id,
                    "chapter": chapter,
                    "section": section,
                    "related_chunk_ids": [chunk_id] if chunk_id else [],
                }
                continue
            if definition and not merged[key]["definition"]:
                merged[key]["definition"] = definition
            if chunk_id and chunk_id not in merged[key]["related_chunk_ids"]:
                merged[key]["related_chunk_ids"].append(chunk_id)
    return sorted(merged.values(), key=lambda item: (item["term"].lower(), item["source_id"], item["section"]))


def slugify_identifier(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_scalar(value).lower()).strip("-")
    return slug[:64] or "unknown"


def build_model_id(row: dict[str, Any], model_name: str) -> str:
    page = row.get("page_start")
    name_slug = slugify_identifier(model_name)
    if isinstance(page, int):
        return f"propagation_model:page:{page:05d}:{name_slug}"
    chunk_id = normalize_scalar(row.get("chunk_id")) or "unknown"
    return f"propagation_model:chunk:{slugify_identifier(chunk_id)}:{name_slug}"


def is_noisy_model_name(value: str) -> bool:
    text = normalize_scalar(value)
    lower = text.lower()
    if not text:
        return True
    if "..." in text or len(text.split()) > 12:
        return True
    if lower.startswith(("recognizing", "recommends", "noting", "considering", "table ", "figure ", "fig. ")):
        return True
    if (
        " lists " in lower
        or " is influenced " in lower
        or "listed in table" in lower
        or "shown in fig" in lower
        or "shown in figure" in lower
        or "expressed as follows" in lower
        or "does not depend" in lower
        or "lower bound is based" in lower
        or lower.startswith("values of ")
    ):
        return True
    if lower in NOISY_TERMS:
        return True
    return False


def extract_model_name(section: str, content: str) -> str:
    if section and not is_noisy_model_name(section):
        return section
    return "unknown"


def is_model_formula_candidate(value: str, *, allow_weak_operator: bool = True) -> bool:
    normalized = normalize_scalar(value)
    lower = normalized.lower()
    if not normalized:
        return False
    if normalized.startswith("="):
        return False
    if lower.startswith(("where ", "with ", "for ")):
        return False
    if lower.startswith(("(i.e", "i.e")) or " i.e." in lower:
        return False
    if not any(character.isalnum() for character in normalized):
        return False
    marker_pattern = r"[=+\-*/^≤≥<>]|\\" if allow_weak_operator else r"[=≤≥<>]|\\"
    if not re.search(marker_pattern, normalized):
        return False
    if re.search(r"\b(is|are|denotes|represents)\b", lower) and len(normalized.split()) > 12:
        return False
    return True


def extract_model_formula(equations: list[str], content: str = "") -> str:
    for equation in equations:
        normalized = normalize_scalar(equation)
        if is_model_formula_candidate(normalized):
            return normalized
    for line in normalize_content_text(content).splitlines():
        normalized = normalize_scalar(line)
        if len(normalized.split()) > 30:
            continue
        if is_model_formula_candidate(normalized, allow_weak_operator=False):
            return normalized
    return "unknown"


def is_model_candidate(row: dict[str, Any], content: str, model_name: str) -> bool:
    section = normalize_scalar(row.get("section")).lower()
    lower = content.lower()
    equations = normalize_list(row.get("equations"))
    if model_name == "unknown" or is_noisy_model_name(model_name):
        return False
    if model_name != "unknown" and re.search(r"\b(model|loss|situation)\b", model_name, flags=re.IGNORECASE):
        return True
    if re.search(r"\b(path loss|transmission loss|loss model|propagation loss)\b", section):
        return True
    if equations and re.search(r"\b(model|loss)\b", lower):
        return True
    if re.search(r"\b(input|output|relevant)\s+parameters?\b", lower) and re.search(r"\b(model|loss)\b", lower):
        return True
    return False


def is_known_value(value: Any) -> bool:
    normalized = normalize_scalar(value)
    return bool(normalized) and normalized.lower() != "unknown"


def choose_richer_scalar(current: Any, candidate: Any) -> str:
    current_text = normalize_scalar(current)
    candidate_text = normalize_scalar(candidate)
    if not is_known_value(current_text):
        return candidate_text or current_text
    if is_known_value(candidate_text) and len(candidate_text.split()) > len(current_text.split()):
        return candidate_text
    return current_text


def merge_parameter_dict(current: dict[str, str], candidate: dict[str, str]) -> dict[str, str]:
    merged = dict(current)
    for name, meaning in candidate.items():
        if name not in merged or not is_known_value(merged[name]):
            merged[name] = meaning
    return merged


def extract_where_parameter_dict(content: str) -> dict[str, str]:
    if not re.search(r"\bwhere\b", content, flags=re.IGNORECASE):
        return {}
    names = re.findall(r"(?:^|\s|[;,\-–])([A-Za-zα-ωΑ-Ωλ][A-Za-z0-9_α-ωΑ-Ωλ]{0,12})\s*:", content)
    params: dict[str, str] = {}
    for raw_name in names:
        name = normalize_parameter_name(raw_name)
        lowered = name.lower()
        if not name or lowered in PARAMETER_NAME_STOPWORDS or lowered in {"fig", "table"}:
            continue
        if len(name) > 12:
            continue
        if len(name) > 4 and name.islower():
            continue
        if name:
            params[name] = "unknown"
    return params


def infer_output_parameters(content: str, model_name: str, formula_latex: str) -> dict[str, str]:
    name_lower = model_name.lower()
    content_lower = content.lower()
    outputs: dict[str, str] = {}

    if "path morphology" in name_lower:
        return {"MIMO channel model parameters": "unknown"}
    if "cross-correlation" in name_lower:
        return {"cross-correlation coefficient": "unknown"}
    if "angular spread" in name_lower:
        outputs["angular spread of departure"] = "unknown"
        outputs["angular spread of arrival"] = "unknown"
        return outputs

    if re.search(
        r"\bbasic transmission loss\b[^.]{0,140}\b(?:is given|given by|can be calculated|calculated by|defined as)\b",
        content_lower,
    ):
        outputs["basic transmission loss"] = "unknown"
    if re.search(
        r"\b(?:r\.m\.s\.\s+)?delay spread\b[^.]{0,140}\b(?:is given|given by|modelled as|defined as)\b",
        content_lower,
    ):
        outputs["r.m.s. delay spread" if "r.m.s. delay spread" in content_lower else "delay spread"] = "unknown"
    return outputs


def merge_model_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for model in models:
        model_id = normalize_scalar(model.get("model_id"))
        if model_id not in merged:
            merged[model_id] = dict(model)
            continue
        target = merged[model_id]
        for field in (
            "frequency_range",
            "scenario",
            "los_or_nlos",
            "formula_latex",
            "assumptions",
            "limitations",
        ):
            target[field] = choose_richer_scalar(target.get(field), model.get(field))
        target["input_parameters"] = merge_parameter_dict(
            target.get("input_parameters") or {},
            model.get("input_parameters") or {},
        )
        target["output_parameters"] = merge_parameter_dict(
            target.get("output_parameters") or {},
            model.get("output_parameters") or {},
        )
        related_ids = target.setdefault("related_chunk_ids", [])
        for chunk_id in model.get("related_chunk_ids") or []:
            if chunk_id and chunk_id not in related_ids:
                related_ids.append(chunk_id)
    return sorted(merged.values(), key=lambda item: item["model_id"])


def build_propagation_model_index(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for row in rows:
        if not is_itu_standard_model(row):
            continue
        content = normalize_content_text(row.get("content_md"))
        equations = normalize_list(row.get("equations"))
        section = normalize_scalar(row.get("section"))
        chunk_id = normalize_scalar(row.get("chunk_id"))
        model_name = extract_model_name(section, content)
        if not is_model_candidate(row, content, model_name):
            continue
        input_parameters = extract_parameter_dict(content, "input")
        input_parameters = merge_parameter_dict(input_parameters, extract_where_parameter_dict(content))
        formula_latex = extract_model_formula(equations, content)
        output_parameters = extract_parameter_dict(content, "output")
        output_parameters = merge_parameter_dict(output_parameters, infer_output_parameters(content, model_name, formula_latex))
        model = {
            "model_id": build_model_id(row, model_name),
            "source_id": "itu_r_p1411_13",
            "model_name": model_name,
            "frequency_range": extract_frequency_range(content),
            "scenario": extract_scenario(content, section),
            "los_or_nlos": extract_los_or_nlos(content),
            "input_parameters": input_parameters,
            "output_parameters": output_parameters,
            "formula_latex": formula_latex,
            "assumptions": extract_assumptions(content),
            "limitations": extract_limitations(content),
            "related_chunk_ids": [chunk_id] if chunk_id else [],
        }
        models.append(model)
    return merge_model_rows(models)


def build_report(
    input_rows: list[dict[str, Any]],
    formula_rows: list[dict[str, Any]],
    term_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Structured Extraction Report",
        "",
        "Phase status: completed",
        "",
        "## Summary",
        f"- input_chunks: {len(input_rows)}",
        f"- formula_index_rows: {len(formula_rows)}",
        f"- term_index_rows: {len(term_rows)}",
        f"- propagation_model_index_rows: {len(model_rows)}",
        "",
        "## Notes",
        "- Extraction is rule-based only; no LLM calls were used.",
        "- Formula rows are limited to textbook/course chunks with explicit equations.",
        "- Propagation model rows are limited to ITU-R P.1411-13 chunks marked as standard_model.",
        "- Unknown or empty fields indicate no reliable in-chunk evidence was found.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    build_root = project_root / "kb_corpus_build"
    input_path = build_root / "corpus" / "chunks.canonical.jsonl"
    formula_path = build_root / "corpus" / "formula_index.jsonl"
    term_path = build_root / "corpus" / "term_index.jsonl"
    model_path = build_root / "corpus" / "propagation_model_index.jsonl"
    report_path = build_root / "reports" / "structured_extraction_report.md"

    input_rows = load_jsonl(input_path)
    formula_rows = build_formula_index(input_rows)
    term_rows = build_term_index(input_rows)
    model_rows = build_propagation_model_index(input_rows)

    write_jsonl_checked(formula_path, formula_rows)
    write_jsonl_checked(term_path, term_rows)
    write_jsonl_checked(model_path, model_rows)
    write_text_checked(report_path, build_report(input_rows, formula_rows, term_rows, model_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
