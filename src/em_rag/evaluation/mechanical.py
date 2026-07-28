"""Deterministic integrity checks for retrieved evidence."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .models import EvalCase, RetrievalRun


BLOCKING_CODES = {
    "missing_chunk",
    "citation_mismatch",
    "hard_negative_false_positive",
    "missing_required_facet",
}
WARNING_CODES = {
    "duplicate_evidence",
    "formula_context_gap",
    "numeric_support_gap",
}

_FORMULA_PATTERN = re.compile(r"(?:[A-Za-z]\s*=|=|\b(?:log|ln|sqrt)\s*\()")
_CONTEXT_PATTERN = re.compile(
    r"\b(?:where|when|condition|valid|assum|denot|represent|defined|variable|subject\s+to)\w*\b",
    re.IGNORECASE,
)
FORMULA_CONTEXT_WINDOW_CHARS = 96
_UNIT_SYMBOL_TOKENS = frozenset({"%", "\u2030", "\u00b0"})
_MECHANICAL_TOKEN_PATTERN = re.compile(
    r"(?<![\w.])[+-]?(?:\d+\.\d+|\d+)(?!\w)|[^\W\d_]+|[%\u2030\u00b0]",
    re.UNICODE,
)
_NUMBER_TOKEN_PATTERN = re.compile(r"[+-]?(?:\d+\.\d+|\d+)")



@dataclass(frozen=True)
class MechanicalFinding:
    code: str
    severity: Literal["blocking", "warning"]
    query_id: str
    chunk_id: str | None
    detail: str


@dataclass(frozen=True)
class _MechanicalToken:
    value: str
    kind: Literal["number", "alpha", "symbol"]
    start: int
    end: int


@dataclass(frozen=True)
class _TokenizedText:
    source: str
    tokens: tuple[_MechanicalToken, ...]


def _tokenize(value: str) -> _TokenizedText:
    source = unicodedata.normalize("NFKC", value).casefold()
    tokens: list[_MechanicalToken] = []
    for match in _MECHANICAL_TOKEN_PATTERN.finditer(source):
        token_value = match.group(0)
        if _NUMBER_TOKEN_PATTERN.fullmatch(token_value):
            kind: Literal["number", "alpha", "symbol"] = "number"
        elif token_value in _UNIT_SYMBOL_TOKENS:
            kind = "symbol"
        else:
            kind = "alpha"
        tokens.append(
            _MechanicalToken(
                value=token_value,
                kind=kind,
                start=match.start(),
                end=match.end(),
            )
        )
    return _TokenizedText(source=source, tokens=tuple(tokens))


def _normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.value for token in _tokenize(value).tokens)


def _contains_token_sequence(
    haystack: tuple[str, ...], needle: tuple[str, ...]
) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def _tokens_are_adjacent(
    tokenized: _TokenizedText,
    left: _MechanicalToken,
    right: _MechanicalToken,
) -> bool:
    separator = tokenized.source[left.end : right.start]
    return not separator or separator.isspace()


def _query_numeric_phrases(value: str) -> tuple[tuple[str, ...], ...]:
    tokenized = _tokenize(value)
    phrases: list[tuple[str, ...]] = []
    for index, token in enumerate(tokenized.tokens):
        if token.kind != "number":
            continue

        phrase = [token.value]
        if index + 1 < len(tokenized.tokens):
            next_token = tokenized.tokens[index + 1]
            if (
                next_token.kind in {"alpha", "symbol"}
                and _tokens_are_adjacent(tokenized, token, next_token)
            ):
                phrase.append(next_token.value)
                if (
                    next_token.kind == "symbol"
                    and index + 2 < len(tokenized.tokens)
                ):
                    suffix = tokenized.tokens[index + 2]
                    if (
                        suffix.kind == "alpha"
                        and _tokens_are_adjacent(tokenized, next_token, suffix)
                    ):
                        phrase.append(suffix.value)
        phrases.append(tuple(phrase))
    return tuple(phrases)


def _numeric_phrase_supported(
    phrase: tuple[str, ...],
    evidence: tuple[_TokenizedText, ...],
) -> bool:
    for tokenized in evidence:
        tokens = tokenized.tokens
        for start in range(len(tokens) - len(phrase) + 1):
            candidate = tokens[start : start + len(phrase)]
            if tuple(token.value for token in candidate) != phrase:
                continue
            if all(
                _tokens_are_adjacent(tokenized, candidate[index], candidate[index + 1])
                for index in range(len(candidate) - 1)
            ):
                return True
    return False


def _formula_context_windows(text: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            max(0, formula_match.start() - FORMULA_CONTEXT_WINDOW_CHARS),
            min(len(text), formula_match.end() + FORMULA_CONTEXT_WINDOW_CHARS),
        )
        for formula_match in _FORMULA_PATTERN.finditer(text)
    )


def _stable_unique_findings(
    findings: list[MechanicalFinding],
) -> tuple[MechanicalFinding, ...]:
    unique: list[MechanicalFinding] = []
    seen: set[MechanicalFinding] = set()
    for finding in findings:
        if finding not in seen:
            seen.add(finding)
            unique.append(finding)
    return tuple(unique)


def _finding(
    code: str, query_id: str, chunk_id: str | None, detail: str
) -> MechanicalFinding:
    if code in BLOCKING_CODES:
        severity: Literal["blocking", "warning"] = "blocking"
    elif code in WARNING_CODES:
        severity = "warning"
    else:
        raise ValueError(f"unsupported mechanical finding code: {code}")
    return MechanicalFinding(code, severity, query_id, chunk_id, detail)


def _document_text(document: Mapping[str, Any]) -> str:
    text = document.get("text", "")
    return text if isinstance(text, str) else ""


def inspect_run(
    case: EvalCase,
    run: RetrievalRun,
    docstore: Mapping[str, Mapping[str, Any]],
) -> tuple[MechanicalFinding, ...]:
    """Inspect one run using only explicit case data and canonical documents."""
    if case.query_id != run.query_id:
        raise ValueError("case and run query IDs must match")

    findings: list[MechanicalFinding] = []
    seen_chunks: set[str] = set()
    seen_citations: set[str] = set()
    seen_groups: set[str] = set()
    cited_texts: list[str] = []
    cited_tokenized: list[_TokenizedText] = []

    for evidence in sorted(run.results, key=lambda item: item.rank):
        if evidence.chunk_id in seen_chunks:
            findings.append(
                _finding(
                    "duplicate_evidence",
                    case.query_id,
                    evidence.chunk_id,
                    f"chunk ID repeated: {evidence.chunk_id}",
                )
            )
        seen_chunks.add(evidence.chunk_id)

        if evidence.citation in seen_citations:
            findings.append(
                _finding(
                    "duplicate_evidence",
                    case.query_id,
                    evidence.chunk_id,
                    f"citation repeated: {evidence.citation}",
                )
            )
        seen_citations.add(evidence.citation)

        document = docstore.get(evidence.chunk_id)
        if document is None:
            findings.append(
                _finding(
                    "missing_chunk",
                    case.query_id,
                    evidence.chunk_id,
                    f"chunk is absent from canonical docstore: {evidence.chunk_id}",
                )
            )
        else:
            canonical_citation = document.get("citation")
            if canonical_citation != evidence.citation:
                findings.append(
                    _finding(
                        "citation_mismatch",
                        case.query_id,
                        evidence.chunk_id,
                        f"retrieved citation {evidence.citation!r} does not match canonical citation {canonical_citation!r}",
                    )
                )

            group_id = document.get("group_id")
            if isinstance(group_id, str) and group_id:
                if group_id in seen_groups:
                    findings.append(
                        _finding(
                            "duplicate_evidence",
                            case.query_id,
                            evidence.chunk_id,
                            f"evidence group repeated: {group_id}",
                        )
                    )
                seen_groups.add(group_id)

            text = _document_text(document)
            cited_texts.append(text)
            cited_tokenized.append(_tokenize(text))
            for window_start, window_end in _formula_context_windows(text):
                formula_context = text[window_start:window_end]
                if not _CONTEXT_PATTERN.search(formula_context):
                    findings.append(
                        _finding(
                            "formula_context_gap",
                            case.query_id,
                            evidence.chunk_id,
                            "formula lacks variable or condition context within "
                            f"{FORMULA_CONTEXT_WINDOW_CHARS} characters",
                        )
                    )

        if case.is_hard_negative and evidence.score >= run.confidence_threshold:
            findings.append(
                _finding(
                    "hard_negative_false_positive",
                    case.query_id,
                    evidence.chunk_id,
                    "hard-negative evidence meets or exceeds the confidence threshold",
                )
            )

    evidence_tokens = _normalized_tokens(" ".join(cited_texts))
    for facet in case.expected_facets:
        if not _contains_token_sequence(evidence_tokens, _normalized_tokens(facet)):
            findings.append(
                _finding(
                    "missing_required_facet",
                    case.query_id,
                    None,
                    f"expected facet is absent from cited text: {facet}",
                )
            )

    numeric_evidence = tuple(cited_tokenized)
    missing_numeric_phrases = [
        phrase
        for phrase in _query_numeric_phrases(case.query)
        if not _numeric_phrase_supported(phrase, numeric_evidence)
    ]
    if missing_numeric_phrases:
        findings.append(
            _finding(
                "numeric_support_gap",
                case.query_id,
                None,
                "query number or unit is absent from cited text: "
                + ", ".join(" ".join(phrase) for phrase in missing_numeric_phrases),
            )
        )

    return _stable_unique_findings(findings)
