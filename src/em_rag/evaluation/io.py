"""Strict UTF-8 JSON Lines input and output."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    if "run_id" in row and "query_id" in row:
        return ("run", row["run_id"], row["query_id"])
    if "run_id" in row:
        return ("run_id", row["run_id"])
    if "relevance" in row and "query_id" in row and "chunk_id" in row:
        return ("qrel", row["query_id"], row["chunk_id"])
    if "query" in row and "query_id" in row:
        return ("query_id", row["query_id"])
    if "id" in row:
        return ("id", row["id"])
    return None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read strict UTF-8 JSONL objects and reject duplicate primary identities."""
    file_path = Path(path)
    try:
        text = file_path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{file_path} is not valid UTF-8") from error

    rows: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {line_number}")
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid JSON at line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} must be an object")
        identity = _identity(row)
        if identity is not None:
            try:
                duplicate = identity in identities
            except TypeError as error:
                raise ValueError(f"row identity at line {line_number} must be scalar") from error
            if duplicate:
                raise ValueError(f"duplicate row identity at line {line_number}")
            identities.add(identity)
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write deterministic UTF-8 JSONL with LF terminators."""
    encoded_lines: list[bytes] = []
    identities: set[tuple[Any, ...]] = set()
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {line_number} must be an object")
        normalized = dict(row)
        identity = _identity(normalized)
        if identity is not None:
            try:
                duplicate = identity in identities
            except TypeError as error:
                raise ValueError(f"row identity at line {line_number} must be scalar") from error
            if duplicate:
                raise ValueError(f"duplicate row identity at line {line_number}")
            identities.add(identity)
        try:
            line = json.dumps(
                normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            encoded = line.encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError) as error:
            raise ValueError(f"row {line_number} is not UTF-8 JSON serializable") from error
        if encoded.decode("utf-8") != line:
            raise ValueError(f"UTF-8 roundtrip mismatch at row {line_number}")
        encoded_lines.append(encoded)

    payload = b"\n".join(encoded_lines)
    if encoded_lines:
        payload += b"\n"
    if b"\r\n" in payload:
        raise ValueError("CRLF output is not allowed")
    output_path = Path(path)
    output_path.write_bytes(payload)
    written_payload = output_path.read_bytes()
    try:
        written_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{output_path} was not written as valid UTF-8") from error
    if written_payload != payload:
        raise ValueError(f"disk byte mismatch after writing {output_path}")
