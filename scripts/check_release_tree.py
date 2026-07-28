#!/usr/bin/env python3
"""Reject controlled data, secrets, absolute paths, and oversized tracked files."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


FORBIDDEN_PREFIXES = (
    "reference/",
    "kb_corpus_build/corpus/",
    "kb_corpus_build/indexes/",
    "kb_corpus_build/eval/retrieval_quality_v2/qrels/",
    "kb_corpus_build/eval/retrieval_quality_v2/runs/",
)
FORBIDDEN_NAMES = {".env", "retrieval_trace.jsonl"}
TEXT_SUFFIXES = {".cfg", ".css", ".env", ".html", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
SENSITIVE_PATTERNS = (
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
    re.compile(r"/(?:home|Users)/[^/\s]+"),
    re.compile(r"/mnt/[a-z]/(?:Users|home)/[^/\s]+", re.IGNORECASE),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


def tracked_files(root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]


def audit(root: Path, files: list[Path], *, max_bytes: int = 20 * 1024 * 1024) -> list[str]:
    findings: list[str] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if relative.startswith(FORBIDDEN_PREFIXES) or path.name in FORBIDDEN_NAMES:
            findings.append(f"forbidden tracked path: {relative}")
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > max_bytes:
            findings.append(f"oversized tracked file: {relative} ({path.stat().st_size} bytes)")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"Dockerfile", "LICENSE"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"non-UTF-8 release text: {relative}")
            continue
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(text):
                findings.append(f"sensitive text pattern in: {relative}")
                break
    for required in ("README.md", "LICENSE", "config/source-catalog.json", "docs/data-sources.md", "Dockerfile", "compose.yaml"):
        if not (root / required).is_file():
            findings.append(f"missing release file: {required}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--all-files", action="store_true", help="Audit every file below project-root instead of git tracked files.")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    files = [path for path in root.rglob("*") if path.is_file()] if args.all_files else tracked_files(root)
    findings = audit(root, files)
    if findings:
        for finding in findings:
            print(f"FAIL: {finding}")
        return 1
    print("PASS: tracked release tree contains no controlled paths, obvious secrets, absolute user paths, or oversized files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
