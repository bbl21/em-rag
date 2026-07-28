#!/usr/bin/env python3
"""Export a clean product source tree from an explicit allowlist."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


RELEASE_PATH_RENAMES = {"release/public.gitignore": ".gitignore"}


def tracked_paths(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "-z", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(item.decode("utf-8") for item in completed.stdout.split(b"\0") if item)


def committed_bytes(root: Path, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def read_allowlist(path: Path) -> tuple[set[str], tuple[str, ...]]:
    exact: set[str] = set()
    prefixes: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if value.startswith("/") or ".." in Path(value).parts:
            raise ValueError(f"unsafe allowlist entry at line {line_number}: {value}")
        if value.endswith("/"):
            prefixes.append(value)
        else:
            exact.add(value)
    return exact, tuple(prefixes)


def selected_paths(tracked: list[str], exact: set[str], prefixes: tuple[str, ...]) -> list[str]:
    return [path for path in tracked if path in exact or path.startswith(prefixes)]


def export(root: Path, output: Path, allowlist: Path) -> list[str]:
    root = root.resolve()
    output = output.resolve()
    if output == root or root in output.parents and output.name in {"src", "docs", "kb_corpus_build"}:
        raise ValueError("refusing unsafe release output path")
    if output.exists() and any(output.iterdir()):
        raise ValueError("release output directory must be empty or absent")
    exact, prefixes = read_allowlist(allowlist)
    selected = selected_paths(tracked_paths(root), exact, prefixes)
    missing_exact = exact - set(selected)
    if missing_exact:
        raise ValueError("allowlisted tracked files are missing: " + ", ".join(sorted(missing_exact)))
    for relative in selected:
        payload = committed_bytes(root, relative)
        targets = {relative, RELEASE_PATH_RENAMES.get(relative, relative)}
        for target_relative in targets:
            target = output / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allowlist", default="release/allowlist.txt")
    args = parser.parse_args()
    root = Path(args.project_root)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    allowlist = Path(args.allowlist)
    if not allowlist.is_absolute():
        allowlist = root / allowlist
    selected = export(root, output, allowlist)
    print(f"Exported {len(selected)} tracked files to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
