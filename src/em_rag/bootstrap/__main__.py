"""Verify or build the CPU-first local retrieval artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from em_rag.adapters.artifacts import ArtifactStore
from em_rag.adapters.legacy_retrieval import LegacyRetrievalAdapter
from em_rag.application.retrieval import RetrievalService
from em_rag.bootstrap.demo import build_demo, prepare_model
from em_rag.bootstrap.sources import SourceAcquisitionError, acquire_source, load_source_catalog


BUILD_STAGES = ("scan", "extract", "clean", "chunk", "contextualize", "dedup", "structured", "bm25", "vector")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "build", "demo", "prepare-model", "acquire-sources"])
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default=".em-rag-demo", help="Isolated demo artifact root.")
    parser.add_argument("--model-dir", default="", help="Optional local model destination.")
    parser.add_argument("--catalog", default="config/source-catalog.json", help="Public source catalogue JSON.")
    parser.add_argument("--destination", default="reference", help="Destination for acquired source files.")
    return parser.parse_args(argv)


def acquire_sources(root: Path, catalog_arg: str, destination_arg: str) -> int:
    """Acquire catalogue-approved downloads and print manual-source instructions."""
    catalog = Path(catalog_arg)
    if not catalog.is_absolute():
        catalog = root / catalog
    destination = Path(destination_arg)
    if not destination.is_absolute():
        destination = root / destination
    try:
        document = json.loads(catalog.read_text(encoding="utf-8"))
        allowed_hosts = document.get("allowed_hosts") if isinstance(document, dict) else None
        if not isinstance(allowed_hosts, list):
            raise SourceAcquisitionError("catalogue must declare an allowed_hosts list")
        sources = load_source_catalog(catalog, allowed_hosts)
    except (OSError, json.JSONDecodeError, SourceAcquisitionError) as error:
        print(f"source catalogue rejected: {error}", file=sys.stderr)
        return 2

    manual = []
    for source in sources:
        if source.acquisition_mode == "manual_download":
            manual.append(source)
            continue
        try:
            result = acquire_source(source, destination, allowed_hosts)
        except (OSError, SourceAcquisitionError) as error:
            print(f"{source.source_id}: acquisition failed: {error}", file=sys.stderr)
            return 2
        print(f"{result.source_id}: {result.status} ({result.path})")
    if manual:
        print("Manual source instructions:")
        for source in manual:
            print(f"- {source.source_id}: download from {source.url} after accepting its terms; place it at {destination / source.filename}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.project_root).resolve()
    if args.command == "acquire-sources":
        return acquire_sources(root, args.catalog, args.destination)
    if args.command == "prepare-model":
        model_dir = Path(args.model_dir).resolve() if args.model_dir else None
        print(prepare_model(root, model_dir))
        return 0
    if args.command == "demo":
        output = Path(args.output)
        if not output.is_absolute():
            output = root / output
        root = build_demo(root, output)
    if args.command == "build":
        if not (root / "reference").is_dir():
            print("reference/ is missing; see docs/data-sources.md", file=sys.stderr)
            return 2
        runner = root / "kb_corpus_build" / "scripts" / "run_pipeline.py"
        for stage in BUILD_STAGES:
            completed = subprocess.run(
                [sys.executable, str(runner), "--project-root", str(root), "--stage", stage],
                cwd=root,
            )
            if completed.returncode != 0:
                return completed.returncode
    artifacts = ArtifactStore(root)
    adapter = LegacyRetrievalAdapter(root)
    payload = RetrievalService(artifacts, adapter).readiness()
    adapter.close()
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
