"""Product retrieval orchestration with explicit quality and degradation state."""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from em_rag.adapters.artifacts import ArtifactStore
from em_rag.domain.errors import artifact_not_ready, invalid_request, unsupported_language
from em_rag.domain.models import Evidence, RetrievalRequest, RetrievalResponse


class SearchAdapter(Protocol):
    def search(self, query: str, top_k: int, retrieval_mode: str) -> dict[str, Any]: ...


class RetrievalService:
    def __init__(self, artifacts: ArtifactStore, adapter: SearchAdapter, *, max_top_k: int = 50) -> None:
        self.artifacts = artifacts
        self.adapter = adapter
        self.max_top_k = max_top_k

    @staticmethod
    def _public_runtime_error(component: str, status: str) -> str:
        if status == "ok":
            return ""
        if component == "structured" and status == "unavailable":
            return "Structured indexes are not available for this artifact."
        if status == "partial":
            return f"One or more required {component} artifacts are missing."
        if status == "artifact_missing":
            return f"Required {component} artifact is missing."
        if status == "preflight_unavailable":
            return f"{component.title()} preflight check is not configured."
        if status.startswith("disabled_"):
            return f"{component.title()} retrieval is currently disabled."
        return f"{component.title()} runtime is unavailable; check service diagnostics."

    def readiness(self) -> dict[str, Any]:
        snapshot = self.artifacts.snapshot()
        statuses = {
            "bm25": "artifact_missing",
            "structured": "unavailable",
            "vector": "artifact_missing",
        }
        if snapshot.ready:
            for component in statuses:
                preflight = getattr(self.adapter, f"{component}_preflight", None)
                if preflight is None:
                    statuses[component] = "preflight_unavailable"
                else:
                    status, _ = preflight()
                    statuses[component] = str(status)
        value = snapshot.to_dict()
        for component, status in statuses.items():
            value[f"{component}_runtime_status"] = status
            value[f"{component}_runtime_error"] = self._public_runtime_error(component, status)
        structured_ready = statuses["structured"] == "ok" or (
            snapshot.quality_status == "demo_only" and statuses["structured"] == "unavailable"
        )
        value["ready"] = (
            snapshot.ready
            and statuses["bm25"] == "ok"
            and statuses["vector"] == "ok"
            and structured_ready
        )
        return value

    def retrieve(self, request: RetrievalRequest, *, request_id: str | None = None) -> RetrievalResponse:
        query = request.query.strip()
        if not query:
            raise invalid_request("query must not be blank")
        if len(query) > 4096:
            raise invalid_request("query must not exceed 4096 characters")
        if request.top_k < 1 or request.top_k > self.max_top_k:
            raise invalid_request(f"top_k must be between 1 and {self.max_top_k}")
        if request.retrieval_mode not in {"bm25_structured", "hybrid"}:
            raise invalid_request("retrieval_mode must be bm25_structured or hybrid")
        snapshot = self.artifacts.snapshot()
        if not snapshot.ready:
            raise artifact_not_ready(
                "Required artifact components are missing: " + ", ".join(snapshot.missing)
            )
        runtime = self.readiness()
        if not runtime["ready"]:
            raise artifact_not_ready("The retrieval runtime did not pass artifact preflight checks.")
        output = self.adapter.search(query, request.top_k, request.retrieval_mode)
        if output.get("error") == "unsupported_language":
            raise unsupported_language()
        evidence = tuple(
            Evidence(
                evidence_id=f"E{rank}",
                chunk_id=str(row.get("chunk_id") or ""),
                source_id=str(row.get("source_id") or ""),
                source_title=str(row.get("source_title") or ""),
                chapter=str(row.get("chapter") or ""),
                section=str(row.get("section") or ""),
                page_start=row.get("page_start") if isinstance(row.get("page_start"), int) else None,
                page_end=row.get("page_end") if isinstance(row.get("page_end"), int) else None,
                content=str(row.get("content") or row.get("content_preview") or ""),
                citation=str(row.get("citation") or ""),
                rank=rank,
                score=float(row.get("final_score") or 0.0),
                channel_scores={
                    str(name): float(value)
                    for name, value in (row.get("scores") or {}).items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                },
            )
            for rank, row in enumerate(output.get("results") or [], start=1)
        )
        vector_status = str(output.get("vector_status") or "disabled")
        degraded_components: list[str] = []
        warnings: list[str] = []
        if request.retrieval_mode == "hybrid" and vector_status != "ok":
            degraded_components.append("vector")
            warnings.append(f"vector_{vector_status}")
        if output.get("out_of_scope"):
            warnings.append("query_out_of_scope")
        if snapshot.quality_status not in {"pass", "release_eligible"}:
            warnings.append(f"release_quality_{snapshot.quality_status}")
        return RetrievalResponse(
            request_id=request_id or str(uuid.uuid4()),
            artifact_id=snapshot.artifact_id,
            query=query,
            evidence=evidence,
            degraded=bool(degraded_components),
            degraded_components=tuple(degraded_components),
            warnings=tuple(dict.fromkeys(warnings)),
        )
