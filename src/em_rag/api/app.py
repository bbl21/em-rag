"""FastAPI application factory for the local EM RAG product."""

import uuid
from pathlib import Path
from typing import Any

from em_rag.adapters.artifacts import ArtifactStore
from em_rag.adapters.legacy_retrieval import LegacyRetrievalAdapter
from em_rag.adapters.openai_compatible import OpenAICompatibleProvider
from em_rag.application.answer import AnswerService
from em_rag.application.retrieval import RetrievalService
from em_rag.config import Settings
from em_rag.domain.errors import ProductError
from em_rag.domain.models import RetrievalRequest


def create_app(
    settings: Settings | None = None,
    *,
    retrieval_service: RetrievalService | None = None,
    answer_service: AnswerService | None = None,
):
    try:
        from fastapi import FastAPI, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import FileResponse, JSONResponse
        from pydantic import BaseModel, Field
    except ImportError as error:
        raise RuntimeError("FastAPI runtime dependencies are not installed; install em-rag[runtime].") from error

    active_settings = settings or Settings.from_env()
    artifacts = ArtifactStore(active_settings.project_root)
    retrieval = retrieval_service or RetrievalService(
        artifacts,
        LegacyRetrievalAdapter(active_settings.project_root),
        max_top_k=active_settings.max_top_k,
    )
    if answer_service is None:
        provider = None
        if active_settings.provider_configured:
            provider = OpenAICompatibleProvider(
                base_url=active_settings.provider_base_url,
                api_key=active_settings.provider_api_key,
                model=active_settings.provider_model,
                timeout_seconds=active_settings.provider_timeout_seconds,
            )
        answers = AnswerService(retrieval, provider)
    else:
        answers = answer_service

    class RetrieveBody(BaseModel):
        query: str = Field(min_length=1, max_length=4096)
        top_k: int = Field(default=active_settings.default_top_k, ge=1, le=active_settings.max_top_k)
        retrieval_mode: str = Field(default=active_settings.retrieval_mode)

    app = FastAPI(
        title="EM RAG",
        version="0.2.0",
        description="Evidence-grounded local retrieval for electromagnetics and antennas.",
    )

    @app.on_event("shutdown")
    def close_retrieval_runtime() -> None:
        close = getattr(retrieval.adapter, "close", None)
        if callable(close):
            close()

    def error_payload(code: str, message: str, request_id: str) -> dict[str, Any]:
        return {"error": {"code": code, "message": message, "request_id": request_id}}

    @app.middleware("http")
    async def request_identity(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.exception_handler(ProductError)
    async def product_error(request: Request, error: ProductError):
        return JSONResponse(
            status_code=error.status_code,
            content=error_payload(error.code, error.message, request.state.request_id),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=error_payload("invalid_request", "Request validation failed.", request.state.request_id),
        )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception):
        # Keep implementation details and local paths out of the public contract.
        return JSONResponse(
            status_code=500,
            content=error_payload(
                "internal_error",
                "The service encountered an unexpected error.",
                request.state.request_id,
            ),
        )

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    def ready():
        status = retrieval.readiness()
        return JSONResponse(status_code=200 if status["ready"] else 503, content=status)

    @app.get("/v1/system")
    def system() -> dict[str, Any]:
        snapshot = artifacts.snapshot()
        return {
            "service": "em-rag",
            "version": "0.2.0",
            "artifact": snapshot.to_dict(),
            "default_retrieval_mode": active_settings.retrieval_mode,
            "vector_required": True,
            "embedding_backend": active_settings.embedding_backend,
            "answer_provider_configured": active_settings.provider_configured,
            "release_policy": "publish_with_known_quality_risk",
        }

    @app.post("/v1/retrieve")
    def retrieve(body: RetrieveBody, request: Request):
        result = retrieval.retrieve(
            RetrievalRequest(
                query=body.query,
                top_k=body.top_k,
                retrieval_mode=body.retrieval_mode,
            ),
            request_id=request.state.request_id,
        )
        return result.to_dict()

    @app.post("/v1/answer")
    def answer(body: RetrieveBody, request: Request):
        result = answers.answer(
            RetrievalRequest(
                query=body.query,
                top_k=body.top_k,
                retrieval_mode=body.retrieval_mode,
            ),
            request_id=request.state.request_id,
        )
        return result.to_dict()

    web_root = Path(__file__).resolve().parents[1] / "web"

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web_root / "index.html")

    return app
