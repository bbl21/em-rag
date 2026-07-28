"""Stable product errors shared by API and application layers."""

from __future__ import annotations


class ProductError(RuntimeError):
    """An expected failure with a stable public error code."""

    def __init__(self, code: str, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def invalid_request(message: str) -> ProductError:
    return ProductError("invalid_request", message, status_code=422)


def unsupported_language(message: str = "Only English queries are supported in this release.") -> ProductError:
    return ProductError("unsupported_language", message, status_code=422)


def artifact_not_ready(message: str) -> ProductError:
    return ProductError("artifact_not_ready", message, status_code=503)


def retrieval_failed(message: str) -> ProductError:
    return ProductError("retrieval_failed", message, status_code=500)


def provider_not_configured() -> ProductError:
    return ProductError(
        "provider_not_configured",
        "No OpenAI-compatible answer provider is configured; use /v1/retrieve for evidence-only results.",
        status_code=503,
    )


def provider_timeout(message: str = "The answer provider timed out.") -> ProductError:
    return ProductError("provider_timeout", message, status_code=504)


def answer_validation_failed(message: str) -> ProductError:
    return ProductError("answer_validation_failed", message, status_code=502)
