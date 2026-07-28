"""Minimal OpenAI-compatible chat-completions adapter."""

from __future__ import annotations

from typing import Any

from em_rag.domain.errors import ProductError, provider_timeout, retrieval_failed


class OpenAICompatibleProvider:
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout_seconds: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            import httpx
        except ImportError as error:
            raise retrieval_failed("The runtime HTTP client is not installed.") from error
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model": self.model, "messages": messages, "temperature": 0},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise provider_timeout() from error
        except httpx.HTTPError as error:
            raise ProductError("retrieval_degraded", f"Answer provider request failed: {type(error).__name__}", status_code=502) from error
        try:
            payload: Any = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ProductError("retrieval_degraded", "Answer provider returned an invalid response.", status_code=502) from error
        if not isinstance(content, str) or not content.strip():
            raise ProductError("retrieval_degraded", "Answer provider returned an empty answer.", status_code=502)
        return content.strip()
