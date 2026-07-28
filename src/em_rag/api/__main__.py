"""Run the EM RAG API with Uvicorn."""

from __future__ import annotations

import os


def main() -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("Uvicorn is not installed; install em-rag[runtime].") from error
    uvicorn.run(
        "em_rag.api.app:create_app",
        factory=True,
        host=os.environ.get("EM_RAG_HOST", "0.0.0.0"),
        port=int(os.environ.get("EM_RAG_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
