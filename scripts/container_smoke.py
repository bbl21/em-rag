#!/usr/bin/env python3
"""Smoke-test the product HTTP contract of a freshly started container."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8000"


def request_json(path: str) -> tuple[int, dict]:
    try:
        response = urllib.request.urlopen(f"{BASE_URL}{path}", timeout=3)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.loads(response.read().decode("utf-8"))


def post_json(path: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main() -> int:
    require_ready = os.environ.get("EM_RAG_CONTAINER_REQUIRE_READY", "").strip().lower() in {
        "1",
        "yes",
        "true",
        "on",
    }
    deadline = time.monotonic() + 180
    while True:
        try:
            live_status, live = request_json("/health/live")
            if live_status == 200 and live.get("status") == "live":
                break
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("container did not become live within 180 seconds")
        time.sleep(1)

    system_status, system = request_json("/v1/system")
    ready_status, ready = request_json("/health/ready")
    if system_status != 200 or system.get("vector_required") is not True:
        raise RuntimeError("system endpoint does not advertise mandatory vector retrieval")
    if require_ready:
        if ready_status != 200 or ready.get("ready") is not True:
            raise RuntimeError("container must report ready for smoke checks with synthetic demo")
        if ready.get("vector_runtime_status") != "ok":
            raise RuntimeError("container readiness check must report vector runtime ok")
        if ready.get("bm25_runtime_status") != "ok":
            raise RuntimeError("container readiness check must report BM25 runtime ok")
        if ready.get("structured_runtime_status") not in {"ok", "unavailable"}:
            raise RuntimeError("container readiness check returned an invalid structured runtime state")
        if ready.get("vector_runtime_error"):
            raise RuntimeError("container readiness payload should not expose vector runtime error details")
        retrieve_status, retrieval = post_json(
            "/v1/retrieve",
            {"query": "What does S11 represent?", "top_k": 3, "retrieval_mode": "hybrid"},
        )
        evidence = retrieval.get("evidence", [])
        if retrieve_status != 200 or retrieval.get("degraded") is not False:
            raise RuntimeError("container synthetic hybrid retrieval must succeed without degradation")
        if "demo_s11" not in [row.get("chunk_id") for row in evidence]:
            raise RuntimeError("container synthetic hybrid retrieval did not return demo_s11")
        print("PASS: container is live and vector-ready with synthetic demo")
    else:
        # Legacy smoke check: public source image deliberately has no private
        # indexes and should report readiness closed.
        if ready_status != 503 or ready.get("ready") is not False:
            raise RuntimeError("container without a mounted artifact must report not ready")
        print("PASS: container is live and fails readiness closed without private artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
