"""Persistent timeout-isolated client for the local vector worker."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any


class PersistentVectorClient:
    """Keeps one isolated model process resident and restarts it on failure."""

    def __init__(
        self,
        *,
        worker_path: Path,
        build_root: Path,
        environment: dict[str, str],
        timeout_seconds: int,
    ) -> None:
        self.worker_path = worker_path.resolve()
        self.build_root = build_root.resolve()
        self.environment = dict(environment)
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()

    @property
    def process_id(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @staticmethod
    def _read_responses(
        process: subprocess.Popen[str], responses: queue.Queue[dict[str, Any] | None]
    ) -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = {"status": "invalid_output", "error": "worker returned invalid JSON"}
                responses.put(payload if isinstance(payload, dict) else {"status": "invalid_output"})
        finally:
            responses.put(None)

    def _next_response(self) -> dict[str, Any]:
        try:
            response = self._responses.get(timeout=self.timeout_seconds)
        except queue.Empty as error:
            raise TimeoutError(f"vector worker timed out after {self.timeout_seconds} seconds") from error
        if response is None:
            raise RuntimeError("vector worker exited")
        return response

    def _start(self) -> tuple[str, str]:
        if self._process is not None and self._process.poll() is None:
            return "ok", ""
        self._stop()
        self._responses = queue.Queue()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(self.worker_path),
                    "--build-root",
                    str(self.build_root),
                    "--serve",
                ],
                cwd=self.build_root.parent,
                env=self.environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as error:
            return "startup_error", f"{type(error).__name__}: {error}"
        self._process = process
        threading.Thread(
            target=self._read_responses,
            args=(process, self._responses),
            daemon=True,
        ).start()
        try:
            ready = self._next_response()
        except (TimeoutError, RuntimeError) as error:
            self._stop()
            return "timeout" if isinstance(error, TimeoutError) else "startup_error", str(error)
        if ready.get("status") != "ready":
            self._stop()
            return str(ready.get("status") or "startup_error"), str(ready.get("error") or "")
        return "ok", ""

    def _stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def close(self) -> None:
        with self._lock:
            self._stop()

    def query(self, text: str) -> tuple[dict[str, float], str, str]:
        with self._lock:
            status, error = self._start()
            if status != "ok":
                return {}, status, error
            request_id = str(uuid.uuid4())
            assert self._process is not None and self._process.stdin is not None
            try:
                self._process.stdin.write(
                    json.dumps({"request_id": request_id, "query": text}, ensure_ascii=False) + "\n"
                )
                self._process.stdin.flush()
                response = self._next_response()
            except TimeoutError as exc:
                self._stop()
                return {}, "timeout", str(exc)
            except (BrokenPipeError, OSError, RuntimeError) as exc:
                self._stop()
                return {}, "error", f"{type(exc).__name__}: {exc}"
            if response.get("request_id") != request_id:
                self._stop()
                return {}, "invalid_output", "vector worker response identity mismatch"
            response_status = str(response.get("status") or "error")
            raw_scores = response.get("scores")
            if response_status != "ok" or not isinstance(raw_scores, dict):
                return {}, response_status, str(response.get("error") or "")
            scores: dict[str, float] = {}
            for chunk_id, score in raw_scores.items():
                try:
                    scores[str(chunk_id)] = float(score)
                except (TypeError, ValueError):
                    continue
            return scores, "ok", ""

    def __enter__(self) -> "PersistentVectorClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
