import os
import tempfile
import textwrap
import unittest
from pathlib import Path

from em_rag.adapters.vector_client import PersistentVectorClient


class PersistentVectorClientTests(unittest.TestCase):
    def _worker(self, root: Path, *, hang: bool = False) -> Path:
        path = root / "worker.py"
        behavior = "continue" if hang else "pass"
        path.write_text(
            textwrap.dedent(
                f"""
                import json, os, sys
                print(json.dumps({{"status": "ready"}}), flush=True)
                for line in sys.stdin:
                    request = json.loads(line)
                    if {hang!r}:
                        {behavior}
                    print(json.dumps({{
                        "request_id": request["request_id"],
                        "status": "ok",
                        "scores": {{"chunk": float(os.getpid())}},
                    }}), flush=True)
                """
            ),
            encoding="utf-8",
        )
        return path

    def test_reuses_one_worker_process_for_multiple_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with PersistentVectorClient(
                worker_path=self._worker(root),
                build_root=root,
                environment=os.environ.copy(),
                timeout_seconds=3,
            ) as client:
                first, first_status, _ = client.query("first")
                first_pid = client.process_id
                second, second_status, _ = client.query("second")

                self.assertEqual(first_status, "ok")
                self.assertEqual(second_status, "ok")
                self.assertEqual(client.process_id, first_pid)
                self.assertEqual(first["chunk"], second["chunk"])

    def test_timeout_stops_worker_so_next_query_can_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = PersistentVectorClient(
                worker_path=self._worker(root, hang=True),
                build_root=root,
                environment=os.environ.copy(),
                timeout_seconds=1,
            )
            _scores, status, _error = client.query("hang")

            self.assertEqual(status, "timeout")
            self.assertIsNone(client.process_id)


if __name__ == "__main__":
    unittest.main()
