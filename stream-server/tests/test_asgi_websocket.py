"""
Regression tests for the Starlette ASGI WebSocket path (Modal uses this).

GUARDCAM_E2E must be set so YOLO is not loaded; set below before importing app code.

Run:
  cd stream-server && source venv/bin/activate && python -m unittest tests.test_asgi_websocket -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("GUARDCAM_E2E", "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from starlette.testclient import TestClient

from asgi_app import build_app
from shared_test_jpeg import MINIMAL_JPEG_B64


def _frame_msg(camera: str = "front") -> str:
    return json.dumps(
        {"frame": MINIMAL_JPEG_B64, "camera": camera, "timestamp": 0}
    )


class TestAsgiWebSocket(unittest.TestCase):
    def test_json_contract_single_frame(self) -> None:
        with TestClient(build_app()) as client:
            with client.websocket_connect("/") as ws:
                ws.send_text(_frame_msg("front"))
                raw = ws.receive_text()
                data = json.loads(raw)
                for key in ("drowsy", "alert", "confidence", "class", "fps"):
                    self.assertIn(key, data)

    def test_third_frame_alert(self) -> None:
        with TestClient(build_app()) as client:
            with client.websocket_connect("/") as ws:
                for _ in range(3):
                    ws.send_text(_frame_msg("front"))
                    data = json.loads(ws.receive_text())
                self.assertTrue(data["drowsy"])
                self.assertTrue(data["alert"])


if __name__ == "__main__":
    unittest.main()
