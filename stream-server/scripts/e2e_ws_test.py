#!/usr/bin/env python3
"""
End-to-end WebSocket tests against stream-server (no phone required).

Starts server.py subprocesses with GUARDCAM_E2E=1 so JPEG decode + alert logic run
without loading YOLO. Do not use GUARDCAM_E2E in production.

Usage (from repo root or stream-server):
  cd stream-server && source venv/bin/activate && python scripts/e2e_ws_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import websockets

STREAM_SERVER = Path(__file__).resolve().parent.parent
if str(STREAM_SERVER) not in sys.path:
    sys.path.insert(0, str(STREAM_SERVER))
from shared_test_jpeg import MINIMAL_JPEG_B64


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _required_keys(d: dict) -> None:
    for k in ("drowsy", "alert", "confidence", "class", "fps"):
        assert k in d, f"missing key {k}: {d}"


def _start_server(port: int, extra_env: dict[str, str]) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["GUARDCAM_E2E"] = "1"
    env["PORT"] = str(port)
    env.update(extra_env)
    return subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=str(STREAM_SERVER),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _send_frames(url: str, n: int, camera: str) -> list[dict]:
    out: list[dict] = []
    payload_base = {
        "frame": MINIMAL_JPEG_B64,
        "camera": camera,
        "timestamp": 0,
    }
    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
        for i in range(n):
            payload_base["timestamp"] = i
            await ws.send(json.dumps(payload_base))
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            out.append(json.loads(raw))
    return out


def _wait_tcp(port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"server did not listen on 127.0.0.1:{port}")


def main() -> int:
    if not STREAM_SERVER.joinpath("server.py").is_file():
        print("Run from stream-server tree (scripts/e2e_ws_test.py)", file=sys.stderr)
        return 1

    # Test 1+2: consecutive drowsy -> alert on 3rd frame
    port1 = _pick_port()
    p1 = _start_server(port1, {})
    try:
        _wait_tcp(port1)
        url1 = f"ws://127.0.0.1:{port1}"

        async def run1():
            rows = await _send_frames(url1, 5, "front")
            assert len(rows) == 5
            for r in rows:
                _required_keys(r)
            assert rows[0]["drowsy"] is True
            assert rows[0]["alert"] is False
            assert rows[1]["alert"] is False
            assert rows[2]["alert"] is True, "expected alert on 3rd consecutive drowsy frame"
            assert rows[2]["drowsy"] is True
            # Cooldown: next immediate alerts may be false until 5s
            assert rows[3]["alert"] is False
            assert rows[4]["alert"] is False

        asyncio.run(run1())
        print("ok: front camera — 3rd frame alert + JSON contract")
    finally:
        p1.terminate()
        try:
            p1.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p1.kill()

    # Test 3: back camera + DROWSINESS_ONLY_FRONT — never drowsy, never alert
    port2 = _pick_port()
    p2 = _start_server(port2, {"DROWSINESS_ONLY_FRONT": "1"})
    try:
        _wait_tcp(port2)
        url2 = f"ws://127.0.0.1:{port2}"

        async def run2():
            rows = await _send_frames(url2, 5, "back")
            for r in rows:
                _required_keys(r)
                assert r["drowsy"] is False
                assert r["alert"] is False
                assert r["class"] == "Non Drowsy"

        asyncio.run(run2())
        print("ok: back camera + DROWSINESS_ONLY_FRONT — awake, no alerts")
    finally:
        p2.terminate()
        try:
            p2.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p2.kill()

    print("\nAll automated E2E checks passed.")
    print("Manual: Expo Go + real WebSocket URL + Start Monitoring — not run in CI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
