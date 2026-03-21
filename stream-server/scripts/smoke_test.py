#!/usr/bin/env python3
"""Send one tiny JPEG frame over WebSocket; prints server JSON response."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets

# Repo root (stream-server/) so shared_test_jpeg imports when run as scripts/smoke_test.py
_SS = Path(__file__).resolve().parent.parent
if str(_SS) not in sys.path:
    sys.path.insert(0, str(_SS))
from shared_test_jpeg import MINIMAL_JPEG_B64


async def run(url: str) -> None:
    payload = json.dumps(
        {
            "frame": MINIMAL_JPEG_B64,
            "camera": "front",
            "timestamp": 0,
        }
    )
    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(payload)
        raw = await asyncio.wait_for(ws.recv(), timeout=60)
    data = json.loads(raw)
    print(json.dumps(data, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--url",
        default="ws://127.0.0.1:8765",
        help="WebSocket URL: ws://... for local server, wss://... for Modal (default: ws://127.0.0.1:8765)",
    )
    args = p.parse_args()
    try:
        asyncio.run(run(args.url))
    except ConnectionRefusedError:
        print("Connection refused — is the server running? (python server.py)", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
