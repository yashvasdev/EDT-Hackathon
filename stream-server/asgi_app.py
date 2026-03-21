"""
Starlette ASGI app for GuardCam WebSocket (used by Modal and local ASGI tests).

For local dev, prefer `python server.py`. For Modal, deploy `modal_app.py`.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure stream-server dir is importable when loaded from Modal image (/guardcam)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from server import GUARDCAM_E2E, load_model, process_session


@asynccontextmanager
async def _lifespan(_app):
    if GUARDCAM_E2E:
        print("🧪 GUARDCAM_E2E=1 — skipping YOLO load (tests only)")
    else:
        load_model()
    yield


async def _websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client = ws.scope.get("client")
    if client:
        client_addr = f"{client[0]}:{client[1]}"
    else:
        client_addr = "unknown"

    async def messages():
        while True:
            try:
                yield await ws.receive_text()
            except WebSocketDisconnect:
                break

    await process_session(client_addr, messages(), ws.send_text)


def build_app() -> Starlette:
    return Starlette(
        lifespan=_lifespan,
        routes=[
            WebSocketRoute("/", _websocket_endpoint),
        ],
    )
