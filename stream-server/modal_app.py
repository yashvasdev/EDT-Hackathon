"""
Deploy GuardCam WebSocket server on Modal (https://modal.com).

Prerequisites:
  pip install modal
  modal token new   # once, links your Modal account

Deploy (persistent URL):
  cd stream-server
  modal deploy modal_app.py

Modal prints a URL like https://YOUR_WORKSPACE--guardcam-ws.modal.run
Use WebSocket Secure with the same host (Expo accepts wss://):
  wss://YOUR_WORKSPACE--guardcam-ws.modal.run

Ephemeral dev server (Ctrl+C to stop):
  modal serve modal_app.py

See DEPLOY_MODAL.md for details.
"""
from __future__ import annotations

from pathlib import Path

import modal

STREAM_SERVER = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(STREAM_SERVER / "requirements.txt"))
    .pip_install("starlette>=0.27")
    .add_local_dir(STREAM_SERVER, remote_path="/guardcam")
)

app = modal.App("guardcam-stream")


# Default Modal function timeout is 300s — WebSockets stay open one invocation; raise for streaming.
@app.function(image=image, timeout=60 * 60 * 2)  # 2 hours per phone session
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="guardcam-ws")
def guardcam_asgi():
    import sys

    if "/guardcam" not in sys.path:
        sys.path.insert(0, "/guardcam")
    from asgi_app import build_app

    return build_app()
