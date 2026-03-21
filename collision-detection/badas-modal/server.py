"""
Shared FastAPI application factory and frame processing logic for BADAS inference.

Used by both the Modal deployment (app.py) and the local server (local_server.py).
"""

import collections
import json
import logging

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from badas.utils.video import apply_temperature_scaling

logger = logging.getLogger(__name__)

TARGET_FPS = 8.0
WINDOW_SIZE = 16


class FrameBuffer:
    """Rolling buffer of preprocessed frames for sliding window inference."""

    def __init__(self, maxlen: int = WINDOW_SIZE):
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=maxlen)
        self._frame_count = 0
        self._stride = 1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def stride(self) -> int:
        return self._stride

    @stride.setter
    def stride(self, value: int) -> None:
        self._stride = max(1, int(value))

    @property
    def ready(self) -> bool:
        return len(self._buffer) >= self._buffer.maxlen

    @property
    def should_predict(self) -> bool:
        return self.ready and (self._frame_count % self._stride == 0)

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a preprocessed (224x224 RGB uint8) frame to the buffer."""
        self._buffer.append(frame)
        self._frame_count += 1

    def get_frames(self) -> list[np.ndarray]:
        """Return current buffer contents as a list."""
        return list(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._frame_count = 0


def decode_jpeg_frame(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes into a 224x224 RGB uint8 numpy array."""
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Failed to decode JPEG frame")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    return frame_resized


def run_inference(model, frames: list[np.ndarray], device: str) -> dict:
    """Run BADAS inference on a 16-frame window. Returns prediction dict."""
    frames_array = np.array(frames)  # (16, 224, 224, 3) uint8

    # Preprocess using the model's processor or manual transform
    if model.processor:
        try:
            inputs = model.processor(videos=frames_array, return_tensors="pt")
            if "pixel_values_videos" in inputs:
                video_tensor = inputs["pixel_values_videos"].squeeze(0)
            elif "pixel_values" in inputs:
                video_tensor = inputs["pixel_values"].squeeze(0)
            else:
                video_tensor = list(inputs.values())[0].squeeze(0)
        except Exception:
            logger.warning("Processor failed, falling back to manual transform")
            video_tensor = model._manual_transform_frames(frames_array)
    else:
        video_tensor = model._manual_transform_frames(frames_array)

    # Add batch dim, move to device
    if video_tensor.dim() == 4:
        video_tensor = video_tensor.unsqueeze(0)
    video_tensor = video_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model.model(video_tensor)
        outputs_scaled = apply_temperature_scaling(outputs, temperature=2.0)
        probs = torch.softmax(outputs_scaled, dim=1)[:, 1].cpu().numpy()
        probability = float(probs[0])

    risk_level = "high" if probability >= 0.7 else "medium" if probability >= 0.4 else "low"
    return {
        "probability": probability,
        "risk_level": risk_level,
        "buffer_size": len(frames),
    }


def create_app(model, device: str) -> FastAPI:
    """Create a FastAPI app wired to the given BADAS model."""
    app = FastAPI(title="BADAS Collision Prediction")

    @app.get("/health")
    async def health():
        return {"status": "ok", "device": device}

    @app.websocket("/ws/predict")
    async def predict_ws(websocket: WebSocket):
        await websocket.accept()
        buffer = FrameBuffer()
        client = websocket.client
        client_addr = f"{client.host}:{client.port}" if client else "unknown"
        logger.info("Device connected: %s", client_addr)
        logger.info("Headers: %s", dict(websocket.headers))

        try:
            while True:
                message = await websocket.receive()

                if "bytes" in message and message["bytes"]:
                    # Binary message: JPEG frame
                    try:
                        frame = decode_jpeg_frame(message["bytes"])
                    except ValueError:
                        logger.warning("Failed to decode JPEG frame from %s", client_addr)
                        await websocket.send_json({"error": "Failed to decode frame"})
                        continue

                    buffer.add_frame(frame)

                    if buffer.should_predict:
                        prediction = run_inference(model, buffer.get_frames(), device)
                        prediction["frame_index"] = buffer.frame_count
                        prediction["timestamp"] = buffer.frame_count / TARGET_FPS
                        logger.info(
                            "Prediction for %s [frame %d]: risk=%s prob=%.3f",
                            client_addr,
                            buffer.frame_count,
                            prediction["risk_level"],
                            prediction["probability"],
                        )
                        await websocket.send_json(prediction)

                elif "text" in message and message["text"]:
                    # Text message: control command
                    try:
                        control = json.loads(message["text"])
                        logger.info("Control message from %s: %s", client_addr, control)
                        if "stride" in control:
                            buffer.stride = control["stride"]
                            await websocket.send_json(
                                {"control": "ok", "stride": buffer.stride}
                            )
                        if "reset" in control and control["reset"]:
                            buffer.reset()
                            await websocket.send_json({"control": "ok", "reset": True})
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON from %s: %s", client_addr, message["text"])
                        await websocket.send_json({"error": "Invalid JSON"})

        except WebSocketDisconnect:
            logger.info(
                "Device disconnected: %s (processed %d frames)", client_addr, buffer.frame_count
            )

    return app
