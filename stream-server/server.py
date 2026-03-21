"""
GuardCam Cloud Server
Receives camera frames from phone via WebSocket, runs YOLO drowsiness detection,
and sends alerts back to the phone.

Runs headless (no display needed) — deploy to any cloud or run locally.

Usage:
    python server.py

Environment variables:
    PORT - Server port (default: 8765)
    DROWSY_THRESHOLD - Min confidence for "drowsy" class (default: 0.6)
    CONSECUTIVE_FRAMES - Drowsy frames in a row before alert (default: 3)
    ALERT_COOLDOWN_SEC - Seconds before another alert (default: 5)
    DROWSINESS_ONLY_FRONT - If 1/true, skip YOLO on back camera (road) frames (default: 0)
    GUARDCAM_E2E - If 1/true, skip model load; valid JPEG frames classify as drowsy (for automated E2E tests only)
    GUARDCAM_BACKEND - mediapipe (default) | yolo — mediapipe matches drowsyness-detection/detect.py heuristics; yolo = HuggingFace classifier
"""

import asyncio
import base64
import json
import os
import sys
import time
from collections import deque
from typing import AsyncIterator, Awaitable, Callable

import numpy as np

# Try to import YOLO — graceful fallback if not installed yet
try:
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics/huggingface not installed — running in DEMO mode (no real detection)")

import websockets


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


# Configuration
PORT = _env_int("PORT", 8765)
DROWSY_THRESHOLD = _env_float("DROWSY_THRESHOLD", 0.6)
CONSECUTIVE_FRAMES = _env_int("CONSECUTIVE_FRAMES", 3)
ALERT_COOLDOWN_SEC = _env_float("ALERT_COOLDOWN_SEC", 5.0)
DROWSINESS_ONLY_FRONT = _env_bool("DROWSINESS_ONLY_FRONT", False)
GUARDCAM_E2E = _env_bool("GUARDCAM_E2E", False)
GUARDCAM_BACKEND = os.environ.get("GUARDCAM_BACKEND", "mediapipe").strip().lower()
if GUARDCAM_BACKEND not in ("yolo", "mediapipe"):
    GUARDCAM_BACKEND = "mediapipe"

# Model (loaded on startup)
model = None


def load_model():
    """Download and load the YOLO drowsiness classification model (skipped for mediapipe backend)."""
    global model
    if GUARDCAM_BACKEND == "mediapipe":
        print("📷 Backend: MediaPipe (no YOLO download — Face Mesh per connection)")
        return

    if not YOLO_AVAILABLE:
        print("⚠️  Skipping model load (ultralytics not installed)")
        return

    print("📥 Downloading YOLO drowsiness model from HuggingFace...")
    try:
        model_path = hf_hub_download(
            repo_id='mosesb/drowsiness-detection-yolo-cls',
            filename='best.pt'
        )
        model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Running in DEMO mode (random predictions)")


def analyze_frame(frame_base64):
    """Run drowsiness detection on a base64 JPEG frame.

    Returns: (is_drowsy: bool, confidence: float, class_name: str)
    """
    if GUARDCAM_E2E:
        try:
            if "," in frame_base64:
                frame_base64 = frame_base64.split(",")[1]
            img_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            import cv2
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return False, 0.0, "Error"
            return True, 0.95, "Drowsy"
        except Exception as e:
            print(f"  Analysis error: {e}")
            return False, 0.0, "Error"

    if model is None:
        # Demo mode — alternate between drowsy/not for testing
        import random
        is_drowsy = random.random() > 0.7
        conf = random.uniform(0.5, 1.0)
        return is_drowsy, conf, "Drowsy" if is_drowsy else "Non Drowsy"

    try:
        # Decode base64 to image
        if "," in frame_base64:
            frame_base64 = frame_base64.split(",")[1]

        img_bytes = base64.b64decode(frame_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)

        # YOLO expects file path or numpy array — use cv2 to decode
        import cv2
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return False, 0.0, "Error"

        # Run inference
        results = model.predict(frame, verbose=False)
        probs = results[0].probs

        top1_idx = probs.top1
        top1_conf = float(probs.top1conf)
        class_name = model.names[top1_idx]

        is_drowsy = (class_name.lower() == "drowsy") and (top1_conf >= DROWSY_THRESHOLD)

        return is_drowsy, top1_conf, class_name

    except Exception as e:
        print(f"  Analysis error: {e}")
        return False, 0.0, "Error"


async def process_session(
    client_addr: str,
    messages: AsyncIterator[str],
    send_text: Callable[[str], Awaitable[None]],
) -> None:
    """Core session loop: JSON frames in, JSON lines out (WebSocket-agnostic)."""
    print(f"\n✅ Phone connected from {client_addr}")

    mp_tracker = None
    if (not GUARDCAM_E2E) and GUARDCAM_BACKEND == "mediapipe":
        from mediapipe_backend import MediaPipeDrowsinessTracker

        mp_tracker = MediaPipeDrowsinessTracker()

    drowsy_streak: deque[bool] = deque(maxlen=CONSECUTIVE_FRAMES)
    last_alert_time = 0.0
    frame_count = 0
    start_time = time.time()

    try:
        async for message in messages:
            try:
                data = json.loads(message)
                frame_data = data.get("frame", "")
                camera = data.get("camera", "front")

                if not frame_data:
                    continue

                frame_count += 1

                cam = str(camera).lower()
                if DROWSINESS_ONLY_FRONT and cam == "back":
                    is_drowsy, confidence, class_name = False, 0.0, "Non Drowsy"
                elif GUARDCAM_E2E:
                    is_drowsy, confidence, class_name = analyze_frame(frame_data)
                elif mp_tracker is not None:
                    is_drowsy, confidence, class_name = mp_tracker.analyze_base64_jpeg(
                        frame_data
                    )
                else:
                    is_drowsy, confidence, class_name = analyze_frame(frame_data)

                drowsy_streak.append(is_drowsy)

                should_alert = False
                now = time.time()

                if (len(drowsy_streak) >= CONSECUTIVE_FRAMES and
                        all(drowsy_streak) and
                        (now - last_alert_time) > ALERT_COOLDOWN_SEC):
                    should_alert = True
                    last_alert_time = now

                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                response = json.dumps({
                    "drowsy": is_drowsy,
                    "alert": should_alert,
                    "confidence": round(confidence, 3),
                    "class": class_name,
                    "camera": camera,
                    "fps": round(fps, 1),
                    "frame_num": frame_count,
                })

                await send_text(response)

                if frame_count % 10 == 0:
                    status = "🚨 DROWSY" if is_drowsy else "✅ Awake"
                    print(f"  Frame {frame_count} | {status} ({confidence:.1%}) | {fps:.1f} FPS")

                if should_alert:
                    print(f"  🚨🚨🚨 ALERT SENT — Driver appears drowsy! ({confidence:.1%})")

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"  Frame processing error: {e}")
                continue
    finally:
        if mp_tracker is not None:
            try:
                mp_tracker.close()
            except Exception:
                pass
        elapsed = time.time() - start_time
        print(f"\n📱 Phone disconnected. Processed {frame_count} frames in {elapsed:.1f}s")


async def handle_client(websocket):
    """websockets library handler (local `python server.py`)."""

    async def message_iter():
        async for m in websocket:
            yield m

    try:
        await process_session(str(websocket.remote_address), message_iter(), websocket.send)
    except websockets.exceptions.ConnectionClosed:
        pass


async def main():
    """Start the WebSocket server."""
    if GUARDCAM_E2E:
        print("🧪 GUARDCAM_E2E=1 — skipping YOLO load (automated tests only)")
    else:
        load_model()

    print()
    print("=" * 50)
    print("  🚛 GuardCam Cloud Server")
    print("=" * 50)
    print()
    print(f"  WebSocket server running on port {PORT}")
    print(f"  Connect from phone: ws://<YOUR-SERVER-IP>:{PORT}")
    print()
    print(f"  Detection settings:")
    print(f"    Drowsy threshold: {DROWSY_THRESHOLD:.0%}")
    print(f"    Consecutive frames: {CONSECUTIVE_FRAMES}")
    print(f"    Alert cooldown: {ALERT_COOLDOWN_SEC}s")
    print(f"    Drowsiness only on front camera: {DROWSINESS_ONLY_FRONT}")
    print(f"    E2E test mode: {GUARDCAM_E2E}")
    print(f"    Backend: {GUARDCAM_BACKEND}")
    print()
    print("  Waiting for phone to connect...")
    print("=" * 50)

    async with websockets.serve(handle_client, "0.0.0.0", PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped.")
        sys.exit(0)
