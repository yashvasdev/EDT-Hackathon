"""
GuardCam Cloud Server
Receives camera frames from phone via WebSocket, runs YOLO drowsiness detection,
and sends alerts back to the phone.

Runs headless (no display needed) — deploy to any cloud or run locally.

Usage:
    python server.py

Environment variables:
    PORT - Server port (default: 8765)
"""

import asyncio
import base64
import json
import os
import sys
import time
from collections import deque

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

# Configuration
PORT = int(os.environ.get("PORT", 8765))
DROWSY_THRESHOLD = 0.6         # Confidence threshold to consider "drowsy"
CONSECUTIVE_FRAMES = 3          # Need N consecutive drowsy frames before alerting
ALERT_COOLDOWN_SEC = 5          # Don't re-alert within this window

# Model (loaded on startup)
model = None


def load_model():
    """Download and load the YOLO drowsiness classification model."""
    global model
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


async def handle_client(websocket):
    """Handle a phone connection — receive frames, run detection, send alerts."""
    client_addr = websocket.remote_address
    print(f"\n✅ Phone connected from {client_addr}")

    # Per-client state
    drowsy_streak = deque(maxlen=CONSECUTIVE_FRAMES)
    last_alert_time = 0
    frame_count = 0
    start_time = time.time()

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                frame_data = data.get("frame", "")
                camera = data.get("camera", "front")

                if not frame_data:
                    continue

                frame_count += 1

                # Run drowsiness detection
                is_drowsy, confidence, class_name = analyze_frame(frame_data)

                # Track consecutive drowsy frames
                drowsy_streak.append(is_drowsy)

                # Check if we should alert
                should_alert = False
                now = time.time()

                if (len(drowsy_streak) >= CONSECUTIVE_FRAMES and
                    all(drowsy_streak) and
                    (now - last_alert_time) > ALERT_COOLDOWN_SEC):
                    should_alert = True
                    last_alert_time = now

                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Send result back to phone
                response = json.dumps({
                    "drowsy": is_drowsy,
                    "alert": should_alert,
                    "confidence": round(confidence, 3),
                    "class": class_name,
                    "camera": camera,
                    "fps": round(fps, 1),
                    "frame_num": frame_count,
                })

                await websocket.send(response)

                # Log periodically
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

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        elapsed = time.time() - start_time
        print(f"\n📱 Phone disconnected. Processed {frame_count} frames in {elapsed:.1f}s")


async def main():
    """Start the WebSocket server."""
    # Load YOLO model on startup
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
