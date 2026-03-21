"""
WebSocket test client for BADAS inference server.

Reads frames from a video file, encodes as JPEG, sends over WebSocket,
and prints predictions as they arrive.

Usage:
    # From the badas-modal directory:
    uv run python test_client.py
    uv run python test_client.py --url wss://your-modal-endpoint.modal.run/ws/predict
    uv run python test_client.py --video path/to/video.mp4 --fps 8
"""

import argparse
import asyncio
import json

import cv2
import websockets


async def send_frames(ws, video_path: str, fps: float, done_event: asyncio.Event):
    """Read video frames and send as JPEG over WebSocket at target FPS."""
    frame_interval = 1.0 / fps

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        done_event.set()
        return 0

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(round(original_fps / fps)))

    print(f"Video: {video_path}")
    print(f"Original FPS: {original_fps:.1f}, sampling every {sample_interval} frames")
    print(
        f"Total frames: {total_frames}, estimated sampled: {total_frames // sample_interval}"
    )

    frame_idx = 0
    sent_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            _, jpeg = cv2.imencode(".jpg", frame)
            await ws.send(jpeg.tobytes())
            sent_count += 1
            await asyncio.sleep(frame_interval)

        frame_idx += 1

    cap.release()
    print(f"\nAll {sent_count} frames sent. Waiting for remaining predictions...")
    done_event.set()
    return sent_count


async def receive_predictions(ws, done_event: asyncio.Event):
    """Receive and print predictions as they arrive."""
    count = 0
    while True:
        try:
            # While still sending, wait indefinitely for responses.
            # After sending is done, use a timeout to detect when the server
            # has finished processing.
            timeout = None if not done_event.is_set() else 5.0
            response = await asyncio.wait_for(ws.recv(), timeout=timeout)
            pred = json.loads(response)

            if "probability" in pred:
                count += 1
                ts = pred.get("timestamp", 0)
                prob = pred["probability"]
                risk = pred["risk_level"]
                fi = pred.get("frame_index", "?")
                print(
                    f"  [{count:3d}] frame={fi:<4} ({ts:6.2f}s) | "
                    f"prob={prob:.4f} | risk={risk}"
                )
            elif "error" in pred:
                print(f"  ERROR: {pred['error']}")
            elif "control" in pred:
                print(f"  CONTROL: {pred}")

        except asyncio.TimeoutError:
            break
        except websockets.exceptions.ConnectionClosed:
            break

    return count


async def stream_video(url: str, video_path: str, fps: float):
    print(f"Connecting to {url}...")

    async with websockets.connect(url, open_timeout=60) as ws:
        print("Connected. Streaming frames...\n")

        done_event = asyncio.Event()

        sender = asyncio.create_task(send_frames(ws, video_path, fps, done_event))
        receiver = asyncio.create_task(receive_predictions(ws, done_event))

        sent_count = await sender
        recv_count = await receiver

        print(f"\nDone. Sent {sent_count} frames, received {recv_count} predictions.")


def main():
    parser = argparse.ArgumentParser(description="BADAS WebSocket test client")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/ws/predict",
        help="WebSocket URL (default: ws://localhost:8000/ws/predict)",
    )
    parser.add_argument(
        "--video",
        default="../sample_video.mp4",
        help="Path to video file (default: ../sample_video.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Target send FPS (default: 8.0)",
    )
    args = parser.parse_args()

    asyncio.run(stream_video(args.url, args.video, args.fps))


if __name__ == "__main__":
    main()
