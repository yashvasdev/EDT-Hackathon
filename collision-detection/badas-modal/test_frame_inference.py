"""
Test that BADAS inference works on raw frame batches (not file paths).

This validates the core approach for the WebSocket server:
1. Read frames from a video with cv2
2. Buffer 16 frames in a deque
3. Run inference directly using the model's internal transform + forward pass
4. Compare results against the standard file-based predict() output
"""

import sys
import time
import collections
import numpy as np
import cv2
import torch

sys.path.insert(0, "../badas-uv")

from badas import load_badas_model
from badas.utils.video import apply_temperature_scaling


def extract_frames_from_video(video_path: str, target_fps: float = 8.0):
    """Extract frames from video at target FPS, yielding numpy arrays (H, W, 3) RGB uint8."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(original_fps / target_fps)))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            # BGR -> RGB, resize to 224x224
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            yield frame_resized
        frame_idx += 1

    cap.release()


def simulate_websocket_frames(video_path: str, target_fps: float = 8.0):
    """Simulate what a WebSocket client would send: JPEG-encoded bytes per frame.
    Returns decoded frames (simulating server-side decode)."""
    for frame_rgb in extract_frames_from_video(video_path, target_fps):
        # Encode to JPEG (what client would send)
        _, jpeg_bytes = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        # Decode on server side (what server would do)
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        decoded_resized = cv2.resize(decoded_rgb, (224, 224))

        yield decoded_resized


def run_inference_on_buffer(model, frames: list, device: str) -> float:
    """Run BADAS inference on a 16-frame buffer. Returns collision probability."""
    frames_array = np.array(frames)  # (16, 224, 224, 3) uint8

    # Use the model's preprocessing (albumentations: resize, normalize, to_tensor)
    if model.processor:
        try:
            inputs = model.processor(videos=frames_array, return_tensors="pt")
            if "pixel_values_videos" in inputs:
                video_tensor = inputs["pixel_values_videos"].squeeze(0)
            elif "pixel_values" in inputs:
                video_tensor = inputs["pixel_values"].squeeze(0)
            else:
                video_tensor = list(inputs.values())[0].squeeze(0)
        except Exception as e:
            print(f"Processor failed ({e}), falling back to manual transform")
            video_tensor = model._manual_transform_frames(frames_array)
    else:
        video_tensor = model._manual_transform_frames(frames_array)

    # Add batch dim and move to device
    if video_tensor.dim() == 4:
        video_tensor = video_tensor.unsqueeze(0)  # (1, T, C, H, W)
    video_tensor = video_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model.model(video_tensor)
        outputs_scaled = apply_temperature_scaling(outputs, temperature=2.0)
        probs = torch.softmax(outputs_scaled, dim=1)[:, 1].cpu().numpy()
        return float(probs[0])


def main():
    video_path = "../sample_video.mp4"
    device = "cpu"
    window_size = 16

    # Load model
    print(f"Loading BADAS model on {device}...")
    t0 = time.time()
    model = load_badas_model(device=device)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # -- Test 1: Direct frame buffer inference --
    print("\n" + "=" * 60)
    print("TEST 1: Direct frame buffer inference")
    print("=" * 60)

    buffer = collections.deque(maxlen=window_size)
    predictions = []

    print("Extracting frames and running sliding window inference...")
    t0 = time.time()

    for i, frame in enumerate(extract_frames_from_video(video_path)):
        buffer.append(frame)
        if len(buffer) == window_size:
            prob = run_inference_on_buffer(model, list(buffer), device)
            predictions.append(prob)
            risk = "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low"
            timestamp = i / 8.0  # 8 FPS
            if i % 5 == 0:
                print(f"  Frame {i:4d} ({timestamp:6.2f}s) | prob={prob:.4f} | risk={risk}")

    elapsed = time.time() - t0
    print(f"\nDirect inference: {len(predictions)} windows in {elapsed:.1f}s")

    if predictions:
        preds = np.array(predictions)
        print(f"  Mean probability: {np.mean(preds):.4f}")
        print(f"  Max probability:  {np.max(preds):.4f}")
        print(f"  Min probability:  {np.min(preds):.4f}")

    # -- Test 2: JPEG round-trip (simulating WebSocket) --
    print("\n" + "=" * 60)
    print("TEST 2: JPEG round-trip (simulating WebSocket)")
    print("=" * 60)

    buffer_ws = collections.deque(maxlen=window_size)
    predictions_ws = []

    t0 = time.time()
    for i, frame in enumerate(simulate_websocket_frames(video_path)):
        buffer_ws.append(frame)
        if len(buffer_ws) == window_size:
            prob = run_inference_on_buffer(model, list(buffer_ws), device)
            predictions_ws.append(prob)

    elapsed = time.time() - t0
    print(f"WebSocket-simulated inference: {len(predictions_ws)} windows in {elapsed:.1f}s")

    if predictions_ws:
        preds_ws = np.array(predictions_ws)
        print(f"  Mean probability: {np.mean(preds_ws):.4f}")
        print(f"  Max probability:  {np.max(preds_ws):.4f}")
        print(f"  Min probability:  {np.min(preds_ws):.4f}")

    # -- Test 3: Compare against standard predict() --
    print("\n" + "=" * 60)
    print("TEST 3: Compare against standard file-based predict()")
    print("=" * 60)

    t0 = time.time()
    standard_preds = model.predict(video_path)
    elapsed = time.time() - t0
    valid = standard_preds[~np.isnan(standard_preds)]
    print(f"Standard predict(): {len(valid)} valid predictions in {elapsed:.1f}s")
    if len(valid) > 0:
        print(f"  Mean probability: {np.mean(valid):.4f}")
        print(f"  Max probability:  {np.max(valid):.4f}")
        print(f"  Min probability:  {np.min(valid):.4f}")

    # -- Summary --
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if predictions and len(valid) > 0:
        direct_mean = np.mean(predictions)
        standard_mean = np.mean(valid)
        diff = abs(direct_mean - standard_mean)
        print(f"Direct buffer mean:   {direct_mean:.4f}")
        print(f"Standard predict mean: {standard_mean:.4f}")
        print(f"Difference:            {diff:.4f}")
        if diff < 0.15:
            print("PASS -- Results are reasonably close")
        else:
            print("WARN -- Results diverge significantly (may be due to different frame sampling)")

    if predictions_ws:
        ws_mean = np.mean(predictions_ws)
        direct_mean = np.mean(predictions)
        jpeg_diff = abs(ws_mean - direct_mean)
        print(f"\nJPEG round-trip vs direct: {jpeg_diff:.4f} difference")
        if jpeg_diff < 0.05:
            print("PASS -- JPEG compression has minimal impact")
        else:
            print("WARN -- JPEG compression is affecting predictions")


if __name__ == "__main__":
    main()
