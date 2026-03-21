#!/usr/bin/env python3
"""
Run BADAS collision prediction inference on a video file using CPU.
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

from badas import load_badas_model, preprocess_video


def main():
    parser = argparse.ArgumentParser(
        description="Run BADAS collision prediction on CPU"
    )
    parser.add_argument(
        "video_path",
        type=str,
        nargs="?",
        default="../sample_video.mp4",
        help="Path to input video file (default: ../sample_video.mp4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Collision probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to custom checkpoint (default: download from HuggingFace)",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"Video: {video_path.resolve()}")
    print(f"Device: cpu")
    print(f"Threshold: {args.threshold}")
    print("-" * 60)

    # Load model
    print("Loading BADAS model...")
    t0 = time.time()
    model = load_badas_model(device="cpu", checkpoint_path=args.checkpoint)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Run inference
    print(f"Running inference on: {video_path.name}")
    t0 = time.time()
    predictions = model.predict(str(video_path))
    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.1f}s")

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total frame predictions: {len(predictions)}")

    # Filter out NaN values for stats
    valid_preds = predictions[~np.isnan(predictions)]
    if len(valid_preds) > 0:
        print(f"Valid predictions: {len(valid_preds)}")
        print(f"Average risk: {np.mean(valid_preds):.4f}")
        print(f"Max risk: {np.max(valid_preds):.4f}")
        print(f"Min risk: {np.min(valid_preds):.4f}")
    else:
        print("No valid predictions produced.")
        return

    # Find high-risk moments
    high_risk_indices = np.where(valid_preds >= args.threshold)[0]
    if len(high_risk_indices) > 0:
        print(
            f"\nCOLLISION WARNING: {len(high_risk_indices)} high-risk frames detected!"
        )
        for idx in high_risk_indices[:10]:
            timestamp = idx * 0.125  # 8 FPS
            print(f"  Frame {idx} ({timestamp:.1f}s): {valid_preds[idx]:.4f}")
        if len(high_risk_indices) > 10:
            print(f"  ... and {len(high_risk_indices) - 10} more")
    else:
        print(f"\nNo collision risk above threshold ({args.threshold}) detected.")

    # Print all predictions summary
    print("\nPer-frame predictions (sampled):")
    step = max(1, len(predictions) // 20)
    for i in range(0, len(predictions), step):
        val = predictions[i]
        timestamp = i * 0.125
        status = "NaN" if np.isnan(val) else f"{val:.4f}"
        print(f"  Frame {i:4d} ({timestamp:6.1f}s): {status}")


if __name__ == "__main__":
    main()
