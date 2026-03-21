"""
Standalone local server for BADAS collision prediction (CPU, no Modal).

Usage:
    # From the badas-uv directory:
    uv run python ../badas-modal/local_server.py

    # Or with custom options:
    uv run python ../badas-modal/local_server.py --port 8080
"""

import argparse
import logging
import sys
import time

sys.path.insert(0, "../badas-uv")


def main():
    parser = argparse.ArgumentParser(description="BADAS local inference server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--checkpoint", default=None, help="Path to custom checkpoint"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import uvicorn

    from badas import load_badas_model
    from server import create_app

    print(f"Loading BADAS model on {args.device}...")
    t0 = time.time()
    model = load_badas_model(device=args.device, checkpoint_path=args.checkpoint)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    app = create_app(model, args.device)

    print(f"Starting server on {args.host}:{args.port}")
    print(f"WebSocket endpoint: ws://{args.host}:{args.port}/ws/predict")
    print(f"Health check: http://{args.host}:{args.port}/health")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
