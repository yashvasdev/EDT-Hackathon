# GuardCam `stream-server`

Python WebSocket server: receives base64 JPEG frames from the Expo app, runs YOLO drowsiness classification, returns JSON (`drowsy`, `alert`, `confidence`, …). See [TASK_DIVISION.md](../TASK_DIVISION.md) for the API contract.

## Local run

```bash
cd stream-server
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

- **Same Wi‑Fi:** `ws://<your-LAN-IP>:8765` with `python server.py`.
- **Public internet:** deploy on [Modal](https://modal.com) — [DEPLOY_MODAL.md](DEPLOY_MODAL.md) — Yash uses `wss://...modal.run`.

## Smoke test

With `server.py` running:

```bash
source venv/bin/activate
python scripts/smoke_test.py --url ws://127.0.0.1:8765
```

## Automated E2E (no phone)

Runs two subprocess servers with `GUARDCAM_E2E=1` (skips YOLO, forces valid JPEG → drowsy) to verify **3-frame alert**, **JSON contract**, and **back camera + `DROWSINESS_ONLY_FRONT`**. Safe for CI; **never set `GUARDCAM_E2E` in production**.

```bash
source venv/bin/activate
python scripts/e2e_ws_test.py
```

## ASGI / Modal (Starlette)

The same session logic runs behind **Starlette** for Modal (`asgi_app.py`, `modal_app.py`). Regression tests:

```bash
source venv/bin/activate
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `PORT` | `8765` | WebSocket listen port |
| `DROWSY_THRESHOLD` | `0.6` | Min top-1 confidence to count a frame as drowsy |
| `CONSECUTIVE_FRAMES` | `3` | Consecutive drowsy frames required before `alert: true` |
| `ALERT_COOLDOWN_SEC` | `5` | Minimum seconds between alerts |
| `DROWSINESS_ONLY_FRONT` | `0` | Set to `1` to skip YOLO on `camera: back` (road) and treat as awake |
| `GUARDCAM_E2E` | `0` | Test only: skip model load; use with `scripts/e2e_ws_test.py` — **do not deploy** |
| `GUARDCAM_BACKEND` | `mediapipe` | `yolo` = HuggingFace `mosesb/drowsiness-detection-yolo-cls`; default matches `drowsyness-detection/detect.py` heuristics. Modal: set when running `modal deploy` (see `modal_app.py`) |

## Integration (hand off to mobile)

See [HANDOFF.md](HANDOFF.md). Modal deploy: [DEPLOY_MODAL.md](DEPLOY_MODAL.md).
