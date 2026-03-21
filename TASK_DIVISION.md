# GuardCam — Task Division

## Architecture

```
📱 Phone (Expo App)              ☁️ Cloud Server (Python)
┌──────────────────┐             ┌──────────────────────────────┐
│ capture frame    │───internet──▶│ WebSocket server             │
│ encode base64    │             │ ↓                            │
│ send via WS      │             │ Run YOLO drowsiness model    │
│                  │             │ ↓                            │
│ receive alert  ◀──────────────│ Send result: {drowsy, alert} │
│ play alarm 🔊   │             │                              │
└──────────────────┘             └──────────────────────────────┘
```

---

## Shared API Contract (DO NOT CHANGE WITHOUT TELLING EACH OTHER)

**Phone → Server** (every ~200ms):
```json
{"frame": "<base64 JPEG>", "camera": "front", "timestamp": 1234567890}
```

**Server → Phone** (response to each frame):
```json
{"drowsy": true, "alert": true, "confidence": 0.95, "class": "Drowsy", "fps": 4.8}
```

- `drowsy` = is this single frame drowsy?
- `alert` = should phone play alarm RIGHT NOW? (true only after 3+ consecutive drowsy frames, with 5s cooldown)

---

## 👨‍💻 Yash — Phone App + Streaming

**Folder:** `camera-stream-app/`  
**Main file:** `App.js`

### What to do
1. Run `cd camera-stream-app && npx expo start`
2. Scan QR code with **Expo Go** on your Android phone
3. Grant camera permission
4. Test camera flip (front/back) works
5. Test WebSocket connection (enter a URL and tap Connect)
6. Test alert UI — temporarily set `showAlert = true` in the code to see the red "WAKE UP!" overlay
7. Once Anthony has his server running, enter his URL and do a real test

### What the app does
- Opens front camera (driver face) or back camera (road)
- Captures a frame every 200ms (~5 FPS)
- Sends base64 JPEG to server via WebSocket
- Receives `{drowsy, alert, confidence}` back
- Shows **👁️ Awake** or **😴 DROWSY** badge
- On `alert: true` → vibrates phone + shows full-screen red "WAKE UP!" overlay

---

## 👨‍💻 Anthony — Cloud Server + Models

**Folder:** `stream-server/`  
**Main file:** `server.py`

### What to do
1. Get a cloud VM (any provider) with Python 3.9+
2. Clone the repo: `git clone https://github.com/yashvasdev/EDT-Hackathon.git`
3. `cd EDT-Hackathon/stream-server`
4. `pip install -r requirements.txt`
5. `python server.py` → it prints the WebSocket URL
6. Open port **8765** in the VM's firewall / security group
7. Share the URL `ws://<your-vm-ip>:8765` with Yash
8. (Stretch) Add road/collision detection for back camera frames

### What the server does
- Listens for WebSocket connections on port 8765
- Receives base64 JPEG frames from the phone
- Downloads + loads YOLO drowsiness model from HuggingFace on first run
- Runs inference on each frame → "Drowsy" or "Non Drowsy"
- Tracks consecutive drowsy frames (needs 3 in a row)
- Sends JSON result back to the phone
- Has demo mode (random predictions) if YOLO isn't installed yet

---

## 🤝 Integration

1. Anthony gets server running → shares URL
2. Yash enters URL in app → Connect → Start Monitoring
3. Point phone at your face → should see **Awake** or **Drowsy** on screen!
4. Close your eyes for a few seconds → should trigger **WAKE UP!** alarm 🚨
