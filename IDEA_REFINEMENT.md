# 🚛 GuardCam — Smart Phone Dashcam for Driver Safety

> **Hackathon Theme:** A Better Tomorrow
>
> **Elevator Pitch:** Turn any smartphone into a two-way AI dashcam that streams to a computer running real-time ML models — protecting drivers from drowsiness AND road hazards.

---

## 🎯 Problem Statement

Long-haul truck drivers and rideshare drivers face two critical dangers:

1. **Drowsy driving** — one of the leading causes of fatal crashes, responsible for ~100,000 crashes/year in the US alone (NHTSA).
2. **Erratic nearby drivers** — vehicles swerving into their lane, sudden braking, or aggressive lane changes leave little reaction time.

Existing dashcams only *record* incidents. They don't *prevent* them. Commercial drowsiness detection systems (like those from Mobileye or Seeing Machines) cost **$500–$2,000+** and require dedicated hardware.

**Our solution costs $0 in extra hardware** — just mount your phone and connect to a laptop.

---

## 💡 Core Concept

Mount a smartphone on the dashboard so that:

| Camera | Faces | Purpose |
|--------|-------|---------|
| **Front (selfie) camera** | The driver | Drowsiness & distraction detection |
| **Back (main) camera** | The road | Lane intrusion & collision threat detection |

The phone **streams both camera feeds** over WiFi/USB to a computer, which runs the **AI models** and sends **real-time audible alerts** back to the phone (or plays them on the computer's speakers).

---

## 🔑 Key Features

### 1. Drowsiness Detection (Front Camera → Driver)

- **YOLO Classification Model** — a pretrained YOLOv8 classification model (`mosesb/drowsiness-detection-yolo-cls`) that classifies frames as "Drowsy" or "Non Drowsy" (**already working! see below**)
- **Eye Aspect Ratio (EAR) tracking** — complementary detection for when eyes close for too long
- **Yawn detection** — detect open-mouth yawns as early drowsiness signs
- **Head pose estimation** — detect head nodding or tilting
- **Graduated alert system:**
  - ⚠️ *Mild drowsiness* → gentle chime + "Stay alert" voice prompt
  - 🚨 *Severe drowsiness* → loud alarm + "Pull over immediately" voice prompt
  - 📍 Optionally suggest nearest rest stop

### 2. Road Hazard Detection (Back Camera → Road)

- **Lane boundary detection** — understand where your lane is
- **Vehicle tracking** — detect and track nearby vehicles
- **Swerve/intrusion detection** — alert when another vehicle is crossing into your lane
- **Tailgating warning** — alert if a vehicle behind is dangerously close
- **Graduated alert system:**
  - ⚠️ *Vehicle drifting* → "Vehicle approaching from [left/right]"
  - 🚨 *Imminent collision risk* → loud alarm

### 3. Incident Logging (Stretch Goal)

- Auto-save a short video clip when an alert is triggered
- Timestamp + GPS location logging
- Exportable incident report

### 4. Dashboard / Stats (Stretch Goal)

- Trip summary: drowsiness events, road hazards detected
- Drowsiness score over time
- Heat map of where incidents occurred

---

## 🏗️ High-Level Architecture (Client-Server)

The phone acts as a **camera streamer** only. All heavy ML processing runs on a **computer** (laptop/desktop).

```
┌──────────────────────┐          ┌─────────────────────────────────────┐
│   📱 PHONE (Client)  │          │   💻 COMPUTER (Server)              │
│                      │          │                                     │
│  ┌────────────────┐  │  WiFi /  │  ┌─────────────────────────────┐   │
│  │ Front Camera   │──┼──USB ──▶│  │ Face/Eye Detection          │   │
│  │ (Selfie→Driver)│  │ Stream  │  │ (MediaPipe / dlib)          │   │
│  └────────────────┘  │          │  └──────────┬──────────────────┘   │
│                      │          │             │                      │
│  ┌────────────────┐  │  WiFi /  │  ┌──────────▼──────────────────┐   │
│  │ Back Camera    │──┼──USB ──▶│  │ Vehicle/Lane Detection      │   │
│  │ (Main→Road)    │  │ Stream  │  │ (YOLOv8 / OpenCV)           │   │
│  └────────────────┘  │          │  └──────────┬──────────────────┘   │
│                      │          │             │                      │
│  ┌────────────────┐  │  ◀──────│  ┌──────────▼──────────────────┐   │
│  │ 🔊 Speaker     │  │  Alert  │  │ ALERT ENGINE                │   │
│  │ (plays alarm)  │  │  Signal │  │ - Threat assessment         │   │
│  └────────────────┘  │          │  │ - Alert type selection      │   │
│                      │          │  │ - Incident clip saving      │   │
│  ┌────────────────┐  │  ◀──────│  └─────────────────────────────┘   │
│  │ 📊 Status UI   │  │  Status │                                     │
│  │ (optional)     │  │  Update │  ┌─────────────────────────────┐   │
│  └────────────────┘  │          │  │ 🖥️ DASHBOARD (optional)    │   │
│                      │          │  │ - Live annotated feeds      │   │
└──────────────────────┘          │  │ - Alert log                 │   │
                                  │  └─────────────────────────────┘   │
                                  └─────────────────────────────────────┘
```

### Streaming Options

| Method | Pros | Cons |
|--------|------|------|
| **IP Webcam (Android app)** | Free, easy, streams via WiFi | Latency ~100-300ms, WiFi needed |
| **DroidCam / Iriun** | Acts as USB webcam on PC | Usually single camera only |
| **Custom app (WebRTC/RTSP)** | Full control, dual stream | More dev work |
| **USB tethering + ADB** | Low latency, no WiFi needed | Android only, needs USB cable |

---

## 🛠️ Recommended Tech Stack

Since ML runs on the computer, we can use **full-power models** and **Python** for the backend.

| Component | Technology | Notes |
|-----------|-----------|-------|
| **Phone Streaming** | IP Webcam app or custom Android app | Streams both cameras via WiFi/USB |
| **Server (Python)** | Python + OpenCV | Receives video streams, runs ML |
| **Drowsiness Detection** | MediaPipe Face Mesh + dlib | Eye Aspect Ratio, yawn, head pose |
| **Road Detection** | YOLOv8 (Ultralytics) + OpenCV | Vehicle detection, lane tracking |
| **Audio Alerts** | pygame / playsound (on computer) or send alert back to phone | Alarm sounds + voice prompts |
| **Dashboard UI** | Flask/FastAPI web dashboard or simple OpenCV window | Shows annotated camera feeds + alerts |

---

## ⚠️ Concerns & Technical Risks

### ✅ Resolved by Client-Server Architecture

1. ~~**Dual camera simultaneous access**~~ → **RESOLVED.** Since the phone just streams video, we can use existing apps (IP Webcam) or a simple custom app to stream both feeds. No need for complex multi-camera APIs.

2. ~~**Processing power & heat**~~ → **RESOLVED.** All ML runs on the computer. Phone just streams video — minimal battery/heat impact (still needs car charger for long trips).

### 🔴 Critical (New Concerns)

3. **Streaming latency** — Video streaming over WiFi adds 100-300ms latency on top of ML inference time.
   - Total round-trip (stream + inference + alert) should stay under ~500ms.
   - **Mitigation:** Use USB tethering instead of WiFi, compress frames, keep resolution reasonable (720p is enough).

4. **WiFi reliability in a vehicle** — Need a stable connection between phone and laptop.
   - **Mitigation:** Use USB connection (ADB port forwarding) instead of WiFi for reliability. Or create a phone hotspot → laptop connects to it.

### 🟡 Important

4. **Phone mounting & vibration** — Trucks vibrate a lot. The phone mount must be stable.
   - Vibration could cause false positives in face detection.
   - **Mitigation:** Frame smoothing / rolling average to filter out vibration noise.

5. **Lighting conditions** — Nighttime driving, sun glare, tunnel transitions.
   - Front camera: IR-based detection won't work on phones (no IR camera). Need to handle low-light.
   - Back camera: Headlights, rain, fog can degrade detection.
   - **Mitigation:** Image preprocessing (histogram equalization, adaptive brightness).

6. **False positives/negatives** — A false alarm while driving is a distraction. A missed alarm is a safety failure.
   - Need careful threshold tuning.
   - **Mitigation:** Graduated alert system, requiring sustained detection before alarming.

### 🟢 Minor

7. **Legal considerations** — Recording/monitoring drivers may have legal implications in some jurisdictions.
8. **Phone placement uniformity** — Different mounting positions affect camera angles.

---

## 🔬 Current Progress (What's Been Built)

Your teammate has already started building the **drowsiness detection** component!

### `drowsyness-detection/` — YOLO Drowsiness Classifier

| Item | Details |
|------|---------|
| **Python version** | 3.12 |
| **Package manager** | `uv` (lockfile present) |
| **Key dependencies** | `ultralytics` (YOLOv8), `kagglehub`, `huggingface-hub` |
| **Dataset** | [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) from Kaggle — face images labeled "Drowsy" / "Non Drowsy" |
| **Model** | Pretrained YOLOv8 classification model from HuggingFace: `mosesb/drowsiness-detection-yolo-cls` |
| **Status** | ✅ Working! Tested on sample image — predicted "Drowsy" with **100% confidence** |

#### What the notebook does (`data_exploration.ipynb`):
1. Downloads the Kaggle drowsiness dataset
2. Downloads a pretrained YOLO classification model from HuggingFace
3. Runs inference on a sample driver face image
4. Successfully classifies it as "Drowsy" with 1.0 confidence

#### What's still needed:
- [ ] Connect this model to a **live camera stream** (receive frames from phone)
- [ ] Integrate **audio alarm** when drowsiness is detected
- [ ] Add frame-by-frame processing loop (not just single image)
- [ ] Handle consecutive frame logic (e.g., alert only after N drowsy frames in a row)
- [ ] Package into a runnable Python server script (not just a notebook)

---

## ❓ Remaining Questions

These don't block us from starting, but answers would help prioritize:

1. **How long is the hackathon?** — Determines scope of what we can build.
    5 hours, need to be done asap 
3. **Will you have a laptop available during the demo?** — Needed for the server.
yes
4. **Back camera faces the road ahead (through windshield), correct?** — Just confirming the mount orientation.
yes

---

## 🏆 What Makes This a Winning Hackathon Entry

- **Theme alignment** — "A Better Tomorrow" = saving lives on the road
- **Accessibility** — No special hardware, just a phone everyone already owns
- **Impact** — Targets truckers, rideshare drivers, long-commute workers
- **Scalability story** — Could expand to fleet management, insurance discounts, etc.
- **Demo-ability** — Drowsiness detection is visually impressive in a live demo

---

## 📋 Updated MVP Scope

### Already Done ✅
- [x] Drowsiness classification model (YOLO) — working, tested on dataset
- [x] Dataset sourced (Kaggle Driver Drowsiness Dataset)
- [x] Python project setup with `uv`

### Must Build Next 🔴
- [ ] Phone → Computer camera streaming setup (IP Webcam or custom app)
this is what I want to start working on right now, help me set up a streamining process from the phone to the computer, I am not really sure where the stream needs to go to right now but for now just to try to get the stream from mobile up, I want to use React Native to build a simple app that can stream the camera feed to the computer, I am not really sure how to do this, so I need your help to set it up, I have a mac and the computer is a windows machine, I am not sure if this matters but I thought I should mention it, I know there is two ways to stream, Web RTC and web sockets, I know web rtc is really really complicated and I want to stay away from this essepcially because we have like 4 -5 ish hours, also how would I test if this stream is working?

- [ ] Live frame processing server (Python script that receives stream + runs YOLO model)
- [ ] Audio alarm trigger when drowsiness detected
- [ ] Consecutive frame logic (alert after N drowsy frames, not single frame)

### Should Have 🎯
- [ ] Back camera road monitoring (vehicle detection with YOLOv8)
- [ ] Lane departure / vehicle intrusion alert
- [ ] Simple dashboard UI showing live feeds + alert status

### Nice to Have 💫
- [ ] Incident clip saving
- [ ] Trip summary dashboard
- [ ] GPS integration for rest stop suggestions
