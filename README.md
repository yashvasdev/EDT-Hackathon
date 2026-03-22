# EDT-Hackathon

# 🚛 GuardCam: AI-Powered Dashcam

> **Democratizing road safety by turning your everyday smartphone into an advanced, AI-powered dual-camera system.**

![GuardCam Banner](https://img.shields.io/badge/Status-Hackathon_MVP-success) ![React Native](https://img.shields.io/badge/Frontend-React_Native_Expo-blue) ![Python](https://img.shields.io/badge/Backend-Python_MediaPipe-yellow) ![WebSockets](https://img.shields.io/badge/Streaming-WebSockets-orange)

## 📌 The Problem
According to the NHTSA, drowsy driving is responsible for over **100,000 crashes** every year. Advanced Driver Monitoring Systems (DMS) exist, but they are incredibly expensive, require professional installation, or are locked behind luxury vehicle price tags.

## 💡 Our Solution
**Software over hardware.** GuardCam uses the power of your existing smartphone cameras coupled with robust Cloud computing to provide real-time protection:
1. **Driver Monitoring:** Checks for micro-sleeps and yawning.
2. **Instant Alerts:** Triggers a severe phone-vibration and a full-screen visual alarm if the driver falls asleep.
3. **No Expensive Gear:** Runs entirely off a mounted phone streaming securely to the cloud.

---

## ⚙️ Technical Architecture

We broke this project into a highly-optimized Edge/Cloud distributed system:

1. **The Edge (React Native):** 
   A lightweight standard-issue Expo app. It securely records highly-compressed video frames via the phone's native camera API and pipes them through a high-bandwidth bidirectional WebSocket connection.
   
2. **The Cloud Brain (Python on Modal GPUs):**
   A headless WebSocket server designed to handle raw base64 frame data. It decodes the frames in real-time and runs them through a deterministic algorithm to classify body language and facial expression. 
   
4. **The Feedback Loop:**
   If the computer vision model detects five consecutive frames of "Drowsy" behavior over a given confidence threshold (0.75), a JSON alert is instantly fired back over the WebSocket to physically trigger the phone's vibration motor before the driver can get into an accident.

---

## 🚀 How to Run Locally 

If you want to test our backend architecture locally (running the AI model on your own hardware instead of the cloud):

### 1. Start the Python AI Server
Ensure you have Python 3.9+ installed.
```bash
cd stream-server
pip install -r requirements.txt
python server.py
```
*The server will start on port `8765` and wait for WebSocket connections.*

### 2. Start the Mobile App
Ensure you have the Expo Go app installed on your iPhone or Android.
```bash
cd camera-stream-app
npm install
npx expo start
```
*Scan the QR code printed in your terminal. When the app opens on your phone, type your computer's local IP address (e.g., `ws://192.168.1.5:8765`) into the app and tap **Connect**.*

---

## 🛠️ Tech Stack Highlights
* **[React Native & Expo Camera]** - For capturing hyper-fast, uncompressed image streams straight from the hardware.
* **[Python & Websockets]** - For building a custom TCP streaming bridge that avoids HTTP overhead.
* **[MediaPipe & OpenCV]** - Heavy lifting for computer vision calculations. Optionally draws complex 468-point 3D face-mesh overlays to test eye-aspect-ratio (EAR) detection limits under the hood.
