import React, { useState, useRef, useEffect, useCallback } from "react"
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  StatusBar,
  Vibration,
  Platform,
} from "react-native"
import { CameraView, useCameraPermissions } from "expo-camera"
import { useWebSocket } from "./hooks/useWebSocket"
import {
  FRAME_INTERVAL_MS,
  DROWSINESS_WS_URL,
  COLLISION_WS_URL,
} from "./constants"
import { initAlertSound, playAlertSound } from "./utils/alertSound"

function base64ToBytes(base64) {
  const raw = atob(base64)
  const bytes = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i++) {
    bytes[i] = raw.charCodeAt(i)
  }
  return bytes
}

let BackgroundCameraModule = null
let BackgroundCameraView = null
if (Platform.OS === "android") {
  try {
    const mod = require("./modules/background-camera")
    BackgroundCameraModule = mod.BackgroundCameraModule
    BackgroundCameraView = mod.BackgroundCameraView
  } catch (_) {}
}

export default function App() {
  const [permission, requestPermission] = useCameraPermissions()

  // --- Concurrent camera support ---
  const [concurrentSupported, setConcurrentSupported] = useState(null) // null = unchecked
  const [concurrentReason, setConcurrentReason] = useState("")

  // --- Driver state (front camera) ---
  // state: "awake" | "drowsy" | "yawning" | "distracted"
  const [driverState, setDriverState] = useState("awake")
  const [alertState, setAlertState] = useState(null) // which state triggered the alert overlay
  const [drowsinessConfidence, setDrowsinessConfidence] = useState(0)
  const [drowsinessFps, setDrowsinessFps] = useState(0)

  // --- Collision state (back camera) ---
  const [collisionConfidence, setCollisionConfidence] = useState(0)
  const [collisionRisk, setCollisionRisk] = useState("")

  // --- Streaming ---
  const [isStreaming, setIsStreaming] = useState(false)
  const backCameraRef = useRef(null)
  const backIntervalRef = useRef(null)
  const isBackCapturingRef = useRef(false)

  // --- Preload alert sound on mount ---
  useEffect(() => {
    initAlertSound()
  }, [])

  // --- Check concurrent support on mount ---
  useEffect(() => {
    if (Platform.OS !== "android" || !BackgroundCameraModule) {
      setConcurrentSupported(false)
      setConcurrentReason("Background camera module only available on Android.")
      return
    }

    BackgroundCameraModule.checkConcurrentSupport().then((result) => {
      setConcurrentSupported(result.supported)
      setConcurrentReason(result.reason)
    })
  }, [])

  // --- Start front camera capture after support is confirmed and view has mounted ---
  useEffect(() => {
    if (!concurrentSupported || !BackgroundCameraModule) return

    let started = false

    BackgroundCameraModule.startCapture(FRAME_INTERVAL_MS).then((result) => {
      started = result.started
      if (!result.started) {
        console.warn("[BackgroundCamera] Failed to start:", result.reason)
      }
    })

    return () => {
      if (started) {
        BackgroundCameraModule.stopCapture()
      }
    }
  }, [concurrentSupported])

  const triggerAlarm = useCallback((state) => {
    setAlertState(state)
    Vibration.vibrate([0, 500, 200, 500, 200, 500])
    playAlertSound()
    setTimeout(() => setAlertState(null), 3000)
  }, [])

  // --- WebSocket connections ---
  const handleDrowsinessMessage = useCallback(
    (data) => {
      const state = data.state || "awake"
      setDriverState(state)
      setDrowsinessConfidence(data.confidence || 0)
      setDrowsinessFps(data.fps || 0)
      if (data.alert) {
        triggerAlarm(state)
      }
    },
    [triggerAlarm],
  )

  const handleCollisionMessage = useCallback((data) => {
    if (data.error) return
    setCollisionConfidence(data.probability || 0)
    setCollisionRisk(data.risk_level || "unknown")
  }, [])

  const drowsinessWs = useWebSocket({
    url: DROWSINESS_WS_URL,
    onMessage: handleDrowsinessMessage,
  })

  const collisionWs = useWebSocket({
    url: COLLISION_WS_URL,
    onMessage: handleCollisionMessage,
    label: "CollisionWS",
  })

  // --- Front camera frames from native module -> drowsiness WS ---
  const drowsinessWsRef = useRef(drowsinessWs)
  drowsinessWsRef.current = drowsinessWs
  const isStreamingRef = useRef(isStreaming)
  isStreamingRef.current = isStreaming

  useEffect(() => {
    if (!BackgroundCameraModule) return

    const frameSub = BackgroundCameraModule.addListener("onFrame", (event) => {
      if (isStreamingRef.current) {
        drowsinessWsRef.current.send({
          frame: event.base64,
          camera: "front",
          timestamp: event.timestamp,
        })
      }
    })

    const errorSub = BackgroundCameraModule.addListener("onError", (event) => {
      console.warn("[BackgroundCamera]", event.message)
    })

    return () => {
      frameSub.remove()
      errorSub.remove()
    }
  }, [])

  // --- Back camera frame capture (via expo-camera CameraView) ---
  const captureAndSendBack = useCallback(async () => {
    if (!backCameraRef.current || isBackCapturingRef.current) return

    isBackCapturingRef.current = true
    try {
      const photo = await backCameraRef.current.takePictureAsync({
        quality: 0.3,
        base64: true,
        exif: false,
        shutterSound: false,
      })

      if (photo?.base64 && collisionWs.isConnected) {
        collisionWs.send(base64ToBytes(photo.base64))
      }
    } catch (_) {
      // skip silently
    } finally {
      isBackCapturingRef.current = false
    }
  }, [collisionWs])

  const startStreaming = useCallback(() => {
    if (!collisionWs.isConnected && !drowsinessWs.isConnected) return

    setIsStreaming(true)
    isBackCapturingRef.current = false

    // Start back camera interval (expo-camera)
    backIntervalRef.current = setInterval(captureAndSendBack, FRAME_INTERVAL_MS)
  }, [collisionWs, drowsinessWs, captureAndSendBack])

  const stopStreaming = useCallback(() => {
    if (backIntervalRef.current) {
      clearInterval(backIntervalRef.current)
      backIntervalRef.current = null
    }
    isBackCapturingRef.current = false
    setIsStreaming(false)
  }, [])

  // --- Connect / Disconnect both ---
  const connectAll = useCallback(() => {
    drowsinessWs.connect()
    collisionWs.connect()
  }, [drowsinessWs, collisionWs])

  const disconnectAll = useCallback(() => {
    stopStreaming()
    drowsinessWs.disconnect()
    collisionWs.disconnect()
    setIsDrowsy(false)
  }, [drowsinessWs, collisionWs, stopStreaming])

  useEffect(() => {
    return () => {
      stopStreaming()
      drowsinessWs.disconnect()
      collisionWs.disconnect()
    }
  }, [])

  const bothConnected = drowsinessWs.isConnected && collisionWs.isConnected
  const eitherConnected = drowsinessWs.isConnected || collisionWs.isConnected

  // --- Permissions ---
  if (!permission) {
    return (
      <View style={s.container}>
        <Text style={s.statusText}>Loading...</Text>
      </View>
    )
  }
  if (!permission.granted) {
    return (
      <View style={s.container}>
        <Text style={s.title}>GuardCam</Text>
        <Text style={s.statusText}>Camera permission is required</Text>
        <TouchableOpacity style={s.btn} onPress={requestPermission}>
          <Text style={s.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    )
  }

  // --- Main UI (Landscape) ---
  return (
    <View style={s.container}>
      <StatusBar hidden />

      {/* Back camera - fullscreen (expo-camera) */}
      <CameraView
        ref={backCameraRef}
        style={s.backCamera}
        facing="back"
        animateShutter={false}
        shutterSound={false}
        pictureSize="640x480"
      >
        <View style={s.overlay}>
          {/* Top bar */}
          <View style={s.topBar}>
            <Text style={s.title}>GuardCam</Text>
            <View style={s.topRight}>
              {isStreaming && (
                <View
                  style={[s.badge, s[`badge_${driverState}`] || s.badge_awake]}
                >
                  <Text style={s.badgeText}>
                    {driverState.toUpperCase()}
                  </Text>
                </View>
              )}
              <View style={[s.dot, isStreaming ? s.dotOn : s.dotOff]} />
            </View>
          </View>

          {/* Bottom bar */}
          <View style={s.botBar}>
            <View style={s.statsRow}>
              {isStreaming && (
                <Text style={s.fps}>
                  Driver:{" "}
                  {drowsinessConfidence > 0
                    ? `${(drowsinessConfidence * 100).toFixed(0)}%`
                    : "..."}{" "}
                  | {drowsinessFps} FPS
                </Text>
              )}
            </View>

            {/* Concurrent camera status */}
            {concurrentSupported === false && (
              <Text style={s.warningText}>
                Front camera unavailable: {concurrentReason}
              </Text>
            )}

            {/* Controls */}
            <View style={s.controlsRow}>
              <Text style={s.statusText}>
                {bothConnected
                  ? "Connected"
                  : eitherConnected
                    ? "Partially connected..."
                    : drowsinessWs.status}
              </Text>
              {!eitherConnected ? (
                <TouchableOpacity style={s.btn} onPress={connectAll}>
                  <Text style={s.btnText}>Connect</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity
                  style={[s.btn, s.btnRed]}
                  onPress={disconnectAll}
                >
                  <Text style={s.btnText}>Disconnect</Text>
                </TouchableOpacity>
              )}
              {eitherConnected &&
                (!isStreaming ? (
                  <TouchableOpacity
                    style={[s.btn, s.btnGreen]}
                    onPress={startStreaming}
                  >
                    <Text style={s.btnText}>Start</Text>
                  </TouchableOpacity>
                ) : (
                  <TouchableOpacity
                    style={[s.btn, s.btnRed]}
                    onPress={stopStreaming}
                  >
                    <Text style={s.btnText}>Stop</Text>
                  </TouchableOpacity>
                ))}
            </View>
          </View>
        </View>

        {/* Collision score */}
        {isStreaming && (
          <Text
            style={[
              s.collisionText,
              collisionRisk === "high"
                ? s.collisionTextHigh
                : collisionRisk === "medium"
                  ? s.collisionTextMedium
                  : s.collisionTextLow,
            ]}
          >
            Collision {(collisionConfidence * 100).toFixed(0)}%
          </Text>
        )}

        {/* ALERT OVERLAY */}
        {alertState && (
          <View
            style={[
              s.alertOverlay,
              s[`alertOverlay_${alertState}`] || s.alertOverlay_drowsy,
            ]}
          >
            <Text style={s.alertTitle}>
              {alertState === "drowsy" && "WAKE UP!"}
              {alertState === "yawning" && "STAY ALERT!"}
              {alertState === "distracted" && "EYES ON ROAD!"}
            </Text>
            <Text style={s.alertSub}>
              {alertState === "drowsy" && "Drowsiness detected -- pull over if needed"}
              {alertState === "yawning" && "Yawning detected -- take a break soon"}
              {alertState === "distracted" && "Distraction detected -- focus on driving"}
            </Text>
          </View>
        )}
      </CameraView>

      {/* Front camera PiP - native TextureView for smooth live preview */}
      {BackgroundCameraView && concurrentSupported && (
        <View style={s.pipContainer}>
          <BackgroundCameraView style={s.pipPreview} />
          <View style={s.pipLabel}>
            <Text style={s.pipLabelText}>Driver</Text>
          </View>
        </View>
      )}
    </View>
  )
}

const s = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0a0a0a",
    justifyContent: "center",
  },
  backCamera: {
    ...StyleSheet.absoluteFillObject,
  },
  overlay: {
    flex: 1,
    justifyContent: "space-between",
    padding: 16,
  },
  topBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  topRight: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  botBar: {
    gap: 8,
  },
  statsRow: {
    flexDirection: "row",
    justifyContent: "flex-start",
    gap: 16,
  },
  controlsRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  title: {
    color: "#fff",
    fontSize: 22,
    fontWeight: "bold",
    textShadowColor: "rgba(0,0,0,0.8)",
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 4,
  },
  dot: {
    width: 14,
    height: 14,
    borderRadius: 7,
    borderWidth: 2,
    borderColor: "#fff",
  },
  dotOn: { backgroundColor: "#f33" },
  dotOff: { backgroundColor: "#666" },
  badge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  badge_drowsy: { backgroundColor: "rgba(220,38,38,0.9)" },
  badge_yawning: { backgroundColor: "rgba(234,179,8,0.9)" },
  badge_distracted: { backgroundColor: "rgba(249,115,22,0.9)" },
  badge_awake: { backgroundColor: "rgba(22,163,74,0.8)" },
  badgeText: { color: "#fff", fontWeight: "bold", fontSize: 14 },
  collisionText: {
    position: "absolute",
    top: 60,
    left: 16,
    fontSize: 28,
    fontWeight: "bold",
    textShadowColor: "rgba(0,0,0,0.8)",
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 4,
  },
  collisionTextHigh: { color: "#ff4444" },
  collisionTextMedium: { color: "#facc15" },
  collisionTextLow: { color: "#4ade80" },
  fps: {
    color: "#0f0",
    fontSize: 14,
    fontFamily: "monospace",
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
    overflow: "hidden",
  },
  warningText: {
    color: "#f90",
    fontSize: 12,
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
    overflow: "hidden",
  },
  statusText: {
    color: "#ccc",
    fontSize: 14,
  },
  btn: {
    backgroundColor: "#2563eb",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: "center",
  },
  btnGreen: { backgroundColor: "#16a34a" },
  btnRed: { backgroundColor: "#dc2626" },
  btnText: { color: "#fff", fontSize: 14, fontWeight: "600" },
  alertOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: "center",
    alignItems: "center",
    zIndex: 100,
  },
  alertOverlay_drowsy: { backgroundColor: "rgba(220,38,38,0.85)" },
  alertOverlay_yawning: { backgroundColor: "rgba(234,179,8,0.85)" },
  alertOverlay_distracted: { backgroundColor: "rgba(249,115,22,0.85)" },
  alertTitle: {
    color: "#fff",
    fontSize: 48,
    fontWeight: "bold",
    letterSpacing: 4,
  },
  alertSub: {
    color: "#fff",
    fontSize: 18,
    marginTop: 10,
    opacity: 0.9,
  },

  // --- PiP (front camera snapshot) ---
  pipContainer: {
    position: "absolute",
    bottom: 20,
    right: 20,
    width: 160,
    height: 120,
    borderRadius: 12,
    overflow: "hidden",
    borderWidth: 2,
    borderColor: "rgba(255,255,255,0.6)",
  },
  pipPreview: {
    width: 160,
    height: 120,
  },
  pipLabel: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingVertical: 2,
    alignItems: "center",
  },
  pipLabelText: {
    color: "#fff",
    fontSize: 11,
    fontWeight: "600",
  },
})
