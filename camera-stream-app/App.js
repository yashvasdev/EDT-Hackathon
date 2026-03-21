import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
    StyleSheet,
    Text,
    View,
    TextInput,
    TouchableOpacity,
    StatusBar,
    Alert,
    Platform,
    Vibration,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

const FRAME_INTERVAL_MS = 500; // 2 FPS to prevent overloading phone memory/network

export default function App() {
    const [permission, requestPermission] = useCameraPermissions();
    const [serverUrl, setServerUrl] = useState('');
    const [isConnected, setIsConnected] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [status, setStatus] = useState('Enter server URL and connect');
    const [frameCount, setFrameCount] = useState(0);
    const [facing, setFacing] = useState('front');
    const [isDrowsy, setIsDrowsy] = useState(false);
    const [showAlert, setShowAlert] = useState(false);
    const [confidence, setConfidence] = useState(0);
    const [serverFps, setServerFps] = useState(0);

    const cameraRef = useRef(null);
    const wsRef = useRef(null);
    const intervalRef = useRef(null);

    useEffect(() => {
        return () => {
            stopStreaming();
            disconnectWs();
        };
    }, []);

    const triggerAlarm = useCallback(() => {
        setShowAlert(true);
        Vibration.vibrate([0, 500, 200, 500, 200, 500]);
        setTimeout(() => setShowAlert(false), 3000);
    }, []);

    const connectWs = useCallback(() => {
        if (!serverUrl.trim()) {
            Alert.alert('Error', 'Please enter the server WebSocket URL\n(e.g. ws://your-server.com:8765)');
            return;
        }

        let url = serverUrl.trim();
        if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
            url = `ws://${url}`;
        }

        setStatus('Connecting...');
        const ws = new WebSocket(url);

        ws.onopen = () => {
            setIsConnected(true);
            setStatus('Connected! Tap "Start Monitoring" to begin.');
            wsRef.current = ws;
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setIsDrowsy(data.drowsy || false);
                setConfidence(data.confidence || 0);
                setServerFps(data.fps || 0);
                if (data.alert) {
                    triggerAlarm();
                }
            } catch (e) { }
        };

        ws.onclose = () => {
            setIsConnected(false);
            setIsStreaming(false);
            setStatus('Disconnected.');
            stopStreaming();
        };

        ws.onerror = () => {
            setStatus('Connection failed. Check URL and try again.');
        };
    }, [serverUrl, triggerAlarm]);

    const disconnectWs = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnected(false);
        setIsStreaming(false);
        setIsDrowsy(false);
        stopStreaming();
    }, []);

    const isCapturingRef = useRef(false);

    const captureAndSendFrame = useCallback(async () => {
        if (!cameraRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            return;
        }

        // Prevent concurrent captures if the previous one is still taking time
        if (isCapturingRef.current) return;

        isCapturingRef.current = true;

        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.05, // Extremely low quality for smallest base64 size (YOLO size is 640x640 anyway)
                base64: true,
                skipProcessing: true, // Crucial for speed
                exif: false,
            });

            if (photo?.base64 && wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({
                    frame: photo.base64,
                    camera: facing,
                    timestamp: Date.now(),
                }));
            }
        } catch (error) {
            // skip silently, for example if app is backgrounded
        } finally {
            isCapturingRef.current = false;
        }
    }, [facing]);

    const startStreaming = useCallback(() => {
        if (!isConnected) return;
        setIsStreaming(true);
        setStatus('🔴 Monitoring for drowsiness...');
        setFrameCount(0);
        isCapturingRef.current = false;
        intervalRef.current = setInterval(captureAndSendFrame, FRAME_INTERVAL_MS);
    }, [isConnected, captureAndSendFrame]);

    const stopStreaming = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setIsStreaming(false);
        isCapturingRef.current = false;
    }, []);

    const toggleCamera = useCallback(() => {
        setFacing((prev) => (prev === 'front' ? 'back' : 'front'));
    }, []);

    // --- Permissions ---
    if (!permission) {
        return <View style={s.container}><Text style={s.statusText}>Loading...</Text></View>;
    }
    if (!permission.granted) {
        return (
            <View style={s.container}>
                <Text style={s.title}>🚛 GuardCam</Text>
                <Text style={s.statusText}>Camera permission is required</Text>
                <TouchableOpacity style={s.btn} onPress={requestPermission}>
                    <Text style={s.btnText}>Grant Permission</Text>
                </TouchableOpacity>
            </View>
        );
    }

    // --- Main UI ---
    return (
        <View style={s.container}>
            <StatusBar barStyle="light-content" />

            <View style={s.camWrap}>
                <CameraView ref={cameraRef} style={s.cam} facing={facing}>
                    <View style={s.overlay}>
                        {/* Top bar */}
                        <View style={s.topBar}>
                            <Text style={s.title}>🚛 GuardCam</Text>
                            <View style={s.topRight}>
                                {isStreaming && (
                                    <View style={[s.badge, isDrowsy ? s.badgeDrowsy : s.badgeAwake]}>
                                        <Text style={s.badgeText}>{isDrowsy ? '😴 DROWSY' : '👁️ Awake'}</Text>
                                    </View>
                                )}
                                <View style={[s.dot, isStreaming ? s.dotOn : s.dotOff]} />
                            </View>
                        </View>
                        {/* Bottom bar */}
                        <View style={s.botBar}>
                            <Text style={s.camLabel}>{facing === 'front' ? '👤 Driver' : '🛣️ Road'}</Text>
                            {isStreaming && (
                                <Text style={s.fps}>
                                    {confidence > 0 ? `${(confidence * 100).toFixed(0)}%` : '...'} | {serverFps} FPS
                                </Text>
                            )}
                        </View>
                    </View>

                    {/* ALERT OVERLAY */}
                    {showAlert && (
                        <View style={s.alertOverlay}>
                            <Text style={s.alertIcon}>🚨</Text>
                            <Text style={s.alertTitle}>WAKE UP!</Text>
                            <Text style={s.alertSub}>Drowsiness detected — stay alert!</Text>
                        </View>
                    )}
                </CameraView>
            </View>

            {/* Controls */}
            <View style={s.controls}>
                <Text style={s.statusText}>{status}</Text>

                <TextInput
                    style={s.input}
                    placeholder="ws://your-server.com:8765"
                    placeholderTextColor="#666"
                    value={serverUrl}
                    onChangeText={setServerUrl}
                    autoCapitalize="none"
                    autoCorrect={false}
                    editable={!isConnected}
                />

                <View style={s.row}>
                    {!isConnected ? (
                        <TouchableOpacity style={s.btn} onPress={connectWs}>
                            <Text style={s.btnText}>Connect</Text>
                        </TouchableOpacity>
                    ) : (
                        <TouchableOpacity style={[s.btn, s.btnRed]} onPress={disconnectWs}>
                            <Text style={s.btnText}>Disconnect</Text>
                        </TouchableOpacity>
                    )}
                    <TouchableOpacity style={[s.btn, s.btnGray]} onPress={toggleCamera}>
                        <Text style={s.btnText}>🔄 Flip</Text>
                    </TouchableOpacity>
                </View>

                {isConnected && (
                    <View style={s.row}>
                        {!isStreaming ? (
                            <TouchableOpacity style={[s.btn, s.btnGreen]} onPress={startStreaming}>
                                <Text style={s.btnText}>▶ Start Monitoring</Text>
                            </TouchableOpacity>
                        ) : (
                            <TouchableOpacity style={[s.btn, s.btnRed]} onPress={stopStreaming}>
                                <Text style={s.btnText}>⏹ Stop</Text>
                            </TouchableOpacity>
                        )}
                    </View>
                )}
            </View>
        </View>
    );
}

const s = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#0a0a0a', justifyContent: 'center' },
    camWrap: { flex: 1, overflow: 'hidden' },
    cam: { flex: 1 },
    overlay: { flex: 1, justifyContent: 'space-between', padding: 16 },
    topBar: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 30 },
    topRight: { flexDirection: 'row', alignItems: 'center', gap: 10 },
    botBar: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
    title: { color: '#fff', fontSize: 22, fontWeight: 'bold', textShadowColor: 'rgba(0,0,0,0.8)', textShadowOffset: { width: 1, height: 1 }, textShadowRadius: 4 },
    dot: { width: 14, height: 14, borderRadius: 7, borderWidth: 2, borderColor: '#fff' },
    dotOn: { backgroundColor: '#f33' },
    dotOff: { backgroundColor: '#666' },
    badge: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 16 },
    badgeDrowsy: { backgroundColor: 'rgba(220,38,38,0.9)' },
    badgeAwake: { backgroundColor: 'rgba(22,163,74,0.8)' },
    badgeText: { color: '#fff', fontWeight: 'bold', fontSize: 14 },
    camLabel: { color: '#fff', fontSize: 16, fontWeight: '600', backgroundColor: 'rgba(0,0,0,0.5)', paddingHorizontal: 12, paddingVertical: 4, borderRadius: 12, overflow: 'hidden' },
    fps: { color: '#0f0', fontSize: 14, fontFamily: 'monospace', backgroundColor: 'rgba(0,0,0,0.5)', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8, overflow: 'hidden' },
    alertOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(220,38,38,0.85)', justifyContent: 'center', alignItems: 'center', zIndex: 100 },
    alertIcon: { fontSize: 80, marginBottom: 10 },
    alertTitle: { color: '#fff', fontSize: 48, fontWeight: 'bold', letterSpacing: 4 },
    alertSub: { color: '#fff', fontSize: 18, marginTop: 10, opacity: 0.9 },
    controls: { backgroundColor: '#111', padding: 16, paddingBottom: 16 },
    statusText: { color: '#ccc', fontSize: 14, textAlign: 'center', marginBottom: 12 },
    input: { backgroundColor: '#222', color: '#fff', fontSize: 16, paddingHorizontal: 14, paddingVertical: 10, borderRadius: 8, borderWidth: 1, borderColor: '#444', marginBottom: 12 },
    row: { flexDirection: 'row', gap: 10, marginBottom: 8 },
    btn: { flex: 1, backgroundColor: '#2563eb', paddingVertical: 12, borderRadius: 8, alignItems: 'center' },
    btnGray: { backgroundColor: '#444', flex: 0, paddingHorizontal: 20 },
    btnGreen: { backgroundColor: '#16a34a' },
    btnRed: { backgroundColor: '#dc2626' },
    btnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
