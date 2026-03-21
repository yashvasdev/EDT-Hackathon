import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
    StyleSheet,
    Text,
    View,
    TouchableOpacity,
    StatusBar,
    Vibration,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useWebSocket } from './hooks/useWebSocket';
import {
    FRAME_INTERVAL_MS,
    DROWSINESS_WS_URL,
    COLLISION_WS_URL,
} from './constants';

export default function App() {
    const [permission, requestPermission] = useCameraPermissions();

    // --- Drowsiness state (front camera) ---
    const [isDrowsy, setIsDrowsy] = useState(false);
    const [showAlert, setShowAlert] = useState(false);
    const [drowsinessConfidence, setDrowsinessConfidence] = useState(0);
    const [drowsinessFps, setDrowsinessFps] = useState(0);

    // --- Collision state (back camera) ---
    const [collisionConfidence, setCollisionConfidence] = useState(0);
    const [collisionFps, setCollisionFps] = useState(0);

    // --- Streaming ---
    const [isStreaming, setIsStreaming] = useState(false);
    const frontCameraRef = useRef(null);
    const backCameraRef = useRef(null);
    const frontIntervalRef = useRef(null);
    const backIntervalRef = useRef(null);
    const isFrontCapturingRef = useRef(false);
    const isBackCapturingRef = useRef(false);

    const triggerAlarm = useCallback(() => {
        setShowAlert(true);
        Vibration.vibrate([0, 500, 200, 500, 200, 500]);
        setTimeout(() => setShowAlert(false), 3000);
    }, []);

    // --- WebSocket connections ---
    const handleDrowsinessMessage = useCallback(
        (data) => {
            setIsDrowsy(data.drowsy || false);
            setDrowsinessConfidence(data.confidence || 0);
            setDrowsinessFps(data.fps || 0);
            if (data.alert) {
                triggerAlarm();
            }
        },
        [triggerAlarm],
    );

    const handleCollisionMessage = useCallback((data) => {
        setCollisionConfidence(data.confidence || 0);
        setCollisionFps(data.fps || 0);
    }, []);

    const drowsinessWs = useWebSocket({
        url: DROWSINESS_WS_URL,
        onMessage: handleDrowsinessMessage,
    });

    const collisionWs = useWebSocket({
        url: COLLISION_WS_URL,
        onMessage: handleCollisionMessage,
    });

    // --- Frame capture helpers ---
    const captureAndSend = useCallback(async (cameraRef, ws, facing, isCapturingRef) => {
        if (!cameraRef.current || isCapturingRef.current) return;

        isCapturingRef.current = true;
        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.05,
                base64: true,
                skipProcessing: true,
                exif: false,
            });

            if (photo?.base64) {
                ws.send({
                    frame: photo.base64,
                    camera: facing,
                    timestamp: Date.now(),
                });
            }
        } catch (_) {
            // skip silently, e.g. if app is backgrounded
        } finally {
            isCapturingRef.current = false;
        }
    }, []);

    const startStreaming = useCallback(() => {
        if (!drowsinessWs.isConnected || !collisionWs.isConnected) return;

        setIsStreaming(true);
        isFrontCapturingRef.current = false;
        isBackCapturingRef.current = false;

        frontIntervalRef.current = setInterval(
            () => captureAndSend(frontCameraRef, drowsinessWs, 'front', isFrontCapturingRef),
            FRAME_INTERVAL_MS,
        );
        backIntervalRef.current = setInterval(
            () => captureAndSend(backCameraRef, collisionWs, 'back', isBackCapturingRef),
            FRAME_INTERVAL_MS,
        );
    }, [drowsinessWs, collisionWs, captureAndSend]);

    const stopStreaming = useCallback(() => {
        if (frontIntervalRef.current) {
            clearInterval(frontIntervalRef.current);
            frontIntervalRef.current = null;
        }
        if (backIntervalRef.current) {
            clearInterval(backIntervalRef.current);
            backIntervalRef.current = null;
        }
        setIsStreaming(false);
        isFrontCapturingRef.current = false;
        isBackCapturingRef.current = false;
    }, []);

    // --- Connect / Disconnect both ---
    const connectAll = useCallback(() => {
        drowsinessWs.connect();
        collisionWs.connect();
    }, [drowsinessWs, collisionWs]);

    const disconnectAll = useCallback(() => {
        stopStreaming();
        drowsinessWs.disconnect();
        collisionWs.disconnect();
        setIsDrowsy(false);
    }, [drowsinessWs, collisionWs, stopStreaming]);

    useEffect(() => {
        return () => {
            stopStreaming();
            drowsinessWs.disconnect();
            collisionWs.disconnect();
        };
    }, []);

    const bothConnected = drowsinessWs.isConnected && collisionWs.isConnected;
    const eitherConnected = drowsinessWs.isConnected || collisionWs.isConnected;

    // --- Permissions ---
    if (!permission) {
        return (
            <View style={s.container}>
                <Text style={s.statusText}>Loading...</Text>
            </View>
        );
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
        );
    }

    // --- Main UI (Landscape) ---
    return (
        <View style={s.container}>
            <StatusBar hidden />

            {/* Back camera - fullscreen */}
            <CameraView ref={backCameraRef} style={s.backCamera} facing="back">
                <View style={s.overlay}>
                    {/* Top bar */}
                    <View style={s.topBar}>
                        <Text style={s.title}>GuardCam</Text>
                        <View style={s.topRight}>
                            {isStreaming && (
                                <View style={[s.badge, isDrowsy ? s.badgeDrowsy : s.badgeAwake]}>
                                    <Text style={s.badgeText}>
                                        {isDrowsy ? 'DROWSY' : 'Awake'}
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
                                <>
                                    <Text style={s.fps}>
                                        Road:{' '}
                                        {collisionConfidence > 0
                                            ? `${(collisionConfidence * 100).toFixed(0)}%`
                                            : '...'}{' '}
                                        | {collisionFps} FPS
                                    </Text>
                                    <Text style={s.fps}>
                                        Driver:{' '}
                                        {drowsinessConfidence > 0
                                            ? `${(drowsinessConfidence * 100).toFixed(0)}%`
                                            : '...'}{' '}
                                        | {drowsinessFps} FPS
                                    </Text>
                                </>
                            )}
                        </View>

                        {/* Controls */}
                        <View style={s.controlsRow}>
                            <Text style={s.statusText}>
                                {bothConnected
                                    ? 'Connected'
                                    : eitherConnected
                                      ? 'Partially connected...'
                                      : drowsinessWs.status}
                            </Text>
                            {!eitherConnected ? (
                                <TouchableOpacity style={s.btn} onPress={connectAll}>
                                    <Text style={s.btnText}>Connect</Text>
                                </TouchableOpacity>
                            ) : (
                                <TouchableOpacity style={[s.btn, s.btnRed]} onPress={disconnectAll}>
                                    <Text style={s.btnText}>Disconnect</Text>
                                </TouchableOpacity>
                            )}
                            {bothConnected &&
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

                {/* ALERT OVERLAY */}
                {showAlert && (
                    <View style={s.alertOverlay}>
                        <Text style={s.alertTitle}>WAKE UP!</Text>
                        <Text style={s.alertSub}>Drowsiness detected -- stay alert!</Text>
                    </View>
                )}
            </CameraView>

            {/* Front camera - PiP bottom right */}
            <View style={s.pipContainer}>
                <CameraView ref={frontCameraRef} style={s.pipCamera} facing="front" />
                <View style={s.pipLabel}>
                    <Text style={s.pipLabelText}>Driver</Text>
                </View>
            </View>
        </View>
    );
}

const PIP_WIDTH = 160;
const PIP_HEIGHT = 120;

const s = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#0a0a0a',
        justifyContent: 'center',
    },
    backCamera: {
        ...StyleSheet.absoluteFillObject,
    },
    overlay: {
        flex: 1,
        justifyContent: 'space-between',
        padding: 16,
    },
    topBar: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    topRight: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 10,
    },
    botBar: {
        gap: 8,
    },
    statsRow: {
        flexDirection: 'row',
        justifyContent: 'flex-start',
        gap: 16,
    },
    controlsRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 10,
    },
    title: {
        color: '#fff',
        fontSize: 22,
        fontWeight: 'bold',
        textShadowColor: 'rgba(0,0,0,0.8)',
        textShadowOffset: { width: 1, height: 1 },
        textShadowRadius: 4,
    },
    dot: {
        width: 14,
        height: 14,
        borderRadius: 7,
        borderWidth: 2,
        borderColor: '#fff',
    },
    dotOn: { backgroundColor: '#f33' },
    dotOff: { backgroundColor: '#666' },
    badge: {
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 16,
    },
    badgeDrowsy: { backgroundColor: 'rgba(220,38,38,0.9)' },
    badgeAwake: { backgroundColor: 'rgba(22,163,74,0.8)' },
    badgeText: { color: '#fff', fontWeight: 'bold', fontSize: 14 },
    fps: {
        color: '#0f0',
        fontSize: 14,
        fontFamily: 'monospace',
        backgroundColor: 'rgba(0,0,0,0.5)',
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 8,
        overflow: 'hidden',
    },
    statusText: {
        color: '#ccc',
        fontSize: 14,
    },
    btn: {
        backgroundColor: '#2563eb',
        paddingVertical: 8,
        paddingHorizontal: 16,
        borderRadius: 8,
        alignItems: 'center',
    },
    btnGreen: { backgroundColor: '#16a34a' },
    btnRed: { backgroundColor: '#dc2626' },
    btnText: { color: '#fff', fontSize: 14, fontWeight: '600' },
    alertOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(220,38,38,0.85)',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 100,
    },
    alertTitle: {
        color: '#fff',
        fontSize: 48,
        fontWeight: 'bold',
        letterSpacing: 4,
    },
    alertSub: {
        color: '#fff',
        fontSize: 18,
        marginTop: 10,
        opacity: 0.9,
    },

    // --- PiP (front camera) ---
    pipContainer: {
        position: 'absolute',
        bottom: 20,
        right: 20,
        width: PIP_WIDTH,
        height: PIP_HEIGHT,
        borderRadius: 12,
        overflow: 'hidden',
        borderWidth: 2,
        borderColor: 'rgba(255,255,255,0.6)',
    },
    pipCamera: {
        width: PIP_WIDTH,
        height: PIP_HEIGHT,
    },
    pipLabel: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: 'rgba(0,0,0,0.5)',
        paddingVertical: 2,
        alignItems: 'center',
    },
    pipLabelText: {
        color: '#fff',
        fontSize: 11,
        fontWeight: '600',
    },
});
