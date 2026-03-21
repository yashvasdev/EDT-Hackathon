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
import { Audio } from 'expo-av';

const FRAME_INTERVAL_MS = 200; // ~5 FPS

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
    const alarmSoundRef = useRef(null);

    // Load alarm sound on mount
    useEffect(() => {
        loadAlarmSound();
        return () => {
            stopStreaming();
            disconnectWs();
            unloadSound();
        };
    }, []);

    const loadAlarmSound = async () => {
        try {
            await Audio.setAudioModeAsync({
                playsInSilentModeIOS: true,
                staysActiveInBackground: true,
            });
            // Use a built-in system-style alert — no external file needed
            // We'll use Vibration + repeated beep as the alarm
        } catch (e) {
            console.log('Audio setup error:', e);
        }
    };

    const unloadSound = async () => {
        if (alarmSoundRef.current) {
            await alarmSoundRef.current.unloadAsync();
        }
    };

    const triggerAlarm = useCallback(async () => {
        setShowAlert(true);

        // Vibrate pattern: vibrate 500ms, pause 200ms, repeat 3 times
        Vibration.vibrate([0, 500, 200, 500, 200, 500]);

        // Auto-dismiss alert overlay after 3 seconds
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
            setStatus('Connected! Tap "Start Streaming" to begin.');
            wsRef.current = ws;
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                // Update detection state
                setIsDrowsy(data.drowsy || false);
                setConfidence(data.confidence || 0);
                setServerFps(data.fps || 0);

                // Trigger alarm if server says to alert
                if (data.alert) {
                    triggerAlarm();
                }
            } catch (e) {
                // Ignore parse errors
            }
        };

        ws.onclose = () => {
            setIsConnected(false);
            setIsStreaming(false);
            setStatus('Disconnected. Reconnect to continue.');
            stopStreaming();
        };

        ws.onerror = (error) => {
            setStatus('Connection failed. Check the URL and try again.');
            console.log('WebSocket error:', error.message);
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

    const captureAndSendFrame = useCallback(async () => {
        if (!cameraRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            return;
        }

        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.3,
                base64: true,
                skipProcessing: true,
                exif: false,
            });

            if (photo?.base64 && wsRef.current?.readyState === WebSocket.OPEN) {
                const message = JSON.stringify({
                    frame: photo.base64,
                    camera: facing,
                    timestamp: Date.now(),
                });

                wsRef.current.send(message);
                setFrameCount((prev) => prev + 1);
            }
        } catch (error) {
            console.log('Frame capture error:', error.message);
        }
    }, [facing]);

    const startStreaming = useCallback(() => {
        if (!isConnected) {
            Alert.alert('Not connected', 'Connect to the server first.');
            return;
        }

        setIsStreaming(true);
        setStatus('🔴 Streaming — monitoring for drowsiness...');
        setFrameCount(0);

        intervalRef.current = setInterval(captureAndSendFrame, FRAME_INTERVAL_MS);
    }, [isConnected, captureAndSendFrame]);

    const stopStreaming = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setIsStreaming(false);
        setIsDrowsy(false);
        if (isConnected) {
            setStatus('Streaming paused. Tap to resume.');
        }
    }, [isConnected]);

    const toggleCamera = useCallback(() => {
        setFacing((prev) => (prev === 'front' ? 'back' : 'front'));
    }, []);

    // Permission handling
    if (!permission) {
        return <View style={styles.container}><Text style={styles.statusText}>Loading...</Text></View>;
    }

    if (!permission.granted) {
        return (
            <View style={styles.container}>
                <Text style={styles.title}>🚛 GuardCam</Text>
                <Text style={styles.statusText}>Camera permission is required to monitor for drowsiness</Text>
                <TouchableOpacity style={styles.button} onPress={requestPermission}>
                    <Text style={styles.buttonText}>Grant Camera Permission</Text>
                </TouchableOpacity>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <StatusBar barStyle="light-content" />

            {/* Camera Preview */}
            <View style={styles.cameraContainer}>
                <CameraView
                    ref={cameraRef}
                    style={styles.camera}
                    facing={facing}
                >
                    {/* Overlay */}
                    <View style={styles.overlay}>
                        <View style={styles.topBar}>
                            <Text style={styles.title}>🚛 GuardCam</Text>
                            <View style={styles.topRight}>
                                {isStreaming && (
                                    <View style={[
                                        styles.detectionBadge,
                                        isDrowsy ? styles.badgeDrowsy : styles.badgeAwake
                                    ]}>
                                        <Text style={styles.badgeText}>
                                            {isDrowsy ? '😴 DROWSY' : '👁️ Awake'}
                                        </Text>
                                    </View>
                                )}
                                <View style={[styles.indicator, isStreaming ? styles.indicatorActive : styles.indicatorInactive]} />
                            </View>
                        </View>

                        <View style={styles.bottomBar}>
                            <Text style={styles.cameraLabel}>
                                {facing === 'front' ? '👤 Driver Cam' : '🛣️ Road Cam'}
                            </Text>
                            {isStreaming && (
                                <Text style={styles.frameCounter}>
                                    {confidence > 0 ? `${(confidence * 100).toFixed(0)}%` : '...'} | {serverFps} FPS
                                </Text>
                            )}
                        </View>
                    </View>

                    {/* DROWSY ALERT OVERLAY */}
                    {showAlert && (
                        <View style={styles.alertOverlay}>
                            <Text style={styles.alertIcon}>🚨</Text>
                            <Text style={styles.alertTitle}>WAKE UP!</Text>
                            <Text style={styles.alertSubtitle}>Drowsiness detected — stay alert!</Text>
                        </View>
                    )}
                </CameraView>
            </View>

            {/* Controls */}
            <View style={styles.controls}>
                <Text style={styles.statusText}>{status}</Text>

                {/* Server URL Input */}
                <View style={styles.urlRow}>
                    <TextInput
                        style={styles.input}
                        placeholder="ws://your-server.com:8765"
                        placeholderTextColor="#666"
                        value={serverUrl}
                        onChangeText={setServerUrl}
                        autoCapitalize="none"
                        autoCorrect={false}
                        editable={!isConnected}
                    />
                </View>

                {/* Connect / Disconnect */}
                <View style={styles.buttonRow}>
                    {!isConnected ? (
                        <TouchableOpacity style={styles.button} onPress={connectWs}>
                            <Text style={styles.buttonText}>Connect</Text>
                        </TouchableOpacity>
                    ) : (
                        <TouchableOpacity style={[styles.button, styles.buttonDanger]} onPress={disconnectWs}>
                            <Text style={styles.buttonText}>Disconnect</Text>
                        </TouchableOpacity>
                    )}

                    {/* Flip Camera */}
                    <TouchableOpacity style={[styles.button, styles.buttonSecondary]} onPress={toggleCamera}>
                        <Text style={styles.buttonText}>🔄 Flip</Text>
                    </TouchableOpacity>
                </View>

                {/* Stream Controls */}
                {isConnected && (
                    <View style={styles.buttonRow}>
                        {!isStreaming ? (
                            <TouchableOpacity style={[styles.button, styles.buttonStream]} onPress={startStreaming}>
                                <Text style={styles.buttonText}>▶ Start Monitoring</Text>
                            </TouchableOpacity>
                        ) : (
                            <TouchableOpacity style={[styles.button, styles.buttonDanger]} onPress={stopStreaming}>
                                <Text style={styles.buttonText}>⏹ Stop</Text>
                            </TouchableOpacity>
                        )}
                    </View>
                )}
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#0a0a0a',
        justifyContent: 'center',
    },
    cameraContainer: {
        flex: 1,
        overflow: 'hidden',
    },
    camera: {
        flex: 1,
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
        marginTop: Platform.OS === 'android' ? 30 : 50,
    },
    topRight: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 10,
    },
    bottomBar: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
    },
    title: {
        color: '#fff',
        fontSize: 22,
        fontWeight: 'bold',
        textShadowColor: 'rgba(0,0,0,0.8)',
        textShadowOffset: { width: 1, height: 1 },
        textShadowRadius: 4,
    },
    indicator: {
        width: 14,
        height: 14,
        borderRadius: 7,
        borderWidth: 2,
        borderColor: '#fff',
    },
    indicatorActive: {
        backgroundColor: '#ff3333',
    },
    indicatorInactive: {
        backgroundColor: '#666',
    },
    detectionBadge: {
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 16,
    },
    badgeDrowsy: {
        backgroundColor: 'rgba(220, 38, 38, 0.9)',
    },
    badgeAwake: {
        backgroundColor: 'rgba(22, 163, 74, 0.8)',
    },
    badgeText: {
        color: '#fff',
        fontWeight: 'bold',
        fontSize: 14,
    },
    cameraLabel: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
        backgroundColor: 'rgba(0,0,0,0.5)',
        paddingHorizontal: 12,
        paddingVertical: 4,
        borderRadius: 12,
        overflow: 'hidden',
    },
    frameCounter: {
        color: '#0f0',
        fontSize: 14,
        fontFamily: Platform.OS === 'android' ? 'monospace' : 'Courier',
        backgroundColor: 'rgba(0,0,0,0.5)',
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 8,
        overflow: 'hidden',
    },
    // Alert overlay
    alertOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(220, 38, 38, 0.85)',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 100,
    },
    alertIcon: {
        fontSize: 80,
        marginBottom: 10,
    },
    alertTitle: {
        color: '#fff',
        fontSize: 48,
        fontWeight: 'bold',
        letterSpacing: 4,
    },
    alertSubtitle: {
        color: '#fff',
        fontSize: 18,
        marginTop: 10,
        opacity: 0.9,
    },
    // Controls
    controls: {
        backgroundColor: '#111',
        padding: 16,
        paddingBottom: Platform.OS === 'android' ? 16 : 34,
    },
    statusText: {
        color: '#ccc',
        fontSize: 14,
        textAlign: 'center',
        marginBottom: 12,
    },
    urlRow: {
        marginBottom: 12,
    },
    input: {
        backgroundColor: '#222',
        color: '#fff',
        fontSize: 16,
        paddingHorizontal: 14,
        paddingVertical: 10,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#444',
    },
    buttonRow: {
        flexDirection: 'row',
        gap: 10,
        marginBottom: 8,
    },
    button: {
        flex: 1,
        backgroundColor: '#2563eb',
        paddingVertical: 12,
        borderRadius: 8,
        alignItems: 'center',
    },
    buttonSecondary: {
        backgroundColor: '#444',
        flex: 0,
        paddingHorizontal: 20,
    },
    buttonStream: {
        backgroundColor: '#16a34a',
    },
    buttonDanger: {
        backgroundColor: '#dc2626',
    },
    buttonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
    },
});
