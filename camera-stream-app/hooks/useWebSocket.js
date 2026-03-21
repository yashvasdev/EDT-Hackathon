import { useState, useRef, useCallback, useEffect } from 'react';

const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 10000];
const PING_INTERVAL_MS = 15000;

/**
 * Reusable WebSocket hook for streaming camera frames to a server.
 * Automatically reconnects on unexpected disconnects with exponential backoff.
 *
 * @param {object} options
 * @param {string} options.url - WebSocket server URL
 * @param {(data: object) => void} [options.onMessage] - Called with parsed JSON on each server message
 */
export function useWebSocket({ url, onMessage }) {
    const [isConnected, setIsConnected] = useState(false);
    const [status, setStatus] = useState('Disconnected');

    const wsRef = useRef(null);
    const onMessageRef = useRef(onMessage);
    onMessageRef.current = onMessage;

    const intentionalCloseRef = useRef(false);
    const retryIndexRef = useRef(0);
    const retryTimerRef = useRef(null);
    const pingTimerRef = useRef(null);
    const urlRef = useRef(url);
    urlRef.current = url;

    const clearTimers = useCallback(() => {
        if (retryTimerRef.current) {
            clearTimeout(retryTimerRef.current);
            retryTimerRef.current = null;
        }
        if (pingTimerRef.current) {
            clearInterval(pingTimerRef.current);
            pingTimerRef.current = null;
        }
    }, []);

    const startPing = useCallback((ws) => {
        if (pingTimerRef.current) clearInterval(pingTimerRef.current);
        pingTimerRef.current = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ ping: true }));
            }
        }, PING_INTERVAL_MS);
    }, []);

    const openConnection = useCallback((targetUrl) => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        let normalized = targetUrl.trim();
        if (!normalized.startsWith('ws://') && !normalized.startsWith('wss://')) {
            normalized = `ws://${normalized}`;
        }

        setStatus('Connecting...');
        const ws = new WebSocket(normalized);

        ws.onopen = () => {
            wsRef.current = ws;
            retryIndexRef.current = 0;
            setIsConnected(true);
            setStatus('Connected');
            startPing(ws);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessageRef.current?.(data);
            } catch (_) {}
        };

        ws.onclose = () => {
            wsRef.current = null;
            setIsConnected(false);

            if (pingTimerRef.current) {
                clearInterval(pingTimerRef.current);
                pingTimerRef.current = null;
            }

            if (intentionalCloseRef.current) {
                setStatus('Disconnected');
                return;
            }

            // Unexpected close -- schedule reconnect
            const delay = RECONNECT_DELAYS[Math.min(retryIndexRef.current, RECONNECT_DELAYS.length - 1)];
            retryIndexRef.current += 1;
            setStatus(`Reconnecting in ${delay / 1000}s...`);
            retryTimerRef.current = setTimeout(() => {
                openConnection(urlRef.current);
            }, delay);
        };

        ws.onerror = () => {
            // onclose will fire after this, which handles reconnect
            setStatus('Connection error');
        };
    }, [startPing]);

    const connect = useCallback(() => {
        if (!url) return;
        intentionalCloseRef.current = false;
        retryIndexRef.current = 0;
        clearTimers();
        openConnection(url);
    }, [url, clearTimers, openConnection]);

    const disconnect = useCallback(() => {
        intentionalCloseRef.current = true;
        clearTimers();
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnected(false);
        setStatus('Disconnected');
    }, [clearTimers]);

    const send = useCallback((payload) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            if (typeof payload === 'string' || payload instanceof ArrayBuffer || ArrayBuffer.isView(payload)) {
                wsRef.current.send(payload);
            } else {
                wsRef.current.send(JSON.stringify(payload));
            }
        }
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            intentionalCloseRef.current = true;
            clearTimers();
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, [clearTimers]);

    return { isConnected, status, connect, disconnect, send };
}
