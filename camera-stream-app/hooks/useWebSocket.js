import { useState, useRef, useCallback } from 'react';

/**
 * Reusable WebSocket hook for streaming camera frames to a server.
 *
 * @param {object} options
 * @param {string} options.url - WebSocket server URL
 * @param {(data: object) => void} [options.onMessage] - Called with parsed JSON on each server message
 */
export function useWebSocket({ url, onMessage }) {
    const [isConnected, setIsConnected] = useState(false);
    const wsRef = useRef(null);
    const [status, setStatus] = useState('Disconnected');

    const connect = useCallback(() => {
        if (!url) return;

        let normalized = url.trim();
        if (!normalized.startsWith('ws://') && !normalized.startsWith('wss://')) {
            normalized = `ws://${normalized}`;
        }

        setStatus('Connecting...');
        const ws = new WebSocket(normalized);

        ws.onopen = () => {
            wsRef.current = ws;
            setIsConnected(true);
            setStatus('Connected');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage?.(data);
            } catch (_) {}
        };

        ws.onclose = () => {
            wsRef.current = null;
            setIsConnected(false);
            setStatus('Disconnected');
        };

        ws.onerror = () => {
            setStatus('Connection failed');
        };
    }, [url, onMessage]);

    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnected(false);
        setStatus('Disconnected');
    }, []);

    const send = useCallback((payload) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(typeof payload === 'string' ? payload : JSON.stringify(payload));
        }
    }, []);

    return { isConnected, status, connect, disconnect, send };
}
