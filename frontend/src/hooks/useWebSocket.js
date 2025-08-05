import { useState, useEffect, useRef, useCallback } from 'react';

export function useWebSocket(url, options = {}) {
    const [connected, setConnected] = useState(false);
    const [error, setError] = useState(null);
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const reconnectAttempts = useRef(0);
    const maxReconnectAttempts = options.maxReconnectAttempts || 5;
    const reconnectInterval = options.reconnectInterval || 3000;

    const {
        onMessage,
        onConnect,
        onDisconnect,
        onError,
        autoReconnect = true
    } = options;

    const connect = useCallback(() => {
        try {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                return;
            }

            console.log('Connecting to WebSocket:', url);
            const ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket connected');
                setConnected(true);
                setError(null);
                reconnectAttempts.current = 0;
                onConnect?.();
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    onMessage?.(data);
                } catch (err) {
                    console.error('Error parsing WebSocket message:', err);
                }
            };

            ws.onclose = (event) => {
                console.log('WebSocket disconnected:', event.code, event.reason);
                setConnected(false);
                onDisconnect?.(event);

                // Auto-reconnect logic
                if (autoReconnect && reconnectAttempts.current < maxReconnectAttempts) {
                    reconnectAttempts.current++;
                    console.log(`Attempting to reconnect (${reconnectAttempts.current}/${maxReconnectAttempts})...`);

                    reconnectTimeoutRef.current = setTimeout(() => {
                        connect();
                    }, reconnectInterval);
                }
            };

            ws.onerror = (event) => {
                console.error('WebSocket error:', event);
                const errorMessage = 'WebSocket connection error';
                setError(errorMessage);
                onError?.(errorMessage);
            };

        } catch (err) {
            console.error('Error creating WebSocket connection:', err);
            setError(err.message);
        }
    }, [url, onMessage, onConnect, onDisconnect, onError, autoReconnect, maxReconnectAttempts, reconnectInterval]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }

        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        setConnected(false);
    }, []);

    const sendMessage = useCallback((message) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            try {
                const messageString = typeof message === 'string' ? message : JSON.stringify(message);
                wsRef.current.send(messageString);
                return true;
            } catch (err) {
                console.error('Error sending WebSocket message:', err);
                setError(err.message);
                return false;
            }
        } else {
            console.warn('WebSocket is not connected');
            return false;
        }
    }, []);

    // Connect on mount
    useEffect(() => {
        connect();

        // Cleanup on unmount
        return () => {
            disconnect();
        };
    }, [connect, disconnect]);

    // Ping to keep connection alive
    useEffect(() => {
        if (!connected) return;

        const pingInterval = setInterval(() => {
            sendMessage({ type: 'ping' });
        }, 30000); // Ping every 30 seconds

        return () => clearInterval(pingInterval);
    }, [connected, sendMessage]);

    return {
        connected,
        error,
        sendMessage,
        connect,
        disconnect
    };
}