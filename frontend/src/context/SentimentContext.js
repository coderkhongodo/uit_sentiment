import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { sentimentAPI } from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';

// Initial state
const initialState = {
    analyses: [],
    analytics: {
        total_analyses: 0,
        sentiment_distribution: {},
        average_confidence: 0,
        recent_analyses: []
    },
    loading: false,
    error: null,
    connected: false
};

// Action types
const ActionTypes = {
    SET_LOADING: 'SET_LOADING',
    SET_ERROR: 'SET_ERROR',
    ADD_ANALYSIS: 'ADD_ANALYSIS',
    SET_ANALYSES: 'SET_ANALYSES',
    SET_ANALYTICS: 'SET_ANALYTICS',
    SET_CONNECTED: 'SET_CONNECTED',
    CLEAR_ERROR: 'CLEAR_ERROR'
};

// Reducer
function sentimentReducer(state, action) {
    switch (action.type) {
        case ActionTypes.SET_LOADING:
            return { ...state, loading: action.payload };

        case ActionTypes.SET_ERROR:
            return { ...state, error: action.payload, loading: false };

        case ActionTypes.CLEAR_ERROR:
            return { ...state, error: null };

        case ActionTypes.ADD_ANALYSIS:
            return {
                ...state,
                analyses: [action.payload, ...state.analyses.slice(0, 49)], // Keep last 50
                loading: false
            };

        case ActionTypes.SET_ANALYSES:
            return { ...state, analyses: action.payload, loading: false };

        case ActionTypes.SET_ANALYTICS:
            return { ...state, analytics: action.payload };

        case ActionTypes.SET_CONNECTED:
            return { ...state, connected: action.payload };

        default:
            return state;
    }
}

// Context
const SentimentContext = createContext();

// Provider component
export function SentimentProvider({ children }) {
    const [state, dispatch] = useReducer(sentimentReducer, initialState);

    // WebSocket connection
    const { connected, sendMessage } = useWebSocket('ws://localhost:8000/ws', {
        onMessage: (data) => {
            if (data.type === 'new_analysis') {
                dispatch({ type: ActionTypes.ADD_ANALYSIS, payload: data.data });
            }
        },
        onConnect: () => {
            dispatch({ type: ActionTypes.SET_CONNECTED, payload: true });
        },
        onDisconnect: () => {
            dispatch({ type: ActionTypes.SET_CONNECTED, payload: false });
        }
    });

    // Actions
    const actions = {
        // Analyze single text
        analyzeSentiment: async (text, metadata = null) => {
            try {
                dispatch({ type: ActionTypes.SET_LOADING, payload: true });
                dispatch({ type: ActionTypes.CLEAR_ERROR });

                const result = await sentimentAPI.analyze(text, metadata);
                dispatch({ type: ActionTypes.ADD_ANALYSIS, payload: result });

                return result;
            } catch (error) {
                dispatch({ type: ActionTypes.SET_ERROR, payload: error.message });
                throw error;
            }
        },

        // Analyze batch texts
        analyzeBatch: async (texts, metadata = null) => {
            try {
                dispatch({ type: ActionTypes.SET_LOADING, payload: true });
                dispatch({ type: ActionTypes.CLEAR_ERROR });

                const result = await sentimentAPI.analyzeBatch(texts, metadata);

                // Add all results to analyses
                result.results.forEach(analysis => {
                    dispatch({ type: ActionTypes.ADD_ANALYSIS, payload: analysis });
                });

                return result;
            } catch (error) {
                dispatch({ type: ActionTypes.SET_ERROR, payload: error.message });
                throw error;
            }
        },

        // Get analytics
        fetchAnalytics: async () => {
            try {
                const analytics = await sentimentAPI.getAnalytics();
                dispatch({ type: ActionTypes.SET_ANALYTICS, payload: analytics });
                return analytics;
            } catch (error) {
                dispatch({ type: ActionTypes.SET_ERROR, payload: error.message });
                throw error;
            }
        },

        // Get analysis history
        fetchHistory: async (filters = {}) => {
            try {
                dispatch({ type: ActionTypes.SET_LOADING, payload: true });
                const history = await sentimentAPI.getHistory(filters);
                dispatch({ type: ActionTypes.SET_ANALYSES, payload: history });
                return history;
            } catch (error) {
                dispatch({ type: ActionTypes.SET_ERROR, payload: error.message });
                throw error;
            }
        },

        // Real-time analysis via WebSocket
        analyzeRealtime: (text) => {
            if (connected) {
                sendMessage({
                    type: 'analyze',
                    text: text
                });
            }
        },

        // Clear error
        clearError: () => {
            dispatch({ type: ActionTypes.CLEAR_ERROR });
        }
    };

    // Load initial data
    useEffect(() => {
        actions.fetchAnalytics();
        actions.fetchHistory({ limit: 20 });
    }, []);

    const value = {
        ...state,
        ...actions,
        connected
    };

    return (
        <SentimentContext.Provider value={value}>
            {children}
        </SentimentContext.Provider>
    );
}

// Hook to use sentiment context
export function useSentiment() {
    const context = useContext(SentimentContext);
    if (!context) {
        throw new Error('useSentiment must be used within a SentimentProvider');
    }
    return context;
}