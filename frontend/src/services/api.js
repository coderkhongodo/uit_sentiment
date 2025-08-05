import axios from 'axios';

// API base configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor
api.interceptors.request.use(
    (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Response interceptor
api.interceptors.response.use(
    (response) => {
        return response.data;
    },
    (error) => {
        const message = error.response?.data?.detail || error.message || 'An error occurred';
        return Promise.reject(new Error(message));
    }
);

// Sentiment Analysis API
export const sentimentAPI = {
    // Health check
    healthCheck: () => api.get('/health'),

    // Single text analysis
    analyze: (text, metadata = null) =>
        api.post('/analyze', {
            text,
            metadata,
            user_id: localStorage.getItem('user_id') || null
        }),

    // Batch text analysis
    analyzeBatch: (texts, metadata = null) =>
        api.post('/analyze/batch', {
            texts,
            metadata,
            user_id: localStorage.getItem('user_id') || null
        }),

    // Get analytics
    getAnalytics: () => api.get('/analytics'),

    // Get analysis history
    getHistory: (filters = {}) => {
        const params = new URLSearchParams();

        if (filters.limit) params.append('limit', filters.limit);
        if (filters.offset) params.append('offset', filters.offset);
        if (filters.sentiment) params.append('sentiment', filters.sentiment);
        if (filters.user_id) params.append('user_id', filters.user_id);

        return api.get(`/history?${params.toString()}`);
    },

    // Get admin suggestions
    getAdminSuggestions: () => api.get('/admin/suggestions'),
};

// Admin API
export const adminAPI = {
    // Get suggestions
    getSuggestions: () => api.get('/admin/suggestions'),

    // Save feedback
    saveFeedback: (analysisId, feedbackType, feedbackText) =>
        api.post('/admin/feedback', {
            analysis_id: analysisId,
            feedback_type: feedbackType,
            feedback_text: feedbackText
        }),
};

// Export default api instance
export default api;