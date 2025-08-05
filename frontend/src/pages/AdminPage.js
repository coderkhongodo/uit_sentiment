import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    AlertTriangle,
    CheckCircle,
    Info,
    XCircle,
    RefreshCw,
    Settings,
    Users,
    Database,
    Activity,
    TrendingDown,
    TrendingUp,
    MessageSquare
} from 'lucide-react';
import toast from 'react-hot-toast';

import { adminAPI } from '../services/api';
import { useSentiment } from '../context/SentimentContext';

const AdminPage = () => {
    const [suggestions, setSuggestions] = useState([]);
    const [loading, setLoading] = useState(false);
    const [systemStats, setSystemStats] = useState({
        uptime: '2h 34m',
        memoryUsage: '45%',
        cpuUsage: '23%',
        activeConnections: 12
    });

    const { analytics, fetchAnalytics } = useSentiment();

    // Load admin suggestions
    const loadSuggestions = async () => {
        setLoading(true);
        try {
            const response = await adminAPI.getSuggestions();
            setSuggestions(response.suggestions || []);
        } catch (error) {
            toast.error('Lỗi tải gợi ý: ' + error.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadSuggestions();
        fetchAnalytics();
    }, []);

    const getSuggestionIcon = (type) => {
        switch (type) {
            case 'warning': return <AlertTriangle className="text-yellow-500" size={24} />;
            case 'alert': return <XCircle className="text-red-500" size={24} />;
            case 'success': return <CheckCircle className="text-green-500" size={24} />;
            case 'info': return <Info className="text-blue-500" size={24} />;
            case 'error': return <XCircle className="text-red-500" size={24} />;
            default: return <Info className="text-blue-500" size={24} />;
        }
    };

    const getSuggestionBgColor = (type) => {
        switch (type) {
            case 'warning': return 'bg-yellow-50 border-yellow-200';
            case 'alert': return 'bg-red-50 border-red-200';
            case 'success': return 'bg-green-50 border-green-200';
            case 'info': return 'bg-blue-50 border-blue-200';
            case 'error': return 'bg-red-50 border-red-200';
            default: return 'bg-gray-50 border-gray-200';
        }
    };

    const handleSuggestionAction = async (suggestion) => {
        // Handle different suggestion actions
        switch (suggestion.action) {
            case 'review_low_confidence':
                toast.info('Chuyển đến trang phân tích để xem các dự đoán có độ tin cậy thấp');
                break;
            case 'review_negative_feedback':
                toast.info('Chuyển đến dashboard để xem xu hướng phản hồi tiêu cực');
                break;
            case 'maintain_quality':
                toast.success('Tiếp tục duy trì chất lượng dịch vụ tốt!');
                break;
            case 'monitor_activity':
                toast.info('Theo dõi hoạt động hệ thống');
                break;
            case 'check_system':
                toast.warning('Kiểm tra tình trạng hệ thống');
                break;
            default:
                toast.info('Hành động: ' + suggestion.action);
        }
    };

    // Calculate sentiment trends
    const calculateTrends = () => {
        const total = Object.values(analytics.sentiment_distribution).reduce((a, b) => a + b, 0);
        if (total === 0) return { positive: 0, negative: 0, neutral: 0 };

        return {
            positive: (analytics.sentiment_distribution.positive || 0) / total * 100,
            negative: (analytics.sentiment_distribution.negative || 0) / total * 100,
            neutral: (analytics.sentiment_distribution.neutral || 0) / total * 100
        };
    };

    const trends = calculateTrends();

    return (
        <div className="max-w-7xl mx-auto space-y-8">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col sm:flex-row sm:items-center sm:justify-between"
            >
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Quản trị hệ thống</h1>
                    <p className="text-gray-600 mt-1">
                        Giám sát và quản lý hệ thống phân tích cảm xúc
                    </p>
                </div>

                <button
                    onClick={loadSuggestions}
                    disabled={loading}
                    className="btn-primary flex items-center space-x-2 mt-4 sm:mt-0"
                >
                    <RefreshCw className={`${loading ? 'animate-spin' : ''}`} size={16} />
                    <span>Làm mới</span>
                </button>
            </motion.div>

            {/* System Status Cards */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
            >
                <div className="card">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Tổng phân tích</p>
                            <p className="text-2xl font-bold text-gray-900">{analytics.total_analyses}</p>
                            <div className="flex items-center mt-1">
                                <TrendingUp className="text-green-500" size={16} />
                                <span className="text-sm text-green-600 ml-1">+12%</span>
                            </div>
                        </div>
                        <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                            <Database className="text-blue-600" size={24} />
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Kết nối hoạt động</p>
                            <p className="text-2xl font-bold text-gray-900">{systemStats.activeConnections}</p>
                            <div className="flex items-center mt-1">
                                <Activity className="text-green-500" size={16} />
                                <span className="text-sm text-green-600 ml-1">Online</span>
                            </div>
                        </div>
                        <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                            <Users className="text-green-600" size={24} />
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Độ tin cậy TB</p>
                            <p className="text-2xl font-bold text-gray-900">
                                {(analytics.average_confidence * 100).toFixed(1)}%
                            </p>
                            <div className="flex items-center mt-1">
                                <TrendingUp className="text-green-500" size={16} />
                                <span className="text-sm text-green-600 ml-1">+2.3%</span>
                            </div>
                        </div>
                        <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                            <TrendingUp className="text-purple-600" size={24} />
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-sm font-medium text-gray-600">Uptime hệ thống</p>
                            <p className="text-2xl font-bold text-gray-900">{systemStats.uptime}</p>
                            <div className="flex items-center mt-1">
                                <CheckCircle className="text-green-500" size={16} />
                                <span className="text-sm text-green-600 ml-1">Ổn định</span>
                            </div>
                        </div>
                        <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
                            <Settings className="text-orange-600" size={24} />
                        </div>
                    </div>
                </div>
            </motion.div>

            {/* Sentiment Overview */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="grid grid-cols-1 lg:grid-cols-3 gap-6"
            >
                <div className="card">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-gray-900">Cảm xúc tích cực</h3>
                        <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                            <TrendingUp className="text-green-600" size={16} />
                        </div>
                    </div>
                    <div className="text-3xl font-bold text-green-600 mb-2">
                        {trends.positive.toFixed(1)}%
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                            className="bg-green-500 h-2 rounded-full"
                            style={{ width: `${trends.positive}%` }}
                        />
                    </div>
                    <p className="text-sm text-gray-600 mt-2">
                        {analytics.sentiment_distribution.positive || 0} phân tích
                    </p>
                </div>

                <div className="card">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-gray-900">Cảm xúc tiêu cực</h3>
                        <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                            <TrendingDown className="text-red-600" size={16} />
                        </div>
                    </div>
                    <div className="text-3xl font-bold text-red-600 mb-2">
                        {trends.negative.toFixed(1)}%
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                            className="bg-red-500 h-2 rounded-full"
                            style={{ width: `${trends.negative}%` }}
                        />
                    </div>
                    <p className="text-sm text-gray-600 mt-2">
                        {analytics.sentiment_distribution.negative || 0} phân tích
                    </p>
                </div>

                <div className="card">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-gray-900">Cảm xúc trung lập</h3>
                        <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
                            <MessageSquare className="text-gray-600" size={16} />
                        </div>
                    </div>
                    <div className="text-3xl font-bold text-gray-600 mb-2">
                        {trends.neutral.toFixed(1)}%
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                            className="bg-gray-500 h-2 rounded-full"
                            style={{ width: `${trends.neutral}%` }}
                        />
                    </div>
                    <p className="text-sm text-gray-600 mt-2">
                        {analytics.sentiment_distribution.neutral || 0} phân tích
                    </p>
                </div>
            </motion.div>

            {/* Admin Suggestions */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="card"
            >
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold text-gray-900">
                        Gợi ý quản trị
                    </h3>
                    <span className="text-sm text-gray-500">
                        {suggestions.length} gợi ý
                    </span>
                </div>

                {loading ? (
                    <div className="flex items-center justify-center py-8">
                        <RefreshCw className="animate-spin text-gray-400" size={24} />
                        <span className="ml-2 text-gray-500">Đang tải gợi ý...</span>
                    </div>
                ) : suggestions.length === 0 ? (
                    <div className="text-center py-8">
                        <CheckCircle className="text-green-500 mx-auto mb-4" size={48} />
                        <p className="text-gray-500">Không có gợi ý nào</p>
                        <p className="text-sm text-gray-400 mt-1">Hệ thống đang hoạt động bình thường</p>
                    </div>
                ) : (
                    <div className="space-y-4">
                        {suggestions.map((suggestion, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.1 }}
                                className={`p-4 rounded-lg border ${getSuggestionBgColor(suggestion.type)}`}
                            >
                                <div className="flex items-start space-x-3">
                                    <div className="flex-shrink-0">
                                        {getSuggestionIcon(suggestion.type)}
                                    </div>
                                    <div className="flex-1">
                                        <h4 className="font-semibold text-gray-900 mb-1">
                                            {suggestion.title}
                                        </h4>
                                        <p className="text-gray-700 mb-3">
                                            {suggestion.message}
                                        </p>
                                        <button
                                            onClick={() => handleSuggestionAction(suggestion)}
                                            className="btn-secondary text-sm"
                                        >
                                            Thực hiện
                                        </button>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                )}
            </motion.div>

            {/* System Resources */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-6"
            >
                <div className="card">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                        Tài nguyên hệ thống
                    </h3>

                    <div className="space-y-4">
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span>CPU Usage</span>
                                <span>{systemStats.cpuUsage}</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                    className="bg-blue-500 h-2 rounded-full"
                                    style={{ width: systemStats.cpuUsage }}
                                />
                            </div>
                        </div>

                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span>Memory Usage</span>
                                <span>{systemStats.memoryUsage}</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                    className="bg-green-500 h-2 rounded-full"
                                    style={{ width: systemStats.memoryUsage }}
                                />
                            </div>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                        Hoạt động gần đây
                    </h3>

                    <div className="space-y-3">
                        <div className="flex items-center space-x-3 text-sm">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span className="text-gray-600">Hệ thống khởi động thành công</span>
                            <span className="text-gray-400 ml-auto">2h trước</span>
                        </div>

                        <div className="flex items-center space-x-3 text-sm">
                            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                            <span className="text-gray-600">Model PhoBERT được tải</span>
                            <span className="text-gray-400 ml-auto">2h trước</span>
                        </div>

                        <div className="flex items-center space-x-3 text-sm">
                            <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                            <span className="text-gray-600">Database được khởi tạo</span>
                            <span className="text-gray-400 ml-auto">2h trước</span>
                        </div>

                        <div className="flex items-center space-x-3 text-sm">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span className="text-gray-600">API endpoints sẵn sàng</span>
                            <span className="text-gray-400 ml-auto">2h trước</span>
                        </div>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

export default AdminPage;