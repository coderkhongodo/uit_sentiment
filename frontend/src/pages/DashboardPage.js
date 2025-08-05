import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    BarChart3,
    TrendingUp,
    Users,
    Clock,
    Filter,
    Download,
    RefreshCw
} from 'lucide-react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    PieChart,
    Pie,
    Cell,
    LineChart,
    Line,
    Area,
    AreaChart
} from 'recharts';
import { useSentiment } from '../context/SentimentContext';
import { formatDate, isValidDate } from '../utils/dateUtils';

const DashboardPage = () => {
    const [timeRange, setTimeRange] = useState('7d'); // 1d, 7d, 30d
    const [refreshing, setRefreshing] = useState(false);

    const {
        analytics,
        fetchAnalytics,
        fetchHistory,
        analyses
    } = useSentiment();

    // Refresh data
    const handleRefresh = async () => {
        setRefreshing(true);
        try {
            await Promise.all([
                fetchAnalytics(),
                fetchHistory({ limit: 100 })
            ]);
        } finally {
            setRefreshing(false);
        }
    };

    useEffect(() => {
        handleRefresh();
    }, [timeRange]);

    // Process data for charts
    const processChartData = () => {
        const days = timeRange === '1d' ? 1 : timeRange === '7d' ? 7 : 30;
        const dateRange = Array.from({ length: days }, (_, i) => {
            const date = new Date();
            date.setDate(date.getDate() - (days - 1 - i));
            date.setHours(0, 0, 0, 0);
            return date;
        });

        return dateRange.map(date => {
            const dayAnalyses = analyses.filter(analysis => {
                if (!isValidDate(analysis.timestamp)) return false;
                const analysisDate = new Date(analysis.timestamp);
                analysisDate.setHours(0, 0, 0, 0);
                return analysisDate.getTime() === date.getTime();
            });

            const sentimentCounts = {
                positive: dayAnalyses.filter(a => a.sentiment === 'positive').length,
                negative: dayAnalyses.filter(a => a.sentiment === 'negative').length,
                neutral: dayAnalyses.filter(a => a.sentiment === 'neutral').length,
            };

            return {
                date: formatDate(date),
                fullDate: date,
                total: dayAnalyses.length,
                ...sentimentCounts,
                avgConfidence: dayAnalyses.length > 0
                    ? dayAnalyses.reduce((sum, a) => sum + (a.confidence || 0), 0) / dayAnalyses.length
                    : 0
            };
        });
    };

    const chartData = processChartData();

    // Pie chart data
    const pieData = Object.entries(analytics.sentiment_distribution).map(([sentiment, count]) => ({
        name: sentiment === 'positive' ? 'Tích cực' :
            sentiment === 'negative' ? 'Tiêu cực' : 'Trung lập',
        value: count,
        color: sentiment === 'positive' ? '#10b981' :
            sentiment === 'negative' ? '#ef4444' : '#6b7280'
    }));

    const COLORS = ['#10b981', '#ef4444', '#6b7280'];

    // Stats cards data
    const statsCards = [
        {
            title: 'Tổng phân tích',
            value: analytics.total_analyses.toLocaleString(),
            icon: BarChart3,
            color: 'blue',
            change: '+12%'
        },
        {
            title: 'Độ tin cậy TB',
            value: `${(analytics.average_confidence * 100).toFixed(1)}%`,
            icon: TrendingUp,
            color: 'green',
            change: '+2.3%'
        },
        {
            title: 'Phân tích hôm nay',
            value: chartData[chartData.length - 1]?.total || 0,
            icon: Clock,
            color: 'purple',
            change: '+8%'
        },
        {
            title: 'Người dùng hoạt động',
            value: '24',
            icon: Users,
            color: 'orange',
            change: '+5%'
        }
    ];

    return (
        <div className="max-w-7xl mx-auto space-y-8">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col sm:flex-row sm:items-center sm:justify-between"
            >
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
                    <p className="text-gray-600 mt-1">
                        Tổng quan về hoạt động phân tích cảm xúc
                    </p>
                </div>

                <div className="flex items-center space-x-4 mt-4 sm:mt-0">
                    {/* Time Range Filter */}
                    <div className="flex bg-gray-100 rounded-lg p-1">
                        {[
                            { key: '1d', label: '1 ngày' },
                            { key: '7d', label: '7 ngày' },
                            { key: '30d', label: '30 ngày' }
                        ].map(({ key, label }) => (
                            <button
                                key={key}
                                onClick={() => setTimeRange(key)}
                                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${timeRange === key
                                    ? 'bg-white text-gray-900 shadow-sm'
                                    : 'text-gray-600 hover:text-gray-900'
                                    }`}
                            >
                                {label}
                            </button>
                        ))}
                    </div>

                    {/* Refresh Button */}
                    <button
                        onClick={handleRefresh}
                        disabled={refreshing}
                        className="btn-secondary flex items-center space-x-2"
                    >
                        <RefreshCw className={`${refreshing ? 'animate-spin' : ''}`} size={16} />
                        <span>Làm mới</span>
                    </button>
                </div>
            </motion.div>

            {/* Stats Cards */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
            >
                {statsCards.map((stat, index) => {
                    const Icon = stat.icon;
                    return (
                        <div key={stat.title} className="card">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                                    <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
                                    <p className="text-sm text-green-600 mt-1">{stat.change}</p>
                                </div>
                                <div className={`w-12 h-12 bg-${stat.color}-100 rounded-lg flex items-center justify-center`}>
                                    <Icon className={`text-${stat.color}-600`} size={24} />
                                </div>
                            </div>
                        </div>
                    );
                })}
            </motion.div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Sentiment Trend Chart */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="card"
                >
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-lg font-semibold text-gray-900">
                            Xu hướng cảm xúc
                        </h3>
                        <Filter size={20} className="text-gray-400" />
                    </div>

                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="date" />
                            <YAxis />
                            <Tooltip
                                labelFormatter={(label) => `Ngày ${label}`}
                                formatter={(value, name) => [
                                    value,
                                    name === 'positive' ? 'Tích cực' :
                                        name === 'negative' ? 'Tiêu cực' : 'Trung lập'
                                ]}
                            />
                            <Area
                                type="monotone"
                                dataKey="positive"
                                stackId="1"
                                stroke="#10b981"
                                fill="#10b981"
                                fillOpacity={0.6}
                            />
                            <Area
                                type="monotone"
                                dataKey="neutral"
                                stackId="1"
                                stroke="#6b7280"
                                fill="#6b7280"
                                fillOpacity={0.6}
                            />
                            <Area
                                type="monotone"
                                dataKey="negative"
                                stackId="1"
                                stroke="#ef4444"
                                fill="#ef4444"
                                fillOpacity={0.6}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </motion.div>

                {/* Sentiment Distribution Pie Chart */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="card"
                >
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-lg font-semibold text-gray-900">
                            Phân bố cảm xúc
                        </h3>
                        <Download size={20} className="text-gray-400 cursor-pointer hover:text-gray-600" />
                    </div>

                    <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                            <Pie
                                data={pieData}
                                cx="50%"
                                cy="50%"
                                labelLine={false}
                                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                outerRadius={80}
                                fill="#8884d8"
                                dataKey="value"
                            >
                                {pieData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip />
                        </PieChart>
                    </ResponsiveContainer>

                    {/* Legend */}
                    <div className="flex justify-center space-x-6 mt-4">
                        {pieData.map((entry, index) => (
                            <div key={entry.name} className="flex items-center space-x-2">
                                <div
                                    className="w-3 h-3 rounded-full"
                                    style={{ backgroundColor: COLORS[index] }}
                                />
                                <span className="text-sm text-gray-600">
                                    {entry.name}: {entry.value}
                                </span>
                            </div>
                        ))}
                    </div>
                </motion.div>
            </div>

            {/* Confidence Trend */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="card"
            >
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold text-gray-900">
                        Xu hướng độ tin cậy
                    </h3>
                    <div className="text-sm text-gray-500">
                        Độ tin cậy trung bình: {(analytics.average_confidence * 100).toFixed(1)}%
                    </div>
                </div>

                <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                        <Tooltip
                            labelFormatter={(label) => `Ngày ${label}`}
                            formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Độ tin cậy TB']}
                        />
                        <Line
                            type="monotone"
                            dataKey="avgConfidence"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </motion.div>
        </div>
    );
};

export default DashboardPage;