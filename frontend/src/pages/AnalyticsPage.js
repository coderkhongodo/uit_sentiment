import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Search,
    Filter,
    Download,
    Eye,
    Calendar,
    TrendingUp,
    BarChart3
} from 'lucide-react';
import { useSentiment } from '../context/SentimentContext';
import SentimentResult from '../components/SentimentResult';
import { formatTimestamp } from '../utils/dateUtils';

const AnalyticsPage = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [sentimentFilter, setSentimentFilter] = useState('all');
    const [dateFilter, setDateFilter] = useState('all');
    const [currentPage, setCurrentPage] = useState(1);
    const [selectedAnalysis, setSelectedAnalysis] = useState(null);

    const {
        analyses,
        fetchHistory,
        loading,
        analytics
    } = useSentiment();

    const itemsPerPage = 10;

    // Filter analyses
    const filteredAnalyses = analyses.filter(analysis => {
        const matchesSearch = analysis.text.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesSentiment = sentimentFilter === 'all' || analysis.sentiment === sentimentFilter;

        let matchesDate = true;
        if (dateFilter !== 'all') {
            const analysisDate = new Date(analysis.timestamp);
            const now = new Date();

            switch (dateFilter) {
                case 'today':
                    matchesDate = analysisDate.toDateString() === now.toDateString();
                    break;
                case 'week':
                    const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                    matchesDate = analysisDate >= weekAgo;
                    break;
                case 'month':
                    const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                    matchesDate = analysisDate >= monthAgo;
                    break;
                default:
                    matchesDate = true;
            }
        }

        return matchesSearch && matchesSentiment && matchesDate;
    });

    // Pagination
    const totalPages = Math.ceil(filteredAnalyses.length / itemsPerPage);
    const startIndex = (currentPage - 1) * itemsPerPage;
    const paginatedAnalyses = filteredAnalyses.slice(startIndex, startIndex + itemsPerPage);

    // Load more data when needed
    useEffect(() => {
        if (analyses.length < 100) {
            fetchHistory({ limit: 100 });
        }
    }, []);

    const getSentimentBadge = (sentiment) => {
        const colors = {
            positive: 'bg-green-100 text-green-800',
            negative: 'bg-red-100 text-red-800',
            neutral: 'bg-gray-100 text-gray-800'
        };

        const labels = {
            positive: 'Tích cực',
            negative: 'Tiêu cực',
            neutral: 'Trung lập'
        };

        return (
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[sentiment]}`}>
                {labels[sentiment]}
            </span>
        );
    };

    const getConfidenceBadge = (confidence) => {
        let color = 'bg-gray-100 text-gray-800';
        if (confidence >= 0.8) color = 'bg-green-100 text-green-800';
        else if (confidence >= 0.6) color = 'bg-yellow-100 text-yellow-800';
        else color = 'bg-red-100 text-red-800';

        return (
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${color}`}>
                {(confidence * 100).toFixed(1)}%
            </span>
        );
    };

    const exportData = () => {
        const csvContent = [
            ['Thời gian', 'Văn bản', 'Cảm xúc', 'Độ tin cậy', 'Tích cực', 'Tiêu cực', 'Trung lập'],
            ...filteredAnalyses.map(analysis => [
                formatTimestamp(analysis.timestamp),
                `"${analysis.text.replace(/"/g, '""')}"`,
                analysis.sentiment,
                ((analysis.confidence || 0) * 100).toFixed(1) + '%',
                ((analysis.probabilities?.positive || 0) * 100).toFixed(1) + '%',
                ((analysis.probabilities?.negative || 0) * 100).toFixed(1) + '%',
                ((analysis.probabilities?.neutral || 0) * 100).toFixed(1) + '%'
            ])
        ].map(row => row.join(',')).join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `sentiment_analysis_${new Date().toISOString().split('T')[0]}.csv`;
        link.click();
    };

    return (
        <div className="max-w-7xl mx-auto space-y-8">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col sm:flex-row sm:items-center sm:justify-between"
            >
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Phân tích chi tiết</h1>
                    <p className="text-gray-600 mt-1">
                        Xem và phân tích lịch sử các kết quả sentiment analysis
                    </p>
                </div>

                <button
                    onClick={exportData}
                    className="btn-primary flex items-center space-x-2 mt-4 sm:mt-0"
                >
                    <Download size={16} />
                    <span>Xuất dữ liệu</span>
                </button>
            </motion.div>

            {/* Summary Stats */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="grid grid-cols-1 md:grid-cols-4 gap-6"
            >
                <div className="card text-center">
                    <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                        <BarChart3 className="text-blue-600" size={24} />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900">{filteredAnalyses.length}</h3>
                    <p className="text-gray-600">Kết quả tìm thấy</p>
                </div>

                <div className="card text-center">
                    <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                        <TrendingUp className="text-green-600" size={24} />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900">
                        {filteredAnalyses.length > 0
                            ? (filteredAnalyses.reduce((sum, a) => sum + a.confidence, 0) / filteredAnalyses.length * 100).toFixed(1)
                            : 0}%
                    </h3>
                    <p className="text-gray-600">Độ tin cậy TB</p>
                </div>

                <div className="card text-center">
                    <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                        <div className="w-6 h-6 bg-green-500 rounded-full"></div>
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900">
                        {filteredAnalyses.filter(a => a.sentiment === 'positive').length}
                    </h3>
                    <p className="text-gray-600">Tích cực</p>
                </div>

                <div className="card text-center">
                    <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                        <div className="w-6 h-6 bg-red-500 rounded-full"></div>
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900">
                        {filteredAnalyses.filter(a => a.sentiment === 'negative').length}
                    </h3>
                    <p className="text-gray-600">Tiêu cực</p>
                </div>
            </motion.div>

            {/* Filters */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="card"
            >
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {/* Search */}
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                        <input
                            type="text"
                            placeholder="Tìm kiếm văn bản..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="input-field pl-10"
                        />
                    </div>

                    {/* Sentiment Filter */}
                    <select
                        value={sentimentFilter}
                        onChange={(e) => setSentimentFilter(e.target.value)}
                        className="input-field"
                    >
                        <option value="all">Tất cả cảm xúc</option>
                        <option value="positive">Tích cực</option>
                        <option value="negative">Tiêu cực</option>
                        <option value="neutral">Trung lập</option>
                    </select>

                    {/* Date Filter */}
                    <select
                        value={dateFilter}
                        onChange={(e) => setDateFilter(e.target.value)}
                        className="input-field"
                    >
                        <option value="all">Tất cả thời gian</option>
                        <option value="today">Hôm nay</option>
                        <option value="week">7 ngày qua</option>
                        <option value="month">30 ngày qua</option>
                    </select>
                </div>
            </motion.div>

            {/* Results Table */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="card overflow-hidden"
            >
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                    Thời gian
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                    Văn bản
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                    Cảm xúc
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                    Độ tin cậy
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                    Hành động
                                </th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {paginatedAnalyses.map((analysis) => (
                                <tr key={analysis.analysis_id} className="hover:bg-gray-50">
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                        {formatTimestamp(analysis.timestamp)}
                                    </td>
                                    <td className="px-6 py-4 text-sm text-gray-900 max-w-xs">
                                        <div className="truncate" title={analysis.text}>
                                            {analysis.text.length > 100
                                                ? analysis.text.substring(0, 100) + '...'
                                                : analysis.text
                                            }
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        {getSentimentBadge(analysis.sentiment)}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        {getConfidenceBadge(analysis.confidence)}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                        <button
                                            onClick={() => setSelectedAnalysis(analysis)}
                                            className="text-primary-600 hover:text-primary-900 flex items-center space-x-1"
                                        >
                                            <Eye size={16} />
                                            <span>Xem</span>
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                    <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
                        <div className="flex-1 flex justify-between sm:hidden">
                            <button
                                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                                disabled={currentPage === 1}
                                className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
                            >
                                Trước
                            </button>
                            <button
                                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                                disabled={currentPage === totalPages}
                                className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
                            >
                                Sau
                            </button>
                        </div>
                        <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
                            <div>
                                <p className="text-sm text-gray-700">
                                    Hiển thị <span className="font-medium">{startIndex + 1}</span> đến{' '}
                                    <span className="font-medium">
                                        {Math.min(startIndex + itemsPerPage, filteredAnalyses.length)}
                                    </span>{' '}
                                    trong <span className="font-medium">{filteredAnalyses.length}</span> kết quả
                                </p>
                            </div>
                            <div>
                                <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px">
                                    {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                                        <button
                                            key={page}
                                            onClick={() => setCurrentPage(page)}
                                            className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${page === currentPage
                                                ? 'z-10 bg-primary-50 border-primary-500 text-primary-600'
                                                : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                                                }`}
                                        >
                                            {page}
                                        </button>
                                    ))}
                                </nav>
                            </div>
                        </div>
                    </div>
                )}
            </motion.div>

            {/* Analysis Detail Modal */}
            {selectedAnalysis && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
                    <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                        <div className="p-6">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-semibold">Chi tiết phân tích</h3>
                                <button
                                    onClick={() => setSelectedAnalysis(null)}
                                    className="text-gray-400 hover:text-gray-600"
                                >
                                    ✕
                                </button>
                            </div>
                            <SentimentResult analysis={selectedAnalysis} />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AnalyticsPage;