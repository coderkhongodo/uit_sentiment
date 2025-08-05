import React from 'react';
import { motion } from 'framer-motion';
import { Smile, Frown, Meh, Clock } from 'lucide-react';
import { formatTime } from '../utils/dateUtils';

const RecentAnalyses = ({ analyses = [] }) => {
    const getSentimentIcon = (sentiment) => {
        switch (sentiment) {
            case 'positive': return <Smile className="text-green-500" size={16} />;
            case 'negative': return <Frown className="text-red-500" size={16} />;
            case 'neutral': return <Meh className="text-gray-500" size={16} />;
            default: return null;
        }
    };

    const getSentimentLabel = (sentiment) => {
        switch (sentiment) {
            case 'positive': return 'Tích cực';
            case 'negative': return 'Tiêu cực';
            case 'neutral': return 'Trung lập';
            default: return 'Không xác định';
        }
    };

    const getSentimentBadgeColor = (sentiment) => {
        switch (sentiment) {
            case 'positive': return 'bg-green-100 text-green-800';
            case 'negative': return 'bg-red-100 text-red-800';
            case 'neutral': return 'bg-gray-100 text-gray-800';
            default: return 'bg-gray-100 text-gray-800';
        }
    };

    const truncateText = (text, maxLength = 60) => {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    };

    if (analyses.length === 0) {
        return (
            <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Phân tích gần đây
                </h3>
                <div className="text-center py-8">
                    <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <Clock className="text-gray-400" size={24} />
                    </div>
                    <p className="text-gray-500">Chưa có phân tích nào</p>
                    <p className="text-sm text-gray-400 mt-1">
                        Nhập văn bản để bắt đầu phân tích cảm xúc
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Phân tích gần đây
            </h3>

            <div className="space-y-3">
                {analyses.map((analysis, index) => (
                    <motion.div
                        key={analysis.analysis_id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 transition-colors duration-200"
                    >
                        <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center space-x-2">
                                {getSentimentIcon(analysis.sentiment)}
                                <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getSentimentBadgeColor(analysis.sentiment)}`}>
                                    {getSentimentLabel(analysis.sentiment)}
                                </span>
                            </div>

                            <div className="flex items-center space-x-1 text-gray-500 text-xs">
                                <Clock size={12} />
                                <span>
                                    {formatTime(analysis.timestamp)}
                                </span>
                            </div>
                        </div>

                        <p className="text-sm text-gray-700 mb-2">
                            "{truncateText(analysis.text)}"
                        </p>

                        <div className="flex items-center justify-between">
                            <div className="text-xs text-gray-500">
                                Độ tin cậy: {(analysis.confidence * 100).toFixed(1)}%
                            </div>

                            {/* Mini probability bars */}
                            <div className="flex items-center space-x-1">
                                {Object.entries(analysis.probabilities).map(([sentiment, probability]) => (
                                    <div
                                        key={sentiment}
                                        className="w-8 h-1 bg-gray-200 rounded-full overflow-hidden"
                                        title={`${getSentimentLabel(sentiment)}: ${(probability * 100).toFixed(1)}%`}
                                    >
                                        <div
                                            className={`h-full ${sentiment === 'positive' ? 'bg-green-500' :
                                                sentiment === 'negative' ? 'bg-red-500' : 'bg-gray-500'
                                                }`}
                                            style={{ width: `${probability * 100}%` }}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>

            {analyses.length >= 5 && (
                <div className="mt-4 text-center">
                    <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
                        Xem tất cả →
                    </button>
                </div>
            )}
        </div>
    );
};

export default RecentAnalyses;