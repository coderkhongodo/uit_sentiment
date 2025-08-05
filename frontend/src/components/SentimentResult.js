import React from 'react';
import { motion } from 'framer-motion';
import { Smile, Frown, Meh, Clock, TrendingUp } from 'lucide-react';
import { formatTimestamp } from '../utils/dateUtils';

const SentimentResult = ({ analysis }) => {
    if (!analysis) return null;

    const getSentimentIcon = (sentiment) => {
        switch (sentiment) {
            case 'positive': return <Smile className="text-green-500" size={24} />;
            case 'negative': return <Frown className="text-red-500" size={24} />;
            case 'neutral': return <Meh className="text-gray-500" size={24} />;
            default: return null;
        }
    };

    const getSentimentColor = (sentiment) => {
        switch (sentiment) {
            case 'positive': return 'text-green-600 bg-green-50 border-green-200';
            case 'negative': return 'text-red-600 bg-red-50 border-red-200';
            case 'neutral': return 'text-gray-600 bg-gray-50 border-gray-200';
            default: return 'text-gray-600 bg-gray-50 border-gray-200';
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

    const getConfidenceColor = (confidence) => {
        if (confidence >= 0.8) return 'text-green-600 bg-green-100';
        if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100';
        return 'text-red-600 bg-red-100';
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card"
        >
            <div className="flex items-start justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                    Kết quả phân tích mới nhất
                </h3>
                <div className="flex items-center space-x-1 text-gray-500 text-sm">
                    <Clock size={14} />
                    <span>
                        {formatTimestamp(analysis.timestamp)}
                    </span>
                </div>
            </div>

            {/* Text */}
            <div className="mb-4">
                <p className="text-gray-700 bg-gray-50 p-3 rounded-lg border">
                    "{analysis.text}"
                </p>
            </div>

            {/* Main Result */}
            <div className={`flex items-center justify-between p-4 rounded-lg border-2 ${getSentimentColor(analysis.sentiment)}`}>
                <div className="flex items-center space-x-3">
                    {getSentimentIcon(analysis.sentiment)}
                    <div>
                        <h4 className="font-semibold text-lg">
                            {getSentimentLabel(analysis.sentiment)}
                        </h4>
                        <p className="text-sm opacity-75">
                            Cảm xúc được phát hiện
                        </p>
                    </div>
                </div>

                <div className="text-right">
                    <div className={`inline-flex items-center space-x-1 px-2 py-1 rounded-full text-sm font-medium ${getConfidenceColor(analysis.confidence)}`}>
                        <TrendingUp size={14} />
                        <span>{(analysis.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <p className="text-xs opacity-75 mt-1">Độ tin cậy</p>
                </div>
            </div>

            {/* Probability Breakdown */}
            <div className="mt-4">
                <h5 className="text-sm font-medium text-gray-700 mb-3">
                    Chi tiết xác suất:
                </h5>
                <div className="space-y-2">
                    {Object.entries(analysis.probabilities).map(([sentiment, probability]) => (
                        <div key={sentiment} className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                                {getSentimentIcon(sentiment)}
                                <span className="text-sm font-medium">
                                    {getSentimentLabel(sentiment)}
                                </span>
                            </div>
                            <div className="flex items-center space-x-2">
                                <div className="w-24 bg-gray-200 rounded-full h-2">
                                    <div
                                        className={`h-2 rounded-full ${sentiment === 'positive' ? 'bg-green-500' :
                                            sentiment === 'negative' ? 'bg-red-500' : 'bg-gray-500'
                                            }`}
                                        style={{ width: `${probability * 100}%` }}
                                    />
                                </div>
                                <span className="text-sm font-medium w-12 text-right">
                                    {(probability * 100).toFixed(1)}%
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Analysis ID */}
            <div className="mt-4 pt-3 border-t border-gray-200">
                <p className="text-xs text-gray-500">
                    ID: {analysis.analysis_id}
                </p>
            </div>
        </motion.div>
    );
};

export default SentimentResult;