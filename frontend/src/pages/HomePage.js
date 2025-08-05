import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
    Send,
    FileText,
    Loader2,
    Smile,
    Frown,
    Meh,
    TrendingUp,
    Users,
    Zap
} from 'lucide-react';
import TextareaAutosize from 'react-textarea-autosize';
import toast from 'react-hot-toast';

import { useSentiment } from '../context/SentimentContext';
import SentimentResult from '../components/SentimentResult';
import RecentAnalyses from '../components/RecentAnalyses';

const HomePage = () => {
    const [text, setText] = useState('');
    const [batchTexts, setBatchTexts] = useState('');
    const [mode, setMode] = useState('single'); // 'single' or 'batch'

    const {
        analyzeSentiment,
        analyzeBatch,
        loading,
        error,
        analytics,
        analyses
    } = useSentiment();

    const handleSingleAnalysis = async () => {
        if (!text.trim()) {
            toast.error('Vui lòng nhập văn bản để phân tích');
            return;
        }

        try {
            await analyzeSentiment(text.trim());
            toast.success('Phân tích thành công!');
            setText('');
        } catch (error) {
            toast.error('Lỗi phân tích: ' + error.message);
        }
    };

    const handleBatchAnalysis = async () => {
        if (!batchTexts.trim()) {
            toast.error('Vui lòng nhập các văn bản để phân tích');
            return;
        }

        const texts = batchTexts
            .split('\n')
            .map(t => t.trim())
            .filter(t => t.length > 0);

        if (texts.length === 0) {
            toast.error('Không có văn bản hợp lệ để phân tích');
            return;
        }

        if (texts.length > 50) {
            toast.error('Tối đa 50 văn bản mỗi lần phân tích');
            return;
        }

        try {
            const result = await analyzeBatch(texts);
            toast.success(`Phân tích thành công ${result.total_count} văn bản!`);
            setBatchTexts('');
        } catch (error) {
            toast.error('Lỗi phân tích: ' + error.message);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            if (mode === 'single') {
                handleSingleAnalysis();
            } else {
                handleBatchAnalysis();
            }
        }
    };

    const getSentimentIcon = (sentiment) => {
        switch (sentiment) {
            case 'positive': return <Smile className="text-green-500" size={20} />;
            case 'negative': return <Frown className="text-red-500" size={20} />;
            case 'neutral': return <Meh className="text-gray-500" size={20} />;
            default: return null;
        }
    };

    const getSentimentColor = (sentiment) => {
        switch (sentiment) {
            case 'positive': return 'text-green-600 bg-green-50';
            case 'negative': return 'text-red-600 bg-red-50';
            case 'neutral': return 'text-gray-600 bg-gray-50';
            default: return 'text-gray-600 bg-gray-50';
        }
    };

    return (
        <div className="max-w-7xl mx-auto space-y-8">
            {/* Hero Section */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center space-y-4"
            >
                <h1 className="text-4xl md:text-5xl font-bold text-gradient">
                    Phân tích cảm xúc tiếng Việt
                </h1>
                <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                    Sử dụng mô hình PhoBERT để phân tích cảm xúc trong văn bản tiếng Việt
                    với độ chính xác cao. Hỗ trợ phân tích theo thời gian thực và batch processing.
                </p>
            </motion.div>

            {/* Stats Cards */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="grid grid-cols-1 md:grid-cols-3 gap-6"
            >
                <div className="card text-center">
                    <div className="flex items-center justify-center w-12 h-12 bg-blue-100 rounded-lg mx-auto mb-4">
                        <TrendingUp className="text-blue-600" size={24} />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900">
                        {analytics.total_analyses.toLocaleString()}
                    </h3>
                    <p className="text-gray-600">Tổng số phân tích</p>
                </div>

                <div className="card text-center">
                    <div className="flex items-center justify-center w-12 h-12 bg-green-100 rounded-lg mx-auto mb-4">
                        <Zap className="text-green-600" size={24} />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900">
                        {(analytics.average_confidence * 100).toFixed(1)}%
                    </h3>
                    <p className="text-gray-600">Độ tin cậy trung bình</p>
                </div>

                <div className="card text-center">
                    <div className="flex items-center justify-center w-12 h-12 bg-purple-100 rounded-lg mx-auto mb-4">
                        <Users className="text-purple-600" size={24} />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900">
                        {Object.values(analytics.sentiment_distribution).reduce((a, b) => a + b, 0)}
                    </h3>
                    <p className="text-gray-600">Phân tích hôm nay</p>
                </div>
            </motion.div>

            {/* Analysis Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Input Section */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="space-y-6"
                >
                    <div className="card">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-2xl font-bold text-gray-900">
                                Phân tích cảm xúc
                            </h2>

                            {/* Mode Toggle */}
                            <div className="flex bg-gray-100 rounded-lg p-1">
                                <button
                                    onClick={() => setMode('single')}
                                    className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${mode === 'single'
                                            ? 'bg-white text-gray-900 shadow-sm'
                                            : 'text-gray-600 hover:text-gray-900'
                                        }`}
                                >
                                    Đơn lẻ
                                </button>
                                <button
                                    onClick={() => setMode('batch')}
                                    className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${mode === 'batch'
                                            ? 'bg-white text-gray-900 shadow-sm'
                                            : 'text-gray-600 hover:text-gray-900'
                                        }`}
                                >
                                    Hàng loạt
                                </button>
                            </div>
                        </div>

                        {mode === 'single' ? (
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Nhập văn bản cần phân tích
                                    </label>
                                    <TextareaAutosize
                                        value={text}
                                        onChange={(e) => setText(e.target.value)}
                                        onKeyDown={handleKeyPress}
                                        placeholder="Ví dụ: Thầy giảng rất hay và nhiệt tình..."
                                        className="input-field resize-none"
                                        minRows={4}
                                        maxRows={8}
                                    />
                                    <p className="text-xs text-gray-500 mt-1">
                                        Nhấn Ctrl+Enter để phân tích nhanh
                                    </p>
                                </div>

                                <button
                                    onClick={handleSingleAnalysis}
                                    disabled={loading || !text.trim()}
                                    className="btn-primary w-full flex items-center justify-center space-x-2"
                                >
                                    {loading ? (
                                        <Loader2 className="animate-spin" size={20} />
                                    ) : (
                                        <Send size={20} />
                                    )}
                                    <span>{loading ? 'Đang phân tích...' : 'Phân tích cảm xúc'}</span>
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Nhập nhiều văn bản (mỗi dòng một văn bản)
                                    </label>
                                    <TextareaAutosize
                                        value={batchTexts}
                                        onChange={(e) => setBatchTexts(e.target.value)}
                                        onKeyDown={handleKeyPress}
                                        placeholder={`Thầy giảng rất hay và nhiệt tình
Môn học này khó quá
Bài giảng rất dễ hiểu`}
                                        className="input-field resize-none"
                                        minRows={6}
                                        maxRows={12}
                                    />
                                    <p className="text-xs text-gray-500 mt-1">
                                        Tối đa 50 văn bản. Nhấn Ctrl+Enter để phân tích nhanh
                                    </p>
                                </div>

                                <button
                                    onClick={handleBatchAnalysis}
                                    disabled={loading || !batchTexts.trim()}
                                    className="btn-primary w-full flex items-center justify-center space-x-2"
                                >
                                    {loading ? (
                                        <Loader2 className="animate-spin" size={20} />
                                    ) : (
                                        <FileText size={20} />
                                    )}
                                    <span>{loading ? 'Đang phân tích...' : 'Phân tích hàng loạt'}</span>
                                </button>
                            </div>
                        )}

                        {error && (
                            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                                <p className="text-red-600 text-sm">{error}</p>
                            </div>
                        )}
                    </div>

                    {/* Sentiment Distribution */}
                    <div className="card">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">
                            Phân bố cảm xúc
                        </h3>
                        <div className="space-y-3">
                            {Object.entries(analytics.sentiment_distribution).map(([sentiment, count]) => {
                                const total = Object.values(analytics.sentiment_distribution).reduce((a, b) => a + b, 0);
                                const percentage = total > 0 ? (count / total * 100) : 0;

                                return (
                                    <div key={sentiment} className="flex items-center justify-between">
                                        <div className="flex items-center space-x-2">
                                            {getSentimentIcon(sentiment)}
                                            <span className="capitalize font-medium">
                                                {sentiment === 'positive' ? 'Tích cực' :
                                                    sentiment === 'negative' ? 'Tiêu cực' : 'Trung lập'}
                                            </span>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <div className="w-20 bg-gray-200 rounded-full h-2">
                                                <div
                                                    className={`h-2 rounded-full ${sentiment === 'positive' ? 'bg-green-500' :
                                                            sentiment === 'negative' ? 'bg-red-500' : 'bg-gray-500'
                                                        }`}
                                                    style={{ width: `${percentage}%` }}
                                                />
                                            </div>
                                            <span className="text-sm font-medium w-12 text-right">
                                                {percentage.toFixed(1)}%
                                            </span>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </motion.div>

                {/* Results Section */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="space-y-6"
                >
                    {/* Latest Result */}
                    {analyses.length > 0 && (
                        <SentimentResult analysis={analyses[0]} />
                    )}

                    {/* Recent Analyses */}
                    <RecentAnalyses analyses={analyses.slice(0, 5)} />
                </motion.div>
            </div>
        </div>
    );
};

export default HomePage;