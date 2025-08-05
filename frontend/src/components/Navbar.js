import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
    Home,
    BarChart3,
    TrendingUp,
    Settings,
    Menu,
    X,
    Wifi,
    WifiOff
} from 'lucide-react';
import { useSentiment } from '../context/SentimentContext';

const Navbar = () => {
    const [isOpen, setIsOpen] = useState(false);
    const location = useLocation();
    const { connected } = useSentiment();

    const navigation = [
        { name: 'Trang chủ', href: '/', icon: Home },
        { name: 'Dashboard', href: '/dashboard', icon: BarChart3 },
        { name: 'Phân tích', href: '/analytics', icon: TrendingUp },
        { name: 'Quản trị', href: '/admin', icon: Settings },
    ];

    const isActive = (path) => location.pathname === path;

    return (
        <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                    {/* Logo */}
                    <div className="flex items-center">
                        <Link to="/" className="flex items-center space-x-2">
                            <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                                <span className="text-white font-bold text-sm">SA</span>
                            </div>
                            <span className="text-xl font-bold text-gradient">
                                Sentiment Analysis
                            </span>
                        </Link>
                    </div>

                    {/* Desktop Navigation */}
                    <div className="hidden md:flex items-center space-x-8">
                        {navigation.map((item) => {
                            const Icon = item.icon;
                            return (
                                <Link
                                    key={item.name}
                                    to={item.href}
                                    className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200 ${isActive(item.href)
                                            ? 'bg-primary-100 text-primary-700'
                                            : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                                        }`}
                                >
                                    <Icon size={18} />
                                    <span>{item.name}</span>
                                </Link>
                            );
                        })}

                        {/* Connection Status */}
                        <div className="flex items-center space-x-2">
                            {connected ? (
                                <div className="flex items-center space-x-1 text-green-600">
                                    <Wifi size={16} />
                                    <span className="text-xs">Kết nối</span>
                                </div>
                            ) : (
                                <div className="flex items-center space-x-1 text-red-600">
                                    <WifiOff size={16} />
                                    <span className="text-xs">Mất kết nối</span>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Mobile menu button */}
                    <div className="md:hidden flex items-center">
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            className="text-gray-600 hover:text-gray-900 focus:outline-none focus:text-gray-900 p-2"
                        >
                            {isOpen ? <X size={24} /> : <Menu size={24} />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Navigation */}
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="md:hidden bg-white border-t border-gray-200"
                >
                    <div className="px-2 pt-2 pb-3 space-y-1">
                        {navigation.map((item) => {
                            const Icon = item.icon;
                            return (
                                <Link
                                    key={item.name}
                                    to={item.href}
                                    onClick={() => setIsOpen(false)}
                                    className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-base font-medium transition-colors duration-200 ${isActive(item.href)
                                            ? 'bg-primary-100 text-primary-700'
                                            : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                                        }`}
                                >
                                    <Icon size={20} />
                                    <span>{item.name}</span>
                                </Link>
                            );
                        })}

                        {/* Mobile Connection Status */}
                        <div className="px-3 py-2">
                            {connected ? (
                                <div className="flex items-center space-x-2 text-green-600">
                                    <Wifi size={16} />
                                    <span className="text-sm">Đã kết nối</span>
                                </div>
                            ) : (
                                <div className="flex items-center space-x-2 text-red-600">
                                    <WifiOff size={16} />
                                    <span className="text-sm">Mất kết nối</span>
                                </div>
                            )}
                        </div>
                    </div>
                </motion.div>
            )}
        </nav>
    );
};

export default Navbar;