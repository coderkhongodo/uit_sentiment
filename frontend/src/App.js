import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion } from 'framer-motion';

// Components
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import DashboardPage from './pages/DashboardPage';
import AnalyticsPage from './pages/AnalyticsPage';
import AdminPage from './pages/AdminPage';

// Context
import { SentimentProvider } from './context/SentimentContext';

function App() {
    return (
        <SentimentProvider>
            <Router>
                <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
                    <Navbar />

                    <motion.main
                        className="container mx-auto px-4 py-8"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                    >
                        <Routes>
                            <Route path="/" element={<HomePage />} />
                            <Route path="/dashboard" element={<DashboardPage />} />
                            <Route path="/analytics" element={<AnalyticsPage />} />
                            <Route path="/admin" element={<AdminPage />} />
                        </Routes>
                    </motion.main>
                </div>
            </Router>
        </SentimentProvider>
    );
}

export default App;