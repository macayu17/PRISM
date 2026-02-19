import { Outlet } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import Navbar from './Navbar';
import LiquidBackground from './LiquidBackground'; // Import
import { Brain, Github } from 'lucide-react';

const pageTransition = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
    transition: { duration: 0.3, ease: 'easeOut' },
};

export default function Layout() {
    return (
        <>
            <LiquidBackground />
            <Navbar />
            <main className="page-wrapper">
                <motion.div {...pageTransition}>
                    <Outlet />
                </motion.div>
            </main>
            <footer style={{
                borderTop: '1px solid var(--border-color)',
                padding: '1.5rem 0',
                textAlign: 'center',
                color: 'var(--text-muted)',
                fontSize: '0.85rem',
            }}>
                <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                    <Brain size={16} />
                    <span>NeuroAssess — Parkinson's Disease Assessment System</span>
                    <span style={{ margin: '0 0.5rem' }}>•</span>
                    <span>Research & Educational Use Only</span>
                </div>
            </footer>
        </>
    );
}
