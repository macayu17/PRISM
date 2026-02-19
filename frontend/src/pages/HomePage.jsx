import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
    Brain, Stethoscope, FileText, Shield, ArrowRight,
    Cpu, Activity, Zap, CheckCircle, Server
} from 'lucide-react';
import BrainScene from '../components/BrainScene';
import { getSystemStatus } from '../api/client';
import './HomePage.css';

const fadeUp = (delay = 0) => ({
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6, delay, ease: 'easeOut' },
});

export default function HomePage() {
    const [status, setStatus] = useState(null);

    useEffect(() => {
        getSystemStatus().then(setStatus).catch(() => setStatus(null));
        const id = setInterval(() => {
            getSystemStatus().then(setStatus).catch(() => setStatus(null));
        }, 30000);
        return () => clearInterval(id);
    }, []);

    return (
        <div className="container">
            {/* Hero */}
            <section className="hero-section">
                <div className="hero-content">
                    <motion.div {...fadeUp(0)}>
                        <span className="section-title">AI-Powered Diagnostics</span>
                    </motion.div>
                    <motion.h1 {...fadeUp(0.1)} className="hero-title">
                        Parkinson's Disease<br />
                        <span className="gradient-text">Assessment System</span>
                    </motion.h1>
                    <motion.p {...fadeUp(0.2)} className="hero-description">
                        Advanced multimodal machine learning combining transformers and ensemble
                        methods for comprehensive diagnostic support and early detection.
                    </motion.p>
                    <motion.div {...fadeUp(0.3)} className="hero-buttons">
                        <Link to="/assessment" className="btn-primary">
                            <Stethoscope size={18} />
                            Start Assessment
                            <ArrowRight size={16} />
                        </Link>
                        <Link to="/about" className="btn-secondary">
                            Learn More
                        </Link>
                    </motion.div>
                </div>
                <motion.div
                    className="hero-brain"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 1, delay: 0.3 }}
                >
                    <BrainScene />
                </motion.div>
            </section>

            {/* Stats */}
            <motion.section {...fadeUp(0.4)} className="stats-row">
                {[
                    { label: 'XGBoost Accuracy', value: '97.27%', icon: <Zap size={20} /> },
                    { label: 'Transformer Accuracy', value: '91.32%', icon: <Cpu size={20} /> },
                    { label: 'Patient Features', value: '24+', icon: <Activity size={20} /> },
                    { label: 'Diagnostic Classes', value: '4', icon: <Brain size={20} /> },
                ].map((s, i) => (
                    <div key={i} className="glass-card stat-card">
                        <div className="stat-icon">{s.icon}</div>
                        <div className="stat-value">{s.value}</div>
                        <div className="stat-label">{s.label}</div>
                    </div>
                ))}
            </motion.section>

            {/* Features */}
            <motion.section {...fadeUp(0.5)} style={{ marginTop: '3rem' }}>
                <h2 style={{ textAlign: 'center', marginBottom: '0.5rem' }}>
                    Why <span className="gradient-text">NeuroAssess</span>?
                </h2>
                <p style={{ textAlign: 'center', maxWidth: 600, margin: '0 auto 2rem' }}>
                    Combining cutting-edge AI with clinical research standards
                </p>
                <div className="grid-3">
                    {[
                        {
                            icon: <Cpu size={28} />,
                            title: 'AI-Powered Analysis',
                            desc: 'Multimodal ensemble combining XGBoost, SVM, and medical transformers (PubMedBERT, BioGPT, Clinical-T5).',
                            items: ['97.27% Accuracy', 'Ensemble Consensus', 'Real-time Predictions'],
                        },
                        {
                            icon: <FileText size={28} />,
                            title: 'Automated Reports',
                            desc: 'Generate detailed clinical reports with diagnostic predictions, feature analysis, and evidence-based recommendations.',
                            items: ['Clinical Summaries', 'Risk Assessment', 'PDF Export'],
                        },
                        {
                            icon: <Shield size={28} />,
                            title: 'Research-Grade Quality',
                            desc: 'Built on validated PPMI clinical datasets and established diagnostic criteria for Parkinson\'s disease.',
                            items: ['PPMI Dataset Trained', 'Clinical Validation', 'Leak-Free Splits'],
                        },
                    ].map((f, i) => (
                        <div key={i} className="glass-card feature-card">
                            <div className="feature-icon">{f.icon}</div>
                            <h4>{f.title}</h4>
                            <p>{f.desc}</p>
                            <ul className="feature-list">
                                {f.items.map((item, j) => (
                                    <li key={j}><CheckCircle size={14} /> {item}</li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </div>
            </motion.section>

            {/* Diagnostic Categories */}
            <motion.section {...fadeUp(0.6)} style={{ marginTop: '3rem' }}>
                <h2 style={{ textAlign: 'center', marginBottom: '2rem' }}>
                    Diagnostic Categories
                </h2>
                <div className="grid-4">
                    {[
                        { label: 'Healthy Control', abbr: 'HC', color: '#10b981', desc: 'No signs of movement disorders' },
                        { label: "Parkinson's Disease", abbr: 'PD', color: '#ef4444', desc: 'Characteristic motor symptoms' },
                        { label: 'SWEDD', abbr: 'SWEDD', color: '#f59e0b', desc: 'Symptoms without dopamine deficit' },
                        { label: 'Prodromal PD', abbr: 'PROD', color: '#3b82f6', desc: 'Early stage, subtle symptoms' },
                    ].map((c, i) => (
                        <div key={i} className="glass-card category-card" style={{ '--cat-color': c.color }}>
                            <div className="category-dot" />
                            <h4>{c.label}</h4>
                            <span className="badge badge-accent">{c.abbr}</span>
                            <p style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>{c.desc}</p>
                        </div>
                    ))}
                </div>
            </motion.section>

            {/* System Status */}
            <motion.section {...fadeUp(0.7)} style={{ marginTop: '3rem', marginBottom: '2rem' }}>
                <div className="glass-card-static" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <Server size={24} style={{ color: status?.system_initialized ? 'var(--success)' : 'var(--danger)' }} />
                    <div style={{ flex: 1 }}>
                        <h4 style={{ fontSize: '0.95rem', marginBottom: '0.2rem' }}>System Status</h4>
                        <p style={{ fontSize: '0.85rem', margin: 0 }}>
                            {status?.system_initialized && status?.models_loaded
                                ? 'All systems operational. Ready for assessments.'
                                : status?.system_initialized
                                    ? 'System initializing. Models loading...'
                                    : 'Connect Flask backend on port 5000 to enable predictions.'}
                        </p>
                    </div>
                    <span className={`badge ${status?.models_loaded ? 'badge-success' : 'badge-warning'}`}>
                        {status?.models_loaded ? 'Online' : 'Offline'}
                    </span>
                </div>
            </motion.section>
        </div>
    );
}
