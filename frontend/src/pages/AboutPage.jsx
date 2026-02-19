import { motion } from 'framer-motion';
import {
    Brain, Activity, Dna, Pill, BarChart3, Database,
    Cpu, Shield, Code, AlertTriangle, Clock
} from 'lucide-react';
import './AboutPage.css';

const fadeUp = (d = 0) => ({
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5, delay: d },
});

const models = [
    { name: 'XGBoost', accuracy: 97.27, color: '#10b981' },
    { name: 'SVM', accuracy: 93.06, color: '#3b82f6' },
    { name: 'Transformer', accuracy: 91.32, color: '#f59e0b' },
    { name: 'Ensemble', accuracy: 97.21, color: '#6366f1' },
];

export default function AboutPage() {
    return (
        <div className="container">
            <motion.div {...fadeUp(0)} style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
                <span className="section-title">About NeuroAssess</span>
                <h2>AI-Powered Parkinson's Disease Assessment</h2>
                <p style={{ maxWidth: 650, margin: '0.5rem auto 0' }}>
                    Combining traditional machine learning with advanced transformer models
                    to provide comprehensive diagnostic support.
                </p>
            </motion.div>

            {/* PD Overview */}
            <motion.div {...fadeUp(0.1)} className="glass-card-static" style={{ marginBottom: '1.5rem' }}>
                <div className="about-header">
                    <Brain size={22} />
                    <h3>Understanding Parkinson's Disease</h3>
                </div>
                <p style={{ marginBottom: '1.5rem' }}>
                    Parkinson's disease is a progressive neurodegenerative disorder affecting movement control.
                    It occurs when neurons in the substantia nigra become impaired, reducing dopamine production
                    and leading to characteristic motor symptoms.
                </p>

                <div className="grid-2" style={{ gap: '1rem' }}>
                    <div className="info-box">
                        <h4><Activity size={16} /> Motor Symptoms</h4>
                        <ul>
                            <li><strong>Tremor:</strong> Rhythmic shaking, often starting in hands at rest</li>
                            <li><strong>Rigidity:</strong> Muscle stiffness in limbs and trunk</li>
                            <li><strong>Bradykinesia:</strong> Slowness of movement</li>
                            <li><strong>Postural Instability:</strong> Impaired balance</li>
                        </ul>
                    </div>
                    <div className="info-box">
                        <h4><Brain size={16} /> Non-Motor Symptoms</h4>
                        <ul>
                            <li><strong>Cognitive:</strong> Memory problems, slowed thinking</li>
                            <li><strong>Mood:</strong> Depression, anxiety, apathy</li>
                            <li><strong>Sleep:</strong> REM behaviour disorder, insomnia</li>
                            <li><strong>Sensory:</strong> Loss of smell, pain</li>
                        </ul>
                    </div>
                    <div className="info-box">
                        <h4><Dna size={16} /> Risk Factors</h4>
                        <ul>
                            <li><strong>Age:</strong> Risk increases after age 60</li>
                            <li><strong>Genetics:</strong> Family history (5-10% hereditary)</li>
                            <li><strong>Gender:</strong> Men 1.5× more likely</li>
                            <li><strong>Environmental:</strong> Pesticide/toxin exposure</li>
                        </ul>
                    </div>
                    <div className="info-box">
                        <h4><Pill size={16} /> Treatment Options</h4>
                        <ul>
                            <li><strong>Levodopa/Carbidopa:</strong> Most effective medication</li>
                            <li><strong>DBS:</strong> Deep brain stimulation surgery</li>
                            <li><strong>Physical therapy:</strong> Exercise programs</li>
                            <li><strong>Support:</strong> Occupational and speech therapy</li>
                        </ul>
                    </div>
                </div>

                <div className="alert alert-info" style={{ marginTop: '1.25rem' }}>
                    <Clock size={16} />
                    <span>
                        <strong>Disease Progression:</strong> Symptoms typically start on one side and gradually affect both sides.
                        Early diagnosis and treatment can help manage symptoms for many years.
                    </span>
                </div>
            </motion.div>

            {/* Model Performance */}
            <motion.div {...fadeUp(0.2)} className="grid-2" style={{ marginBottom: '1.5rem' }}>
                <div className="glass-card-static">
                    <div className="about-header">
                        <BarChart3 size={22} />
                        <h3>Model Performance</h3>
                    </div>
                    {models.map((m) => (
                        <div key={m.name} className="model-bar">
                            <div className="model-bar-header">
                                <span>{m.name}</span>
                                <span style={{ fontWeight: 700, color: m.color }}>{m.accuracy}%</span>
                            </div>
                            <div className="progress-bar">
                                <div className="progress-fill" style={{ width: `${m.accuracy}%`, background: m.color }} />
                            </div>
                        </div>
                    ))}
                    <p className="footnote">
                        <Shield size={12} /> Performance metrics based on PPMI dataset validation
                    </p>
                </div>

                <div className="glass-card-static">
                    <div className="about-header">
                        <Database size={22} />
                        <h3>Dataset Details</h3>
                    </div>
                    <div className="grid-2" style={{ gap: '0.75rem', marginBottom: '1rem' }}>
                        <div className="stat-box"><span className="gradient-text" style={{ fontSize: '1.5rem', fontWeight: 800 }}>22,402</span><small>Total Samples</small></div>
                        <div className="stat-box"><span className="gradient-text" style={{ fontSize: '1.5rem', fontWeight: 800 }}>24+</span><small>Features</small></div>
                    </div>
                    <h5 style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>Class Distribution</h5>
                    {[
                        { name: 'Healthy Control', badge: 'badge-success' },
                        { name: "Parkinson's Disease", badge: 'badge-danger' },
                        { name: 'SWEDD', badge: 'badge-warning' },
                        { name: 'Prodromal PD', badge: 'badge-info' },
                    ].map((c) => (
                        <div key={c.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                            <span style={{ fontSize: '0.9rem' }}>{c.name}</span>
                            <span className={`badge ${c.badge}`}>{c.name.split(' ')[0]}</span>
                        </div>
                    ))}
                </div>
            </motion.div>

            {/* Architecture */}
            <motion.div {...fadeUp(0.3)} className="glass-card-static" style={{ marginBottom: '1.5rem' }}>
                <div className="about-header">
                    <Cpu size={22} />
                    <h3>Technical Architecture</h3>
                </div>
                <div className="grid-3">
                    <div className="info-box">
                        <h4><Code size={16} /> Data Pipeline</h4>
                        <ul>
                            <li>Feature selection & engineering</li>
                            <li>Missing value imputation</li>
                            <li>Leak-free patient splits</li>
                        </ul>
                    </div>
                    <div className="info-box">
                        <h4><Cpu size={16} /> ML Models</h4>
                        <ul>
                            <li>XGBoost, SVM, LightGBM</li>
                            <li>PubMedBERT, BioGPT, Clinical-T5</li>
                            <li>Stacking ensemble classifier</li>
                        </ul>
                    </div>
                    <div className="info-box">
                        <h4><Brain size={16} /> RAG System</h4>
                        <ul>
                            <li>Medical knowledge base</li>
                            <li>Automated report generation</li>
                            <li>Clinical recommendations</li>
                        </ul>
                    </div>
                </div>
            </motion.div>

            {/* Disclaimer */}
            <motion.div {...fadeUp(0.4)} className="alert alert-warning" style={{ marginBottom: '2rem' }}>
                <AlertTriangle size={18} />
                <div>
                    <strong>Disclaimer:</strong> This AI system provides diagnostic support for research and educational
                    purposes. It should not replace professional medical judgment. All predictions should be validated
                    by qualified healthcare providers.
                </div>
            </motion.div>
        </div>
    );
}
