import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
    Brain, Activity, Dna, Pill, BarChart3, Database,
    Cpu, Shield, Code, AlertTriangle, Clock
} from 'lucide-react';
import { getModelMetricsSummary } from '../api/client';
import {
    alertClass,
    badgeClass,
    glassPanel,
    innerPanel,
    pageShell,
    progressTrack,
    sectionHeading,
    sectionTitle,
} from '../lib/ui';

const fadeUp = (d = 0) => ({
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5, delay: d },
});

const modelTextClasses = ['text-emerald-400', 'text-blue-400', 'text-amber-400', 'text-indigo-400', 'text-rose-400', 'text-teal-400'];
const modelBgClasses = ['bg-emerald-500', 'bg-blue-500', 'bg-amber-500', 'bg-indigo-500', 'bg-rose-500', 'bg-teal-500'];

export default function AboutPage() {
    const [metrics, setMetrics] = useState([]);

    useEffect(() => {
        getModelMetricsSummary()
            .then((data) => setMetrics(Array.isArray(data?.models) ? data.models : []))
            .catch(() => setMetrics([]));
    }, []);

    return (
        <div className={pageShell}>
            <motion.div {...fadeUp(0)} className="mb-10 text-center">
                <span className={sectionTitle}>About NeuroAssess</span>
                <h2>AI-Powered Parkinson&apos;s Disease Assessment</h2>
                <p className="mx-auto mt-2 max-w-2xl text-base leading-7 text-slate-400">
                    Combining traditional machine learning with advanced transformer models
                    to provide comprehensive diagnostic support.
                </p>
            </motion.div>

            <motion.div {...fadeUp(0.1)} className={`${glassPanel} mb-6 bg-black/25`}>
                <div className={sectionHeading}>
                    <Brain size={22} />
                    <h3>Understanding Parkinson&apos;s Disease</h3>
                </div>
                <p className="mb-6 text-base leading-8 text-slate-400">
                    Parkinson&apos;s disease is a progressive neurodegenerative disorder affecting movement control.
                    It occurs when neurons in the substantia nigra become impaired, reducing dopamine production
                    and leading to characteristic motor symptoms.
                </p>

                <div className="grid gap-4 md:grid-cols-2">
                    <div className={innerPanel}>
                        <h4 className="mb-4 flex items-center gap-2 text-base"><Activity size={16} /> Motor Symptoms</h4>
                        <ul className="space-y-2 text-sm leading-6 text-slate-300">
                            <li><strong>Tremor:</strong> Rhythmic shaking, often starting in hands at rest</li>
                            <li><strong>Rigidity:</strong> Muscle stiffness in limbs and trunk</li>
                            <li><strong>Bradykinesia:</strong> Slowness of movement</li>
                            <li><strong>Postural Instability:</strong> Impaired balance</li>
                        </ul>
                    </div>
                    <div className={innerPanel}>
                        <h4 className="mb-4 flex items-center gap-2 text-base"><Brain size={16} /> Non-Motor Symptoms</h4>
                        <ul className="space-y-2 text-sm leading-6 text-slate-300">
                            <li><strong>Cognitive:</strong> Memory problems, slowed thinking</li>
                            <li><strong>Mood:</strong> Depression, anxiety, apathy</li>
                            <li><strong>Sleep:</strong> REM behaviour disorder, insomnia</li>
                            <li><strong>Sensory:</strong> Loss of smell, pain</li>
                        </ul>
                    </div>
                    <div className={innerPanel}>
                        <h4 className="mb-4 flex items-center gap-2 text-base"><Dna size={16} /> Risk Factors</h4>
                        <ul className="space-y-2 text-sm leading-6 text-slate-300">
                            <li><strong>Age:</strong> Risk increases after age 60</li>
                            <li><strong>Genetics:</strong> Family history (5-10% hereditary)</li>
                            <li><strong>Gender:</strong> Men 1.5× more likely</li>
                            <li><strong>Environmental:</strong> Pesticide/toxin exposure</li>
                        </ul>
                    </div>
                    <div className={innerPanel}>
                        <h4 className="mb-4 flex items-center gap-2 text-base"><Pill size={16} /> Treatment Options</h4>
                        <ul className="space-y-2 text-sm leading-6 text-slate-300">
                            <li><strong>Levodopa/Carbidopa:</strong> Most effective medication</li>
                            <li><strong>DBS:</strong> Deep brain stimulation surgery</li>
                            <li><strong>Physical therapy:</strong> Exercise programs</li>
                            <li><strong>Support:</strong> Occupational and speech therapy</li>
                        </ul>
                    </div>
                </div>

                <div className={`${alertClass('info')} mt-5`}>
                    <Clock size={16} className="mt-0.5 shrink-0" />
                    <span>
                        <strong>Disease Progression:</strong> Symptoms typically start on one side and gradually affect both sides.
                        Early diagnosis and treatment can help manage symptoms for many years.
                    </span>
                </div>
            </motion.div>

            <motion.div {...fadeUp(0.2)} className="mb-6 grid gap-6 xl:grid-cols-2">
                <div className={`${glassPanel} bg-black/25`}>
                    <div className={sectionHeading}>
                        <BarChart3 size={22} />
                        <h3>Model Performance</h3>
                    </div>
                    {metrics.length > 0 ? (
                        <div className="space-y-5">
                            {metrics.map((metric, index) => (
                                <div key={metric.name}>
                                    <div className="mb-2 flex items-center justify-between gap-4 text-sm font-medium">
                                        <span>{metric.name}</span>
                                        <span className={`font-bold ${modelTextClasses[index % modelTextClasses.length]}`}>
                                            {metric.accuracy_pct.toFixed(2)}%
                                        </span>
                                    </div>
                                    <div className={progressTrack}>
                                        <div
                                            className={`h-full rounded-full ${modelBgClasses[index % modelBgClasses.length]}`}
                                            style={{
                                                width: `${metric.accuracy_pct}%`,
                                            }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-sm text-slate-400">Evaluation metrics are unavailable right now.</p>
                    )}
                    <p className="mt-4 flex items-center gap-2 text-xs text-slate-500">
                        <Shield size={12} /> Performance metrics based on PPMI dataset validation
                    </p>
                </div>

                <div className={`${glassPanel} bg-black/25`}>
                    <div className={sectionHeading}>
                        <Database size={22} />
                        <h3>Dataset Details</h3>
                    </div>
                    <div className="mb-4 grid gap-3 sm:grid-cols-2">
                        <div className={`${innerPanel} flex flex-col items-center text-center`}>
                            <span className="bg-gradient-to-r from-sky-300 via-sky-400 to-indigo-300 bg-clip-text text-3xl font-extrabold text-transparent">
                                22,402
                            </span>
                            <small className="mt-2 text-xs uppercase tracking-[0.18em] text-slate-500">
                                Total Samples
                            </small>
                        </div>
                        <div className={`${innerPanel} flex flex-col items-center text-center`}>
                            <span className="bg-gradient-to-r from-sky-300 via-sky-400 to-indigo-300 bg-clip-text text-3xl font-extrabold text-transparent">
                                24+
                            </span>
                            <small className="mt-2 text-xs uppercase tracking-[0.18em] text-slate-500">
                                Features
                            </small>
                        </div>
                    </div>
                    <h5 className="mb-3 text-sm">Class Distribution</h5>
                    <div className="space-y-2">
                        {[
                            { name: 'Healthy Control', badge: 'success' },
                            { name: "Parkinson's Disease", badge: 'danger' },
                            { name: 'SWEDD', badge: 'warning' },
                            { name: 'Prodromal PD', badge: 'info' },
                        ].map((item) => (
                            <div key={item.name} className="flex items-center justify-between gap-4 text-sm">
                                <span>{item.name}</span>
                                <span className={badgeClass(item.badge)}>{item.name.split(' ')[0]}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </motion.div>

            <motion.div {...fadeUp(0.3)} className={`${glassPanel} mb-6 bg-black/25`}>
                <div className={sectionHeading}>
                    <Cpu size={22} />
                    <h3>Technical Architecture</h3>
                </div>
                <div className="grid gap-4 lg:grid-cols-3">
                    <div className={innerPanel}>
                        <h4 className="mb-4 flex items-center gap-2 text-base"><Code size={16} /> Data Pipeline</h4>
                        <ul className="space-y-2 text-sm leading-6 text-slate-300">
                            <li>Feature selection & engineering</li>
                            <li>Missing value imputation</li>
                            <li>Leak-free patient splits</li>
                        </ul>
                    </div>
                    <div className={innerPanel}>
                        <h4 className="mb-4 flex items-center gap-2 text-base"><Cpu size={16} /> ML Models</h4>
                        <ul className="space-y-2 text-sm leading-6 text-slate-300">
                            <li>XGBoost, SVM, LightGBM</li>
                            <li>PubMedBERT, BioGPT, Clinical-T5</li>
                            <li>Stacking ensemble classifier</li>
                        </ul>
                    </div>
                    <div className={innerPanel}>
                        <h4 className="mb-4 flex items-center gap-2 text-base"><Brain size={16} /> RAG System</h4>
                        <ul className="space-y-2 text-sm leading-6 text-slate-300">
                            <li>Medical knowledge base</li>
                            <li>Automated report generation</li>
                            <li>Clinical recommendations</li>
                        </ul>
                    </div>
                </div>
            </motion.div>

            <motion.div {...fadeUp(0.4)} className={`${alertClass('warning')} mb-8`}>
                <AlertTriangle size={18} className="mt-0.5 shrink-0" />
                <div>
                    <strong>Disclaimer:</strong> This AI system provides diagnostic support for research and educational
                    purposes. It should not replace professional medical judgment. All predictions should be validated
                    by qualified healthcare providers.
                </div>
            </motion.div>
        </div>
    );
}
