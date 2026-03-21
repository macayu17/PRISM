import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
    Brain, Stethoscope, FileText, Shield, ArrowRight,
    Cpu, Activity, Zap, CheckCircle, Server
} from 'lucide-react';
import BrainScene from '../components/BrainScene';
import { getModelMetricsSummary, getSystemStatus } from '../api/client';
import {
    badgeClass,
    buttonPrimary,
    buttonSecondary,
    glassPanel,
    glassPanelInteractive,
    pageShell,
    sectionTitle,
} from '../lib/ui';

const fadeUp = (delay = 0) => ({
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6, delay, ease: 'easeOut' },
});

export default function HomePage() {
    const [status, setStatus] = useState(null);
    const [metrics, setMetrics] = useState(null);

    useEffect(() => {
        getSystemStatus().then(setStatus).catch(() => setStatus(null));
        getModelMetricsSummary().then(setMetrics).catch(() => setMetrics(null));
        const id = setInterval(() => {
            getSystemStatus().then(setStatus).catch(() => setStatus(null));
        }, 30000);
        return () => clearInterval(id);
    }, []);

    const bestTraditional = metrics?.best_traditional;
    const bestTransformer = metrics?.best_transformer;
    const trackedModels = metrics?.models?.length || 0;

    return (
        <div className={pageShell}>
            <section className="grid min-h-[calc(100vh-10rem)] items-center gap-10 py-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
                <div className="max-w-[32rem]">
                    <motion.div {...fadeUp(0)}>
                        <span className={sectionTitle}>AI-Powered Diagnostics</span>
                    </motion.div>
                    <motion.h1 {...fadeUp(0.1)} className="mb-6">
                        Parkinson&apos;s Disease
                        <br />
                        <span className="bg-gradient-to-r from-sky-300 via-sky-400 to-indigo-300 bg-clip-text text-transparent">
                            Assessment System
                        </span>
                    </motion.h1>
                    <motion.p {...fadeUp(0.2)} className="mb-10 max-w-xl text-lg leading-8 text-slate-400">
                        Advanced multimodal machine learning combining transformers and ensemble
                        methods for comprehensive diagnostic support and early detection.
                    </motion.p>
                    <motion.div {...fadeUp(0.3)} className="flex flex-wrap items-center gap-4">
                        <Link to="/assessment" className={buttonPrimary}>
                            <Stethoscope size={18} />
                            Start Assessment
                            <ArrowRight size={16} />
                        </Link>
                        <Link to="/about" className={buttonSecondary}>
                            Learn More
                        </Link>
                    </motion.div>
                </div>

                <motion.div
                    className="h-[360px] w-full overflow-hidden rounded-[2rem] border border-white/10 bg-black/30 shadow-[0_20px_60px_rgba(0,0,0,0.35)] md:h-[460px] lg:h-[520px]"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 1, delay: 0.3 }}
                >
                    <BrainScene />
                </motion.div>
            </section>

            <motion.section {...fadeUp(0.4)} className="grid gap-6 sm:grid-cols-2 xl:grid-cols-4">
                {[
                    {
                        label: bestTraditional ? `${bestTraditional.name} Accuracy` : 'Traditional Accuracy',
                        value: bestTraditional ? `${bestTraditional.accuracy_pct.toFixed(2)}%` : 'N/A',
                        icon: <Zap size={20} />,
                    },
                    {
                        label: bestTransformer ? `${bestTransformer.name} Accuracy` : 'Transformer Accuracy',
                        value: bestTransformer ? `${bestTransformer.accuracy_pct.toFixed(2)}%` : 'N/A',
                        icon: <Cpu size={20} />,
                    },
                    {
                        label: 'Tracked Models',
                        value: trackedModels > 0 ? String(trackedModels) : 'N/A',
                        icon: <Activity size={20} />,
                    },
                    { label: 'Diagnostic Classes', value: '4', icon: <Brain size={20} /> },
                ].map((stat) => (
                    <div key={stat.label} className={`${glassPanelInteractive} bg-black/30 p-5`}>
                        <div className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.04] text-white">
                            {stat.icon}
                        </div>
                        <div className="mb-1 text-3xl font-semibold tracking-[-0.03em] text-white">
                            {stat.value}
                        </div>
                        <div className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
                            {stat.label}
                        </div>
                    </div>
                ))}
            </motion.section>

            <motion.section {...fadeUp(0.5)} className="mt-12">
                <h2 className="mb-2 text-center">
                    Why{' '}
                    <span className="bg-gradient-to-r from-sky-300 via-sky-400 to-indigo-300 bg-clip-text text-transparent">
                        NeuroAssess
                    </span>
                    ?
                </h2>
                <p className="mx-auto mb-8 max-w-2xl text-center text-base text-slate-400">
                    Combining cutting-edge AI with clinical research standards
                </p>
                <div className="grid gap-6 lg:grid-cols-3">
                    {[
                        {
                            icon: <Cpu size={28} />,
                            title: 'AI-Powered Analysis',
                            desc: 'Multimodal ensemble combining XGBoost, SVM, and medical transformers (PubMedBERT, BioGPT, Clinical-T5).',
                            items: [
                                bestTraditional
                                    ? `${bestTraditional.name}: ${bestTraditional.accuracy_pct.toFixed(2)}%`
                                    : 'Latest metrics available in evaluation reports',
                                bestTransformer
                                    ? `${bestTransformer.name}: ${bestTransformer.accuracy_pct.toFixed(2)}%`
                                    : 'Transformer metrics available in evaluation reports',
                                'Real-time Predictions',
                            ],
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
                    ].map((feature) => (
                        <div key={feature.title} className={`${glassPanelInteractive} flex h-full flex-col bg-black/25`}>
                            <div className="mb-5 inline-flex h-11 w-11 items-center justify-center rounded-[1.25rem] border border-white/10 bg-black/40 text-white">
                                {feature.icon}
                            </div>
                            <h4 className="mb-2">{feature.title}</h4>
                            <p className="mb-5 text-sm leading-7 text-slate-400">{feature.desc}</p>
                            <ul className="mt-auto space-y-3">
                                {feature.items.map((item) => (
                                    <li key={item} className="flex items-center gap-2 text-sm text-slate-400">
                                        <CheckCircle size={14} className="shrink-0 text-sky-300" />
                                        <span>{item}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </div>
            </motion.section>

            <motion.section {...fadeUp(0.6)} className="mt-12">
                <h2 className="mb-8 text-center">Diagnostic Categories</h2>
                <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
                    {[
                        { label: 'Healthy Control', abbr: 'HC', borderClass: 'border-l-emerald-500', desc: 'No signs of movement disorders' },
                        { label: "Parkinson's Disease", abbr: 'PD', borderClass: 'border-l-rose-500', desc: 'Characteristic motor symptoms' },
                        { label: 'SWEDD', abbr: 'SWEDD', borderClass: 'border-l-amber-500', desc: 'Symptoms without dopamine deficit' },
                        { label: 'Prodromal PD', abbr: 'PROD', borderClass: 'border-l-blue-500', desc: 'Early stage, subtle symptoms' },
                    ].map((category) => (
                        <div
                            key={category.label}
                            className={`${glassPanelInteractive} ${category.borderClass} border-l-4 bg-black/25 p-5`}
                        >
                            <div className="mb-3 flex items-start justify-between gap-3">
                                <h4 className="text-lg">{category.label}</h4>
                                <span className={badgeClass('accent')}>{category.abbr}</span>
                            </div>
                            <p className="text-sm text-slate-400">{category.desc}</p>
                        </div>
                    ))}
                </div>
            </motion.section>

            <motion.section {...fadeUp(0.7)} className="my-12">
                <div className={`${glassPanel} flex flex-col gap-4 bg-black/25 sm:flex-row sm:items-center`}>
                    <Server size={24} className={status?.system_initialized ? 'text-emerald-300' : 'text-rose-300'} />
                    <div className="flex-1">
                        <h4 className="mb-1 text-base">System Status</h4>
                        <p className="m-0 text-sm text-slate-400">
                            {status?.system_initialized && status?.models_loaded
                                ? 'All systems operational. Ready for assessments.'
                                : status?.system_initialized
                                    ? 'System initializing. Models loading...'
                                    : 'Connect Flask backend on port 5000 to enable predictions.'}
                        </p>
                    </div>
                    <span className={badgeClass(status?.models_loaded ? 'success' : 'warning')}>
                        {status?.models_loaded ? 'Online' : 'Offline'}
                    </span>
                </div>
            </motion.section>
        </div>
    );
}
